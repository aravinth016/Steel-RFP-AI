"""Single-supplier LangGraph evaluation flow with bounded, auditable stages."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from src.evaluator import EvaluationError, evaluate_supplier
from src.evidence import verify_evidence
from src.normalization import validate_evaluation
from src.scoring import weighted_score


class EvaluationState(TypedDict, total=False):
    run_id: str
    api_key: str
    model: str
    provider: str
    supplier: dict[str, Any]
    criteria: list[dict[str, Any]]
    shared_documents: list[dict[str, Any]]
    supplier_documents: list[dict[str, Any]]
    use_rag: bool
    buyer_requirements: dict[str, Any]
    cached_evaluation: dict[str, Any] | None
    token_usage: dict[str, int | None]
    evaluation: dict[str, Any]
    evidence_checks: list[dict[str, Any]]
    warnings: list[str]
    traces: list[dict[str, Any]]
    audit_events: list[dict[str, Any]]
    weighted_score: float | None
    error: str | None
    status: str
    record: dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_stage(
    state: EvaluationState,
    stage: str,
    status: str,
    details: dict[str, Any],
    started_at: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed_at = _now()
    traces = list(state.get("traces", []))
    audit = list(state.get("audit_events", []))
    supplier_id = state["supplier"]["supplier_id"]
    traces.append({
        "supplier_id": supplier_id,
        "sequence": len(traces) + 1,
        "node": stage,
        "status": status,
        "started_at": started_at or completed_at,
        "completed_at": completed_at,
        "details": details,
    })
    audit.append({
        "supplier_id": supplier_id,
        "event_type": f"{stage}_{status}",
        "event_at": completed_at,
        "details": details,
    })
    return traces, audit


def _prepare(state: EvaluationState) -> dict[str, Any]:
    started_at = _now()
    details = {
        "provider": state["provider"],
        "model": state["model"],
        "criterion_count": sum(
            bool(item.get("active", True)) for item in state["criteria"]
        ),
        "document_count": len(state["shared_documents"]) + len(state["supplier_documents"]),
    }
    traces, audit = _record_stage(
        state, "prepare", "complete", details, started_at=started_at
    )
    update: dict[str, Any] = {"traces": traces, "audit_events": audit}
    if state.get("cached_evaluation") is not None:
        stage_state = {**state, "traces": traces, "audit_events": audit}
        traces, audit = _record_stage(
            stage_state,
            "evaluate",
            "skipped",
            {"reason": "validated_result_reused"},
        )
        update.update({
            "evaluation": state["cached_evaluation"],
            "traces": traces,
            "audit_events": audit,
            "status": "evaluated",
        })
    else:
        update["status"] = "prepared"
    return update


def _evaluate(state: EvaluationState) -> dict[str, Any]:
    started_at = _now()
    try:
        evaluation = evaluate_supplier(
            api_key=state["api_key"],
            model=state["model"],
            supplier=state["supplier"],
            criteria=state["criteria"],
            shared_documents=state["shared_documents"],
            supplier_documents=state["supplier_documents"],
            use_rag=state["use_rag"],
            provider=state["provider"],
            buyer_requirements=state["buyer_requirements"],
        )
    except EvaluationError as exc:
        message = str(exc)
        traces, audit = _record_stage(
            state,
            "evaluate",
            "failed",
            {"error_type": type(exc).__name__},
            started_at=started_at,
        )
        return {
            "error": message,
            "status": "failed",
            "traces": traces,
            "audit_events": audit,
        }
    except Exception as exc:
        message = f"Evaluation stopped ({type(exc).__name__})."
        traces, audit = _record_stage(
            state,
            "evaluate",
            "failed",
            {"error_type": type(exc).__name__},
            started_at=started_at,
        )
        return {
            "error": message,
            "status": "failed",
            "traces": traces,
            "audit_events": audit,
        }
    traces, audit = _record_stage(
        state, "evaluate", "complete", {}, started_at=started_at
    )
    return {
        "evaluation": evaluation,
        "token_usage": evaluation.get("token_usage", {}),
        "status": "evaluated",
        "traces": traces,
        "audit_events": audit,
    }


def _validate(state: EvaluationState) -> dict[str, Any]:
    started_at = _now()
    try:
        evaluation_with_metadata = state["evaluation"]
        metadata_keys = {"validation_warnings", "provider", "model", "token_usage"}
        metadata = {
            key: value
            for key, value in evaluation_with_metadata.items()
            if key in metadata_keys
        }
        schema_payload = {
            key: value
            for key, value in evaluation_with_metadata.items()
            if key not in metadata_keys
        }
        evaluation = validate_evaluation(schema_payload, state["criteria"])
        evaluation.update(metadata)
    except (ValidationError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            issues = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in exc.errors(include_input=False)
            )
        else:
            issues = str(exc)
        message = f"Strict response validation failed: {issues}"
        traces, audit = _record_stage(
            state,
            "validate",
            "failed",
            {"error_type": type(exc).__name__},
            started_at=started_at,
        )
        return {
            "error": message,
            "status": "failed",
            "traces": traces,
            "audit_events": audit,
        }
    traces, audit = _record_stage(
        state, "validate", "complete", {}, started_at=started_at
    )
    return {
        "evaluation": evaluation,
        "status": "validated",
        "traces": traces,
        "audit_events": audit,
    }


def _verify(state: EvaluationState) -> dict[str, Any]:
    started_at = _now()
    checks = verify_evidence(
        state["evaluation"],
        state["shared_documents"] + state["supplier_documents"],
    )
    warnings = list(state.get("warnings", []))
    warnings.extend(
        f"Evidence citation for {check['evidence_type']} {check['target_id']} could not be verified."
        for check in checks
        if check["status"] == "NOT_VERIFIED"
    )
    counts = {
        "verified": sum(check["status"] == "VERIFIED" for check in checks),
        "not_provided": sum(check["status"] == "NOT_PROVIDED" for check in checks),
        "not_verified": sum(check["status"] == "NOT_VERIFIED" for check in checks),
    }
    traces, audit = _record_stage(
        state, "verify_evidence", "complete", counts, started_at=started_at
    )
    return {
        "evidence_checks": checks,
        "warnings": warnings,
        "status": "evidence_checked",
        "traces": traces,
        "audit_events": audit,
    }


def _score(state: EvaluationState) -> dict[str, Any]:
    started_at = _now()
    try:
        value = weighted_score(
            state["criteria"],
            {
                item["criterion_id"]: item["score"]
                for item in state["evaluation"]["criteria"]
            },
        )
    except (TypeError, ValueError) as exc:
        message = f"Deterministic scoring failed ({type(exc).__name__})."
        traces, audit = _record_stage(
            state,
            "score",
            "failed",
            {"error_type": type(exc).__name__},
            started_at=started_at,
        )
        return {
            "error": message,
            "status": "failed",
            "traces": traces,
            "audit_events": audit,
        }
    traces, audit = _record_stage(
        state, "score", "complete", {}, started_at=started_at
    )
    return {
        "weighted_score": value,
        "status": "scored",
        "traces": traces,
        "audit_events": audit,
    }


def _persist(state: EvaluationState) -> dict[str, Any]:
    started_at = _now()
    traces, audit = _record_stage(
        state,
        "persist",
        "ready" if state.get("error") is None else "failed",
        {"warning_count": len(state.get("warnings", []))},
        started_at=started_at,
    )
    record = {
        "run_id": state["run_id"],
        "supplier_id": state["supplier"]["supplier_id"],
        "status": "complete" if state.get("error") is None else "failed",
        "evaluation": state.get("evaluation"),
        "weighted_score": state.get("weighted_score"),
        "token_usage": state.get("token_usage", {}),
        "evidence_checks": state.get("evidence_checks", []),
        "warnings": state.get("warnings", []),
        "error": state.get("error"),
        "traces": traces,
        "audit_events": audit,
    }
    return {"record": record, "traces": traces, "audit_events": audit}


def _after_evaluate(state: EvaluationState) -> str:
    return "persist" if state.get("error") else "validate"


def _after_prepare(state: EvaluationState) -> str:
    return "validate" if state.get("evaluation") is not None else "evaluate"


def _after_validate(state: EvaluationState) -> str:
    return "persist" if state.get("error") else "verify_evidence"


def _after_evidence(state: EvaluationState) -> str:
    return "persist" if state.get("error") else "score"


def _build_graph():
    graph = StateGraph(EvaluationState)
    graph.add_node("prepare", _prepare)
    graph.add_node("evaluate", _evaluate)
    graph.add_node("validate", _validate)
    graph.add_node("verify_evidence", _verify)
    graph.add_node("score", _score)
    graph.add_node("persist", _persist)
    graph.set_entry_point("prepare")
    graph.add_conditional_edges(
        "prepare", _after_prepare, {"evaluate": "evaluate", "validate": "validate"}
    )
    graph.add_conditional_edges("evaluate", _after_evaluate, {"validate": "validate", "persist": "persist"})
    graph.add_conditional_edges("validate", _after_validate, {"verify_evidence": "verify_evidence", "persist": "persist"})
    graph.add_conditional_edges("verify_evidence", _after_evidence, {"score": "score", "persist": "persist"})
    graph.add_edge("score", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


_EVALUATION_GRAPH = _build_graph()


def run_supplier_evaluation(
    *,
    run_id: str,
    api_key: str,
    model: str,
    provider: str,
    supplier: dict[str, Any],
    criteria: list[dict[str, Any]],
    shared_documents: list[dict[str, Any]],
    supplier_documents: list[dict[str, Any]],
    use_rag: bool,
    buyer_requirements: dict[str, Any],
    cached_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the graph for one supplier, isolating its failure from other suppliers."""
    state = _EVALUATION_GRAPH.invoke({
        "run_id": run_id,
        "api_key": api_key,
        "model": model,
        "provider": provider,
        "supplier": supplier,
        "criteria": criteria,
        "shared_documents": shared_documents,
        "supplier_documents": supplier_documents,
        "use_rag": use_rag,
        "buyer_requirements": buyer_requirements,
        "cached_evaluation": cached_evaluation,
        "warnings": [],
        "traces": [],
        "audit_events": [],
        "status": "started",
    })
    return state["record"]


__all__ = ["run_supplier_evaluation"]
