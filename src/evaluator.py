"""Provider-backed structured evaluation. Weighted scoring remains in Python."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any

from jsonschema import Draft202012Validator

from src.llm_providers import call_llm


class EvaluationError(RuntimeError):
    """An evaluation could not be completed safely."""


_COMPACT_RETRY_INSTRUCTIONS = (
    "\n\nCOMPACT RETRY: Return only the required JSON fields. Keep summary, reasons, and "
    "justifications to one short sentence (12 words maximum). Use one short exact evidence "
    "quote per assessed criterion and one evidence quote per extracted section. Set risks and "
    "clarifications to [] unless essential. Do not repeat document contents."
)


def evaluate_supplier(
    *,
    api_key: str,
    model: str,
    supplier: dict[str, Any],
    criteria: list[dict[str, Any]],
    shared_documents: list[dict[str, Any]],
    supplier_documents: list[dict[str, Any]],
    use_rag: bool = True,
    provider: str = "Cohere.ai",
    mandatory_gates: list[dict[str, str]] | None = None,
    buyer_requirements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        prompt, permitted_pages = _build_prompt(
            supplier,
            criteria,
            shared_documents,
            supplier_documents,
            use_rag,
            mandatory_gates,
            buyer_requirements,
        )
        retry_warnings = []
        try:
            response_text = call_llm(provider, api_key, model, prompt)
        except RuntimeError as exc:
            if provider not in ("Cohere", "Cohere.ai") or "output token limit" not in str(exc):
                raise
            try:
                response_text = call_llm(
                    provider,
                    api_key,
                    model,
                    prompt + _COMPACT_RETRY_INSTRUCTIONS,
                )
            except RuntimeError as retry_exc:
                if "output token limit" not in str(retry_exc):
                    raise
                raise RuntimeError(
                    "Cohere's automatic compact retry also reached the output token limit. "
                    "Reduce the documents submitted for this supplier or choose another model."
                ) from retry_exc
            retry_warnings.append(
                "Cohere reached the output token limit; evaluation was retried once with "
                "compact output instructions."
            )
        raw_result = _parse_json_response(response_text)
        result, validation_warnings = _normalize_result(
            raw_result, criteria, mandatory_gates
        )
        validation_warnings[:0] = retry_warnings
        validation_warnings.extend(
            _clear_unverifiable_citations(
                result,
                supplier_documents,
                shared_documents,
                permitted_pages,
            )
        )
        _validate_result(
            result,
            criteria,
            supplier_documents,
            shared_documents,
            permitted_pages,
        )
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationError(
            f"LLM evaluation failed: {exc}. No valid assessment was produced for this supplier."
        ) from exc
    result["validation_warnings"] = validation_warnings
    result["provider"] = provider
    result["model"] = model
    return result


def _parse_json_response(response_text: str) -> dict[str, Any]:
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        result = None
        for index, character in enumerate(response_text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(response_text[index:])
            except json.JSONDecodeError:
                continue
            if (
                isinstance(candidate, dict)
                and isinstance(candidate.get("gates"), list)
                and isinstance(candidate.get("criteria"), list)
            ):
                result = candidate
                break
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("gates"), list)
        or not isinstance(result.get("criteria"), list)
    ):
        raise ValueError("The selected LLM did not return a valid JSON evaluation object.")
    return result


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _clear_unverifiable_citations(
    result: dict[str, Any],
    supplier_documents: list[dict[str, Any]],
    shared_documents: list[dict[str, Any]],
    permitted_pages: dict[str, set[int]],
) -> list[str]:
    sources = {
        document["id"]: document
        for document in shared_documents + supplier_documents
    }
    supplier_source_ids = {document["id"] for document in supplier_documents}

    def valid_citation(
        source_id: Any,
        page_number: Any,
        quote: Any,
        *,
        supplier_only: bool = False,
    ) -> bool:
        if source_id is None:
            return not quote
        if (
            not isinstance(source_id, str)
            or source_id not in sources
            or (supplier_only and source_id not in supplier_source_ids)
            or isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number not in permitted_pages.get(source_id, set())
        ):
            return False
        if not quote:
            return True
        source_page = next(
            (
                page
                for page in sources[source_id]["pages"]
                if int(page["page"]) == page_number
            ),
            None,
        )
        return source_page is not None and _evidence_quote_matches(
            str(quote), str(source_page["text"])
        )

    warnings: list[str] = []
    for item in result["criteria"]:
        if item.get("source_id") is not None or item.get("evidence"):
            if not valid_citation(
                item.get("source_id"), item.get("page"), item.get("evidence")
            ):
                item.update({
                    "score": 0.0,
                    "status": "NOT_VERIFIABLE",
                    "justification": (
                        "The model's evidence citation could not be verified against the "
                        "cited source page; human review is required."
                    ),
                    "evidence": None,
                    "source_id": None,
                    "page": None,
                })
                warnings.append(
                    f"Criterion {item['criterion_id']!r} was set to NOT_VERIFIABLE because its evidence citation could not be verified."
                )

    for gate in result["gates"]:
        if gate.get("source_id") is not None or gate.get("evidence"):
            if not valid_citation(
                gate.get("source_id"), gate.get("page"), gate.get("evidence")
            ):
                gate.update({
                    "status": "Pending",
                    "reason": (
                        "The model's evidence citation could not be verified; human review "
                        "is required."
                    ),
                    "evidence": None,
                    "source_id": None,
                    "page": None,
                })
                warnings.append(
                    f"Mandatory gate {gate['id']!r} was marked Pending because its evidence citation could not be verified."
                )

    technical = result["technical_offer"]
    if any(
        not valid_citation(
            item.get("source_id"),
            item.get("page"),
            item.get("quote"),
            supplier_only=True,
        )
        for item in technical["evidence"]
    ):
        technical.update({
            "material_description": None,
            "offered_grade": None,
            "offered_specification": None,
            "offered_quantity_mt": None,
            "deviations": [],
            "compliance_status": "NOT_VERIFIABLE",
            "evidence": [],
        })
        warnings.append(
            "Technical offer details were cleared because their evidence citation could not be verified."
        )

    for quality_document in result["quality_documents"]:
        if quality_document.get("source_id") is not None or quality_document.get("evidence_quote"):
            if not valid_citation(
                quality_document.get("source_id"),
                quality_document.get("page"),
                quality_document.get("evidence_quote"),
                supplier_only=True,
            ):
                quality_document.update({
                    "document_type": "Unverified quality document",
                    "provided": None,
                    "heat_or_batch_reference": None,
                    "source_id": None,
                    "page": None,
                    "evidence_quote": None,
                    "verification_status": "NOT_VERIFIABLE",
                })
                warnings.append(
                    "A quality-document result was left unknown because its evidence citation could not be verified."
                )

    for section_name, fields in (
        (
            "commercial_quote",
            (
                "currency", "basic_rate_per_mt", "freight_per_mt", "tax_amount",
                "loading_unloading_amount", "other_charges", "total_quoted_amount",
                "delivery_basis", "quote_valid_until", "payment_terms", "exclusions",
            ),
        ),
        (
            "delivery",
            (
                "quantity_available_mt", "first_delivery_date", "completion_date",
                "delivery_phases", "delivery_location",
            ),
        ),
    ):
        section = result[section_name]
        if any(
            not valid_citation(
                item.get("source_id"),
                item.get("page"),
                item.get("quote"),
                supplier_only=True,
            )
            for item in section["evidence"]
        ):
            for field in fields:
                section[field] = [] if field in ("exclusions", "delivery_phases") else None
            if section_name == "commercial_quote":
                section["comparability_status"] = "PENDING_REVIEW"
            section["evidence"] = []
            warnings.append(
                f"{section_name.replace('_', ' ').capitalize()} details were cleared because their evidence citation could not be verified."
            )
    return warnings


def _normalize_result(
    result: Any,
    criteria: list[dict[str, Any]],
    mandatory_gates: list[dict[str, str]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(result, dict):
        raise EvaluationError("The selected LLM response must be a JSON object.")

    warnings: list[str] = []
    summary = result.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = "The model did not provide an overall summary."
        warnings.append("The model summary was missing or malformed.")

    known = {item["id"]: item for item in criteria if item.get("active", True)}
    supplied: dict[str, dict[str, Any]] = {}
    raw_criteria = result.get("criteria")
    if not isinstance(raw_criteria, list):
        raw_criteria = []
        warnings.append("The model criteria list was missing or malformed.")

    for index, raw_item in enumerate(raw_criteria, start=1):
        if not isinstance(raw_item, dict):
            warnings.append(f"Criterion result {index} was malformed and ignored.")
            continue
        criterion_id = raw_item.get("criterion_id")
        if not isinstance(criterion_id, str) or criterion_id not in known:
            warnings.append(f"Unknown or inactive criterion result {criterion_id!r} was ignored.")
            continue
        if criterion_id in supplied:
            warnings.append(f"Duplicate result for criterion {criterion_id!r} was ignored.")
            continue
        supplied[criterion_id] = raw_item

    normalized_criteria = []
    statuses = {
        "Compliant",
        "Partially compliant",
        "Non-compliant",
        "Not stated",
        "Ambiguous",
        "COMPLIANT",
        "PARTIALLY_COMPLIANT",
        "NON_COMPLIANT",
        "NOT_PROVIDED",
        "NOT_VERIFIABLE",
        "NOT_APPLICABLE",
        "PENDING_REVIEW",
        "NOT_COMPARABLE",
    }
    for criterion_id, criterion in known.items():
        raw_item = supplied.get(criterion_id)
        if raw_item is None:
            normalized_criteria.append({
                "criterion_id": criterion_id,
                "score": 0.0,
                "status": "NOT_PROVIDED",
                "justification": "No valid result was returned for this active criterion.",
                "evidence": None,
                "source_id": None,
                "page": None,
                "risks": ["The model omitted this active criterion; reviewer verification is required."],
                "clarifications": [],
            })
            warnings.append(f"Missing result for active criterion {criterion_id!r}; score set to 0.")
            continue

        raw_score = raw_item.get("score")
        try:
            if isinstance(raw_score, bool) or raw_score is None:
                raise ValueError
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError
        except (TypeError, ValueError):
            score = 0.0
            status = "NOT_VERIFIABLE"
            warnings.append(f"Malformed score for criterion {criterion_id!r}; score set to 0.")
        else:
            status = raw_item.get("status")
            if not isinstance(status, str) or status not in statuses:
                status = "NOT_VERIFIABLE"
                warnings.append(f"Malformed status for criterion {criterion_id!r}; marked NOT_VERIFIABLE.")
            elif status in {"Compliant", "Partially compliant", "Non-compliant", "Not stated", "Ambiguous"}:
                status = {
                    "Compliant": "COMPLIANT",
                    "Partially compliant": "PARTIALLY_COMPLIANT",
                    "Non-compliant": "NON_COMPLIANT",
                    "Not stated": "NOT_PROVIDED",
                    "Ambiguous": "NOT_VERIFIABLE",
                }[status]
        maximum = float(criterion["max_score"])
        if score < 0 or score > maximum:
            clipped_score = min(max(score, 0.0), maximum)
            warnings.append(
                f"Out-of-range score for criterion {criterion_id!r} was clipped to {clipped_score:g}."
            )
            score = clipped_score

        justification = raw_item.get("justification")
        if not isinstance(justification, str) or not justification.strip():
            justification = "The model did not provide a valid justification."
            warnings.append(f"Missing or malformed justification for criterion {criterion_id!r}.")
        evidence = raw_item.get("evidence")
        source_id = raw_item.get("source_id")
        page = raw_item.get("page")
        if evidence is not None and not isinstance(evidence, str):
            evidence, source_id, page = None, None, None
            warnings.append(f"Malformed evidence citation for criterion {criterion_id!r} was cleared.")
        if source_id is not None and not isinstance(source_id, str):
            evidence, source_id, page = None, None, None
            warnings.append(f"Malformed source citation for criterion {criterion_id!r} was cleared.")
        if page is not None and (isinstance(page, bool) or not isinstance(page, int)):
            evidence, source_id, page = None, None, None
            warnings.append(f"Malformed page citation for criterion {criterion_id!r} was cleared.")
        risks = raw_item.get("risks", [])
        clarifications = raw_item.get("clarifications", [])
        if not isinstance(risks, list):
            risks = []
            warnings.append(f"Malformed risks for criterion {criterion_id!r} were cleared.")
        if not isinstance(clarifications, list):
            clarifications = []
            warnings.append(f"Malformed clarifications for criterion {criterion_id!r} were cleared.")
        normalized_criteria.append({
            **raw_item,
            "criterion_id": criterion_id,
            "score": score,
            "status": status,
            "justification": justification,
            "evidence": evidence,
            "source_id": source_id,
            "page": page,
            "risks": [item for item in risks if isinstance(item, str)],
            "clarifications": [item for item in clarifications if isinstance(item, str)],
        })

    configured_gates = mandatory_gates or []
    raw_gates = result.get("gates", [])
    if not isinstance(raw_gates, list):
        raw_gates = []
        warnings.append("The model gates list was malformed and cleared.")
    raw_gates_by_id = {
        gate.get("id"): gate
        for gate in raw_gates
        if isinstance(gate, dict) and isinstance(gate.get("id"), str)
    }
    configured_gate_ids = {gate["id"] for gate in configured_gates}
    if any(gate_id not in configured_gate_ids for gate_id in raw_gates_by_id):
        warnings.append("Model-invented mandatory gates were discarded because they were not configured from the RFP.")
    if len(raw_gates_by_id) != len(raw_gates):
        warnings.append("Malformed or duplicate mandatory-gate results were ignored.")
    gates = []
    gate_statuses = {
        "Pass": "Pass",
        "Fail": "Fail",
        "Pending": "Pending",
        "PASS": "Pass",
        "FAIL": "Fail",
        "PENDING_REVIEW": "Pending",
    }
    for configured in configured_gates:
        gate_id = configured["id"]
        raw_gate = raw_gates_by_id.get(gate_id)
        if raw_gate is None:
            gate = {
                "id": gate_id,
                "requirement": configured["requirement"],
                "status": "Pending",
                "reason": "No model assessment was returned; human review is required.",
                "evidence": None,
                "source_id": None,
                "page": None,
            }
            warnings.append(f"Missing assessment for configured mandatory gate {gate_id!r}.")
            gates.append(gate)
            continue
        gate = raw_gate
        gate_status = gate_statuses.get(gate.get("status"))
        if gate_status is None:
            gate_status = "Pending"
            warnings.append(f"Malformed status for gate {gate_id!r}; marked Pending.")
        evidence = gate.get("evidence")
        source_id = gate.get("source_id")
        page = gate.get("page")
        if evidence is not None and not isinstance(evidence, str):
            evidence, source_id, page = None, None, None
            warnings.append(f"Malformed evidence citation for gate {gate_id!r} was cleared.")
        if source_id is not None and not isinstance(source_id, str):
            evidence, source_id, page = None, None, None
            warnings.append(f"Malformed source citation for gate {gate_id!r} was cleared.")
        if page is not None and (isinstance(page, bool) or not isinstance(page, int)):
            evidence, source_id, page = None, None, None
            warnings.append(f"Malformed page citation for gate {gate_id!r} was cleared.")
        gates.append({
            **gate,
            "id": gate_id,
            "status": gate_status,
            "requirement": configured["requirement"],
            "reason": (
                gate.get("reason")
                if isinstance(gate.get("reason"), str) and gate["reason"].strip()
                else "No valid explanation was returned; human review is required."
            ),
            "evidence": evidence,
            "source_id": source_id,
            "page": page,
        })

    steel_offer, steel_warnings = _normalize_steel_offer(result)
    warnings.extend(steel_warnings)
    normalized = {
        **result,
        "summary": summary,
        "gates": gates,
        "criteria": normalized_criteria,
        **steel_offer,
    }
    return normalized, warnings


def _normalize_steel_offer(result: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []

    def section(name: str) -> dict[str, Any]:
        value = result.get(name)
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        warnings.append(f"Malformed steel extraction section {name!r}; values left unknown.")
        return {}

    def text_value(data: dict[str, Any], key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            warnings.append(f"Malformed extracted field {key!r}; value left unknown.")
            return None
        return value.strip() or None

    def number_value(data: dict[str, Any], key: str) -> float | None:
        value = data.get(key)
        if value is None:
            return None
        try:
            if isinstance(value, bool):
                raise ValueError
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError
            return number
        except (TypeError, ValueError):
            warnings.append(f"Malformed extracted numeric field {key!r}; value left unknown.")
            return None

    def string_list(data: dict[str, Any], key: str) -> list[str]:
        value = data.get(key, [])
        if not isinstance(value, list):
            warnings.append(f"Malformed extracted list {key!r}; list cleared.")
            return []
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    def evidence_list(data: dict[str, Any], key: str = "evidence") -> list[dict[str, Any]]:
        value = data.get(key, [])
        if not isinstance(value, list):
            warnings.append(f"Malformed evidence list {key!r}; list cleared.")
            return []
        evidence = []
        for index, item in enumerate(value, start=1):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("source_id"), str)
                or isinstance(item.get("page"), bool)
                or not isinstance(item.get("page"), int)
                or not isinstance(item.get("quote"), str)
                or not item["quote"].strip()
            ):
                warnings.append(f"Malformed extracted evidence {key}[{index}]; entry ignored.")
                continue
            evidence.append({
                "source_id": item["source_id"],
                "page": item["page"],
                "quote": item["quote"].strip(),
            })
        return evidence

    technical = section("technical_offer")
    compliance = technical.get("compliance_status")
    valid_statuses = {
        "COMPLIANT",
        "PARTIALLY_COMPLIANT",
        "NON_COMPLIANT",
        "NOT_PROVIDED",
        "NOT_VERIFIABLE",
        "NOT_APPLICABLE",
        "PENDING_REVIEW",
    }
    if not isinstance(compliance, str) or compliance not in valid_statuses:
        compliance = "PENDING_REVIEW"
        if technical.get("compliance_status") is not None:
            warnings.append("Malformed technical compliance status; marked PENDING_REVIEW.")
    technical_offer = {
        "material_description": text_value(technical, "material_description"),
        "offered_grade": text_value(technical, "offered_grade"),
        "offered_specification": text_value(technical, "offered_specification"),
        "offered_quantity_mt": number_value(technical, "offered_quantity_mt"),
        "deviations": string_list(technical, "deviations"),
        "compliance_status": compliance,
        "evidence": evidence_list(technical),
    }

    quality_raw = result.get("quality_documents", [])
    quality_documents = []
    if not isinstance(quality_raw, list):
        quality_raw = []
        warnings.append("Malformed quality_documents list; list cleared.")
    for index, item in enumerate(quality_raw, start=1):
        if not isinstance(item, dict):
            warnings.append(f"Malformed quality document {index}; entry ignored.")
            continue
        provided = item.get("provided")
        if not isinstance(provided, bool) and provided is not None:
            provided = None
            warnings.append(f"Malformed provided status for quality document {index}; left unknown.")
        quality_documents.append({
            "document_type": text_value(item, "document_type") or "Unspecified quality document",
            "provided": provided,
            "heat_or_batch_reference": text_value(item, "heat_or_batch_reference"),
            "source_id": item.get("source_id") if isinstance(item.get("source_id"), str) else None,
            "page": item.get("page") if isinstance(item.get("page"), int) and not isinstance(item.get("page"), bool) else None,
            "evidence_quote": text_value(item, "evidence_quote"),
            "verification_status": "NOT_INDEPENDENTLY_VERIFIED",
        })
        if (
            quality_documents[-1]["source_id"] is None
            or quality_documents[-1]["page"] is None
            or quality_documents[-1]["evidence_quote"] is None
        ):
            if any(item.get(key) is not None for key in ("source_id", "page", "evidence_quote")):
                warnings.append(f"Incomplete quality-document citation {index}; citation was cleared.")
            quality_documents[-1].update({
                "source_id": None,
                "page": None,
                "evidence_quote": None,
            })

    commercial = section("commercial_quote")
    comparability = commercial.get("comparability_status")
    commercial_statuses = {"COMPARABLE", "NOT_COMPARABLE", "PENDING_REVIEW", "NOT_PROVIDED"}
    if isinstance(comparability, str):
        comparability = comparability.strip().upper().replace(" ", "_").replace("-", "_")
    if not isinstance(comparability, str) or comparability not in commercial_statuses:
        comparability = "PENDING_REVIEW"
        if commercial.get("comparability_status") is not None:
            warnings.append("Malformed commercial comparability status; marked PENDING_REVIEW.")
    commercial_quote = {
        "currency": text_value(commercial, "currency"),
        "basic_rate_per_mt": number_value(commercial, "basic_rate_per_mt"),
        "freight_per_mt": number_value(commercial, "freight_per_mt"),
        "tax_amount": number_value(commercial, "tax_amount"),
        "loading_unloading_amount": number_value(commercial, "loading_unloading_amount"),
        "other_charges": number_value(commercial, "other_charges"),
        "total_quoted_amount": number_value(commercial, "total_quoted_amount"),
        "delivery_basis": text_value(commercial, "delivery_basis"),
        "quote_valid_until": text_value(commercial, "quote_valid_until"),
        "payment_terms": text_value(commercial, "payment_terms"),
        "exclusions": string_list(commercial, "exclusions"),
        "comparability_status": comparability,
        "evidence": evidence_list(commercial),
    }

    delivery = section("delivery")
    delivery_data = {
        "quantity_available_mt": number_value(delivery, "quantity_available_mt"),
        "first_delivery_date": text_value(delivery, "first_delivery_date"),
        "completion_date": text_value(delivery, "completion_date"),
        "delivery_phases": string_list(delivery, "delivery_phases"),
        "delivery_location": text_value(delivery, "delivery_location"),
        "evidence": evidence_list(delivery),
    }
    return {
        "technical_offer": technical_offer,
        "quality_documents": quality_documents,
        "commercial_quote": commercial_quote,
        "delivery": delivery_data,
    }, warnings


def _build_prompt(
    supplier: dict[str, Any],
    criteria: list[dict[str, Any]],
    shared_documents: list[dict[str, Any]],
    supplier_documents: list[dict[str, Any]],
    use_rag: bool,
    mandatory_gates: list[dict[str, str]] | None = None,
    buyer_requirements: dict[str, Any] | None = None,
) -> tuple[str, dict[str, set[int]]]:
    mandatory_gates = mandatory_gates or []
    active_criteria = [item for item in criteria if item.get("active", True)]
    if use_rag:
        retrieval_by_criterion = {
            criterion["id"]: {
                "requirements": _retrieve_pages(
                    " ".join((criterion["name"], criterion["description"], criterion["rubric"])),
                    shared_documents,
                    limit=2,
                ),
                "supplier": _retrieve_pages(
                    " ".join((criterion["name"], criterion["description"], criterion["rubric"])),
                    supplier_documents,
                    limit=2,
                ),
            }
            for criterion in active_criteria
        }
        retrieval_by_criterion["mandatory_gates"] = {
            "requirements": _retrieve_pages(
                "mandatory required eligibility certificate test statutory safety quality gate",
                shared_documents,
                limit=2,
            ),
            "supplier": _retrieve_pages(
                "mandatory required eligibility certificate test statutory safety quality gate",
                supplier_documents,
                limit=2,
            ),
        }
        retrieved_pages: dict[tuple[str, int], dict[str, Any]] = {}
        for groups in retrieval_by_criterion.values():
            for pages in groups.values():
                for page in pages:
                    ref = (page["source_id"], int(page["page"]))
                    retrieved_pages.setdefault(ref, page)
        selected_pages = list(retrieved_pages.values())[:24]
        selected_refs = {(page["source_id"], int(page["page"])) for page in selected_pages}
        for groups in retrieval_by_criterion.values():
            for scope in groups:
                groups[scope] = [
                    page for page in groups[scope]
                    if (page["source_id"], int(page["page"])) in selected_refs
                ]
        retrieved_refs = {
            "criterion_sources": {
                criterion_id: {
                    scope: [
                        [page["source_id"], int(page["page"])]
                        for page in pages
                    ]
                    for scope, pages in groups.items()
                }
                for criterion_id, groups in retrieval_by_criterion.items()
            },
            "evidence_pages": [
                {
                    "source_id": page["source_id"],
                    "filename": page["filename"],
                    "page": int(page["page"]),
                    "scope": (
                        "requirements"
                        if page["source_id"] in {doc["id"] for doc in shared_documents}
                        else "supplier"
                    ),
                    "text": page["text"],
                }
                for page in selected_pages
            ],
        }
        excerpts = retrieved_refs
    else:
        excerpts = {
            "all_documents": _all_pages(shared_documents, supplier_documents, max_pages=12)
        }
    prompt = (
        "Assess this TMT reinforcement-steel offer for an L&T procurement review. "
        "Decision support only; do not claim to apply undocumented L&T policy. The buyer RFP "
        "and explicitly uploaded requirements govern. Use only the provided evidence and this "
        "supplier's submission. Do not infer missing facts or charges. Missing is not proof of "
        "non-compliance. Return unknowns as null and statuses as COMPLIANT, PARTIALLY_COMPLIANT, "
        "NON_COMPLIANT, NOT_PROVIDED, NOT_VERIFIABLE, NOT_APPLICABLE, PENDING_REVIEW, or "
        "NOT_COMPARABLE. Assess only configured gates; gate decisions are preliminary for human "
        "review; do not invent gates. Do not calculate ranking or weighted totals. "
        "Return one JSON object only. Keep summary, reasons, and justifications to one short "
        "sentence (12 words maximum); risks and clarifications should be empty unless essential. "
        "Use at most one short evidence quote per assessed criterion and one quote per extracted "
        "section. Do not repeat document contents. CITATION FORMAT IS STRICT: in each "
        "criteria item, evidence is either a plain string quote or null; source_id is the exact "
        "source_id copied from an evidence_pages entry; page is that entry's integer page number. "
        "Never make evidence an array or object, and do not put a filename or source label in "
        "source_id. The quote must be a short exact contiguous excerpt from that cited page. "
        "For commercial_quote.comparability_status use exactly one of COMPARABLE, "
        "NOT_COMPARABLE, PENDING_REVIEW, or NOT_PROVIDED; do not use prose or a boolean. For "
        "technical_offer, quality_documents, commercial_quote, and delivery, cite only supplier "
        "pages. In RAG mode, criterion_sources maps each criterion to relevant evidence_pages "
        "by [source_id,page]; each page appears once. Use only those supplied pages.\n\n"
        f"SUPPLIER:{_compact_json(supplier)}\n"
        f"BUYER_REQUIREMENTS:{_compact_json(buyer_requirements or {})}\n"
        f"ACTIVE_CRITERIA:{_compact_json(active_criteria)}\n"
        f"CONFIGURED_GATES:{_compact_json(mandatory_gates)}\n"
        f"RETRIEVAL_MODE:{'RAG' if use_rag else 'ALL_BOUNDED_PAGES'}\n"
        f"EVIDENCE:{_compact_json(excerpts)}\n"
        "JSON_FIELDS: summary:string; gates:[{id,status:Pass|Fail|Pending,requirement,reason,"
        "evidence:string|null,source_id:string|null,page:integer|null}]; "
        "criteria:[{criterion_id,score:number,status,justification,evidence:string|null,"
        "source_id:string|null,page:integer|null,risks:string[],clarifications:string[]}]; "
        "technical_offer:{material_description,offered_grade,offered_specification,"
        "offered_quantity_mt:number|null,deviations:string[],compliance_status,evidence:"
        "[{source_id,page,quote}]}; quality_documents:[{document_type,provided:boolean|null,"
        "heat_or_batch_reference,source_id,page,evidence_quote}]; "
        "commercial_quote:{currency,basic_rate_per_mt,freight_per_mt,tax_amount,"
        "loading_unloading_amount,other_charges,total_quoted_amount,delivery_basis,"
        "quote_valid_until,payment_terms,exclusions:string[],comparability_status,"
        "evidence:[{source_id,page,quote}]}; "
        "delivery:{quantity_available_mt,first_delivery_date,completion_date,"
        "delivery_phases:string[],delivery_location,evidence:[{source_id,page,quote}]}."
    )
    permitted_pages: dict[str, set[int]] = {}
    if use_rag:
        page_sets = excerpts["evidence_pages"]
    else:
        page_sets = excerpts["all_documents"]
    for item in page_sets:
        permitted_pages.setdefault(item["source_id"], set()).add(int(item["page"]))
    return prompt, permitted_pages


def _retrieve_pages(query: str, documents: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    stop_words = {"this", "that", "with", "from", "your", "their", "shall", "should", "will", "must", "into", "and", "the"}
    terms = {
        term.casefold()
        for term in re.findall(r"[a-zA-Z0-9]{3,}", query)
        if term.casefold() not in stop_words
    }
    matches = []
    for document in documents:
        for page in document["pages"]:
            text = str(page["text"])
            words = set(term.casefold() for term in re.findall(r"[a-zA-Z0-9]{3,}", text))
            score = len(terms & words)
            if score:
                matches.append((score, document, page))
    matches.sort(key=lambda item: (-item[0], item[1]["filename"], item[2]["page"]))
    return [
        {
            "source_id": document["id"],
            "filename": document["filename"],
            "page": page["page"],
            "text": str(page["text"])[:4000],
        }
        for _, document, page in matches[:limit]
    ]


def _all_pages(
    shared_documents: list[dict[str, Any]],
    supplier_documents: list[dict[str, Any]],
    *,
    max_pages: int,
) -> list[dict[str, Any]]:
    excerpts = []
    for scope, collection in (("PROJECT REQUIREMENT", shared_documents), ("SUPPLIER SUBMISSION", supplier_documents)):
        for document in collection:
            for page in document["pages"]:
                if len(excerpts) >= max_pages:
                    return excerpts
                excerpts.append({
                    "source_id": document["id"],
                    "scope": scope,
                    "filename": document["filename"],
                    "page": page["page"],
                    "text": str(page["text"])[:6000],
                })
    return excerpts


def _validate_result(
    result: Any,
    criteria: list[dict[str, Any]],
    supplier_documents: list[dict[str, Any]],
    shared_documents: list[dict[str, Any]],
    permitted_pages: dict[str, set[int]],
) -> None:
    schema = {
        "type": "object",
        "required": ["summary", "gates", "criteria"],
        "properties": {
            "summary": {"type": "string"},
            "gates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "status", "requirement", "reason"],
                    "properties": {
                        "id": {"type": "string"},
                        "status": {"enum": ["Pass", "Fail", "Pending"]},
                        "requirement": {"type": "string"},
                        "reason": {"type": "string"},
                        "evidence": {"type": ["string", "null"]},
                        "source_id": {"type": ["string", "null"]},
                        "page": {"type": ["integer", "null"]},
                    },
                    "additionalProperties": True,
                },
            },
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["criterion_id", "score", "status", "justification", "evidence", "source_id", "page"],
                    "properties": {
                        "criterion_id": {"type": "string"},
                        "score": {"type": "number"},
                        "status": {
                            "enum": [
                                "Compliant",
                                "Partially compliant",
                                "Non-compliant",
                                "Not stated",
                                "Ambiguous",
                                "COMPLIANT",
                                "PARTIALLY_COMPLIANT",
                                "NON_COMPLIANT",
                                "NOT_PROVIDED",
                                "NOT_VERIFIABLE",
                                "NOT_APPLICABLE",
                                "PENDING_REVIEW",
                                "NOT_COMPARABLE",
                            ]
                        },
                        "justification": {"type": "string"},
                        "evidence": {"type": ["string", "null"]},
                        "source_id": {"type": ["string", "null"]},
                        "page": {"type": ["integer", "null"]},
                        "risks": {"type": "array", "items": {"type": "string"}},
                        "clarifications": {"type": "array", "items": {"type": "string"}},
                    },
                    "additionalProperties": True,
                },
            },
        },
        "additionalProperties": True,
    }
    errors = sorted(Draft202012Validator(schema).iter_errors(result), key=lambda error: list(error.path))
    if errors:
        raise EvaluationError(f"The selected LLM returned an invalid response: {errors[0].message}")
    expected_ids = {item["id"] for item in criteria if item.get("active", True)}
    actual_ids = [item["criterion_id"] for item in result["criteria"]]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise EvaluationError("The selected LLM did not return exactly one result per active criterion.")
    for item in result["criteria"]:
        maximum = next(c["max_score"] for c in criteria if c["id"] == item["criterion_id"])
        if not 0 <= float(item["score"]) <= float(maximum):
            raise EvaluationError(f"The selected LLM returned an out-of-range score for {item['criterion_id']}.")
    sources = {document["id"]: document for document in shared_documents + supplier_documents}
    for item in result["criteria"] + result["gates"]:
        source_id = item.get("source_id")
        if source_id is None and item.get("evidence"):
            raise EvaluationError("The selected LLM returned an evidence quote without a source document citation.")
        if source_id is None:
            continue
        if source_id not in sources:
            raise EvaluationError("The selected LLM cited a source that is not part of this evaluation.")
        if item.get("page") not in permitted_pages.get(source_id, set()):
            raise EvaluationError("The selected LLM cited a page that was not included in the evaluation context.")
        evidence = item.get("evidence")
        if evidence:
            page_text = next(
                str(page["text"])
                for page in sources[source_id]["pages"]
                if int(page["page"]) == int(item["page"])
            )
            if not _evidence_quote_matches(str(evidence), page_text):
                raise EvaluationError("The selected LLM evidence quote could not be verified in the cited source page.")

    supplier_source_ids = {document["id"] for document in supplier_documents}
    steel_citations = [
        *result.get("technical_offer", {}).get("evidence", []),
        *result.get("commercial_quote", {}).get("evidence", []),
        *result.get("delivery", {}).get("evidence", []),
    ]
    for quality_document in result.get("quality_documents", []):
        if quality_document.get("source_id") is not None:
            steel_citations.append({
                "source_id": quality_document.get("source_id"),
                "page": quality_document.get("page"),
                "quote": quality_document.get("evidence_quote"),
            })
    for citation in steel_citations:
        source_id = citation.get("source_id")
        page = citation.get("page")
        quote = citation.get("quote")
        if source_id not in supplier_source_ids:
            raise EvaluationError("A steel offer citation must refer to this supplier's own document.")
        if page not in permitted_pages.get(source_id, set()):
            raise EvaluationError("A steel offer citation refers to a page not included in this supplier's evaluation context.")
        if not isinstance(quote, str) or not quote.strip():
            raise EvaluationError("A steel offer citation must include an exact evidence quote.")
        source = sources[source_id]
        page_text = next(
            str(item["text"])
            for item in source["pages"]
            if int(item["page"]) == page
        )
        if not _evidence_quote_matches(quote, page_text):
            raise EvaluationError("A steel offer evidence quote could not be verified in the cited supplier page.")


def _evidence_quote_matches(evidence: str, page_text: str) -> bool:
    def normalize(text: str) -> str:
        value = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"(?<=\w)-\s*(?=\w)", "", value)
        return value

    def tokens(text: str) -> list[str]:
        return re.findall(r"[^\W_]+", normalize(text).casefold(), flags=re.UNICODE)

    evidence_tokens = tokens(evidence)
    page_tokens = tokens(page_text)
    if not evidence_tokens or len(evidence_tokens) > len(page_tokens):
        return False

    width = len(evidence_tokens)
    return any(
        page_tokens[index : index + width] == evidence_tokens
        for index in range(len(page_tokens) - width + 1)
    )
