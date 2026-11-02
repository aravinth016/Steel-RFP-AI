from __future__ import annotations

import json
import hashlib
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=False)

from src.database import (
    audit_events,
    criterion_evaluations,
    evaluation_warnings,
    get_app_setting,
    initialize,
    load_criteria,
    list_runs,
    load_rag_corpus,
    load_run,
    list_audit_chain,
    reserve_next_run_id,
    save_app_setting,
    save_criteria,
    save_rag_corpus,
    save_run,
    workflow_traces,
    verify_audit_chain,
)
from src.credentials import persist_api_key
from src.documents import extract_pdf
from src.qa_exports import build_qa_csv, build_qa_pdf, build_report_pdf
from src.evaluator import (
    EvaluationError,
    estimate_output_tokens,
    estimate_prompt_tokens,
)
from src.llm_providers import (
    PROVIDERS,
    call_llm_text,
    model_options,
    secret_name,
    test_connection,
)
from src.workflow import run_supplier_evaluation
from src.rag import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_CONFIG,
    build_qa_prompt,
    chunk_documents,
    embed_texts,
    retrieve_fused,
    scope_documents,
    validate_answer_citations,
)
from src.scoring import (
    STARTER_CRITERIA,
    apply_peer_metrics,
    comparable_price_score,
    rank_suppliers,
    steel_quotes_comparable,
    validate_criteria,
    weighted_score,
)
from src.scenarios import analyze_sensitivity, evaluate_scenario


st.set_page_config(
    page_title="Forge Steel RFP AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)
initialize()

def new_run_id() -> str:
    return reserve_next_run_id()


def new_document(uploaded: Any, scope: str) -> dict[str, Any]:
    extracted = extract_pdf(uploaded.name, uploaded.getvalue())
    return {
        "id": uuid.uuid4().hex[:12],
        "filename": extracted.filename,
        "sha256": extracted.sha256,
        "size_bytes": uploaded.size,
        "scope": scope,
        "pages": extracted.pages,
    }


def prepare_pending_evaluation(
    *,
    shared_files: Any,
    supplier_inputs: list[tuple[str, str, float, Any, str]],
    provider: str,
    model: str,
    criteria: list[dict[str, Any]],
    use_rag: bool,
    prices_comparable: bool,
    buyer_requirements: dict[str, Any],
) -> dict[str, Any]:
    if not shared_files:
        raise ValueError("Upload the buyer RFP or a shared requirement document before evaluating suppliers.")
    shared_docs = [new_document(item, "shared") for item in shared_files]
    suppliers: list[dict[str, Any]] = []
    for name, submitted, experience, files, notes in supplier_inputs:
        if not files:
            raise ValueError(f"{name}: upload at least one quotation or technical offer PDF.")
        docs = [new_document(item, "supplier") for item in files]
        suppliers.append({
            "supplier_id": uuid.uuid4().hex[:12],
            "name": name,
            "submission_date": submitted,
            "experience": experience,
            "price": None,
            "currency": None,
            "notes": notes,
            "documents": [{key: value for key, value in doc.items() if key != "pages"} for doc in docs],
            "_docs": docs,
        })
    run = {
        "run_id": new_run_id(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "RFP Proposal Evaluation",
        "status": "Evaluating",
        "is_demo": False,
        "provider": provider,
        "model": model,
        "rag_embedding_model": st.session_state.get(
            f"rag_embedding_model_{provider}", EMBEDDING_CONFIG[provider][1]
        ),
        "rag_embedding_provider": provider,
        "use_rag": use_rag,
        "prices_comparable": prices_comparable,
        "buyer_requirements": buyer_requirements,
        "ranking_policy": "Weighted PPI descending; earlier submission date; higher experience rating; supplier name ascending.",
        "criteria": criteria,
        "shared_documents": [
            {key: value for key, value in doc.items() if key != "pages"}
            for doc in shared_docs
        ],
        "suppliers": [],
        "expected_supplier_count": len(suppliers),
        "warnings": [],
        "workflow_traces": [],
        "audit_events": [],
    }
    estimates = [
        {
            "supplier_id": supplier["supplier_id"],
            "supplier": supplier["name"],
            "input_tokens": estimate_prompt_tokens(
                supplier={key: value for key, value in supplier.items() if key != "documents" and key != "_docs"},
                criteria=criteria,
                shared_documents=shared_docs,
                supplier_documents=supplier["_docs"],
                use_rag=use_rag,
                buyer_requirements=buyer_requirements,
            ),
            "output_tokens": estimate_output_tokens(criteria),
        }
        for supplier in suppliers
    ]
    run["token_estimates"] = estimates
    return {
        "run": run,
        "suppliers": suppliers,
        "shared_docs": shared_docs,
        "token_estimates": estimates,
    }


def create_demo_run(criteria: list[dict[str, Any]]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    run_id = new_run_id()
    score_sets = [
        ("Apex Steels", [9, 8, 7, 8, 9], 4.5, "2026-10-04", 33040000, "INR"),
        ("Bright Metals", [8, 4, 8, 3, 2], 2.0, "2026-10-06", 28575000, "INR"),
        ("Nexa Steelworks", [9, 10, 9, 8, 8], 4.0, "2026-10-02", 30562000, "INR"),
        ("Orbit Alloys", [5, 8, 6, 7, 6], 5.0, "2026-10-08", 0, "INR"),
    ]
    demo_suppliers = []
    for index, (name, scores, experience, submitted, price, currency) in enumerate(score_sets):
        supplier_id = f"DEMO-{index + 1}"
        doc_id = f"SYNTH-{index + 1}"
        statements = [
            f"SYNTHETIC DEMO DATA — {name} offers TMT reinforcement steel for the illustrative RFP.",
            f"Offered grade and quantity are stated in the {name} synthetic quotation; verify cited documents.",
            f"SYNTHETIC DEMO DATA — {name} quotation price: {currency} {price:,}; fictional only.",
        ]
        results = []
        active_criteria = [c for c in criteria if c.get("active", True)]
        for result_index, criterion in enumerate(active_criteria):
            score = min(scores[result_index % len(scores)], criterion["max_score"])
            page_number = 1 + (len(results) % 2)
            results.append(
                {
                    "criterion_id": criterion["id"],
                    "criterion_name": criterion["name"],
                    "ai_score": score,
                    "ai_status": "COMPLIANT" if score >= 7 else "PARTIALLY_COMPLIANT",
                    "justification": (
                        f"Synthetic example assessment: score {score}/{criterion['max_score']} "
                        f"under the displayed starter rubric. This is not a real supplier assessment."
                    ),
                    "evidence": statements[(len(results) + index) % len(statements)],
                    "source_id": doc_id,
                    "source_file": f"synthetic_{name.lower().replace(' ', '_')}.pdf",
                    "page": page_number,
                    "risks": ["Synthetic example only; validate against actual tender requirements."]
                    if score < 3 else [],
                    "clarifications": ["Confirm this item against actual project documents."]
                    if score < 3 else [],
                    "final_score": score,
                }
            )
        final = weighted_score(criteria, {result["criterion_id"]: result["ai_score"] for result in results})
        demo_suppliers.append(
            {
                "supplier_id": supplier_id,
                "name": name,
                "submission_date": submitted,
                "experience": experience,
                "price": price,
                "currency": currency,
                "notes": "Synthetic fictional steel supplier. Not a real quotation.",
                "technical_offer": {
                    "material_description": "TMT reinforcement bars",
                    "offered_grade": "Fe 500" if name == "Orbit Alloys" else "Fe 500D",
                    "offered_specification": "Synthetic project RFP specification",
                    "offered_quantity_mt": 500,
                    "deviations": ["Grade offered is Fe 500, not Fe 500D"] if name == "Orbit Alloys" else [],
                    "compliance_status": "NON_COMPLIANT" if name == "Orbit Alloys" else "COMPLIANT",
                    "evidence": [],
                },
                "quality_documents": (
                    [] if name == "Bright Metals" else [{
                        "document_type": "Mill Test Certificate",
                        "provided": True,
                        "heat_or_batch_reference": f"SYN-HEAT-{index + 100}",
                        "source_id": doc_id,
                        "page": 2,
                        "evidence_quote": "SYNTHETIC DEMO DATA - sample MTC reference only.",
                        "verification_status": "NOT_INDEPENDENTLY_VERIFIED",
                    }]
                ),
                "commercial_quote": {
                    "currency": currency,
                    "basic_rate_per_mt": [55000, 48000, 51000, 50500][index],
                    "freight_per_mt": [1000, None, 800, 2000][index],
                    "tax_amount": [5040000, None, 4662000, None][index],
                    "loading_unloading_amount": None,
                    "other_charges": None,
                    "total_quoted_amount": price or None,
                    "delivery_basis": ["Delivered to project site", "Ex-works", "Delivered to project site", "Ex-works"][index],
                    "quote_valid_until": None,
                    "payment_terms": "As stated in synthetic quotation",
                    "exclusions": ["Taxes"] if index in (1, 3) else [],
                    "comparability_status": "PENDING_REVIEW" if index in (1, 3) else "COMPARABLE",
                    "evidence": [],
                },
                "delivery": {
                    "quantity_available_mt": [500, 500, 500, 500][index],
                    "first_delivery_date": None,
                    "completion_date": ["2026-11-25", "2026-12-20", "2026-11-15", "2026-11-28"][index],
                    "delivery_phases": [],
                    "delivery_location": "Project site, Pune, Maharashtra",
                    "evidence": [],
                },
                "documents": [
                    {"id": doc_id, "filename": f"synthetic_{name.lower().replace(' ', '_')}.pdf", "sha256": "synthetic", "size_bytes": 0, "scope": "supplier"}
                ],
                "criteria_results": results,
                "summary": (
                    f"Synthetic TMT steel quotation scenario for {name}. Compare grade, quality evidence, "
                    "commercial terms, quantity, delivery, risks, and clarifications. Not a real quotation."
                ),
                "ai_score": final,
                "absolute_score": final,
                "final_score": final,
                "ppi": None,
            }
        )
    run = {
        "run_id": run_id,
        "created_at": now,
        "mode": "RFP Proposal Evaluation",
        "status": "Demo",
        "is_demo": True,
        "provider": "Deterministic synthetic demo evaluator",
        "model": None,
        "use_rag": False,
        "ranking_policy": "Weighted PPI descending; earlier submission date; higher experience rating; supplier name ascending.",
        "criteria": criteria,
        "buyer_requirements": {
            "material": "TMT reinforcement steel",
            "required_quantity_mt": 500,
            "required_grade": "Fe 500D",
            "delivery_location": "Project site, Pune, Maharashtra",
            "required_delivery_date": "2026-11-30",
            "currency": "INR",
        },
        "shared_documents": [
            {"id": "SYNTH-REQ", "filename": "synthetic_steel_rfp.pdf", "sha256": "synthetic", "size_bytes": 0, "scope": "shared"}
        ],
        "suppliers": demo_suppliers,
        "warnings": ["SYNTHETIC DEMO DATA — all suppliers, evidence, scores, dates, and prices are fictional."],
    }
    update_rankings(run)
    save_run(run)
    return run


def update_rankings(run: dict[str, Any]) -> None:
    criteria = run["criteria"]
    supplier_rows = []
    for supplier in run["suppliers"]:
        supplier["ai_score"] = weighted_score(
            criteria,
            {item["criterion_id"]: item.get("ai_score") for item in supplier.get("criteria_results", [])},
        )
        supplier["final_score"] = supplier["ai_score"]
        supplier["absolute_score"] = supplier["ai_score"]
        supplier_rows.append(supplier)
    apply_peer_metrics(criteria, supplier_rows)
    run["suppliers"] = rank_suppliers(supplier_rows)


def apply_commercial_price_scores(run: dict[str, Any], all_suppliers_evaluated: bool) -> None:
    commercial = next(
        (
            item for item in run["criteria"]
            if item.get("active", True)
            and (
                item["id"] == "commercial"
                or any(word in item["name"].casefold() for word in ("commercial",                 "Quoted total", "cost"))
            )
        ),
        None,
    )
    if commercial is None:
        return
    suppliers = run["suppliers"]
    quotes = [supplier.get("commercial_quote", {}) for supplier in suppliers]
    prices = [quote.get("total_quoted_amount") for quote in quotes]
    buyer_requirements = run.get("buyer_requirements", {})
    valid, comparison_reasons = steel_quotes_comparable(
        suppliers,
        required_quantity=buyer_requirements.get("required_quantity_mt"),
        buyer_currency=buyer_requirements.get("currency"),
        reviewer_confirmed=run.get("prices_comparable", False),
        all_suppliers_evaluated=all_suppliers_evaluated,
    )
    lowest = min(prices) if valid else None
    for supplier in suppliers:
        result = next(
            (item for item in supplier.get("criteria_results", []) if item["criterion_id"] == commercial["id"]),
            None,
        )
        if result is None:
            continue
        quote = supplier.get("commercial_quote", {})
        result["llm_proposed_score"] = result.get("ai_score")
        result["llm_proposed_justification"] = result.get("justification")
        result["score_source"] = "system"
        result["source_id"] = None
        result["source_file"] = None
        result["page"] = None
        result["evidence"] = None
        if valid and lowest is not None:
            score = comparable_price_score(
                float(lowest), float(quote["total_quoted_amount"]), float(commercial["max_score"])
            )
            result.update({
                "ai_score": score,
                "ai_status": "COMPLIANT",
                "justification": (
                    f"System-calculated using the RFP-authorized comparable-total-price method: "
                    f"lowest verified comparable total divided by this supplier's total, multiplied "
                    f"by the {commercial['max_score']}-point maximum. The reviewer confirmed common "
                    f"currency, offered quantity, delivery basis, and included cost scope."
                ),
                "price_score_formula": "lowest comparable total quoted amount / supplier total quoted amount × criterion maximum",
            })
        else:
            reason = "Commercial comparison is not scored: " + "; ".join(comparison_reasons) + "."
            result.update({
                "ai_score": None,
                "ai_status": "NOT_COMPARABLE",
                "justification": reason,
                "price_score_formula": None,
            })
        result["final_score"] = result.get("ai_score")


def render_home() -> None:
    st.markdown(
        """
        <style>
        .home-hero {
            padding: 2.4rem 2.5rem;
            border-radius: 24px;
            color: #18324a;
            background: linear-gradient(135deg, #f8fbff 0%, #eef7f5 100%);
            border: 1px solid #d9e8e7;
            border-left: 6px solid #168b79;
            box-shadow: 0 12px 32px rgba(16, 36, 58, .08);
            margin: .25rem 0 1.5rem;
        }
        .home-eyebrow {
            color: #168b79;
            font-size: .78rem;
            font-weight: 750;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: .7rem;
        }
        .home-hero h1 {
            color: #18324a;
            font-size: clamp(2rem, 5vw, 3.25rem);
            line-height: 1.08;
            letter-spacing: -.04em;
            margin: 0 0 .85rem;
        }
        .home-hero p {
            color: #526879;
            font-size: 1.03rem;
            line-height: 1.65;
            max-width: 760px;
            margin: 0;
        }
        .home-section-title {
            font-size: 1.45rem;
            font-weight: 750;
            letter-spacing: -.025em;
            margin: 1.35rem 0 .2rem;
            color: #142a40;
        }
        .home-section-subtitle {
            color: #64748b;
            margin: 0 0 1rem;
        }
        .home-card {
            height: 100%;
            min-height: 142px;
            padding: 1.15rem 1.2rem;
            border: 1px solid #e3eaf0;
            border-radius: 16px;
            background: linear-gradient(180deg, #fff 0%, #f8fbfd 100%);
            box-shadow: 0 5px 18px rgba(24, 50, 73, .055);
        }
        .home-card-icon {
            font-size: 1.25rem;
            margin-bottom: .55rem;
        }
        .home-card h3 {
            color: #18324a;
            font-size: 1.02rem;
            margin: 0 0 .38rem;
        }
        .home-card p {
            color: #5b6b7b;
            line-height: 1.55;
            font-size: .91rem;
            margin: 0;
        }
        .home-demo {
            padding: 1.4rem 1.5rem;
            border: 1px solid #cfe7e4;
            border-radius: 18px;
            background: linear-gradient(120deg, #f0fbf8 0%, #f5f9ff 100%);
            margin: .5rem 0 1rem;
        }
        .home-demo h3 {
            color: #18324a;
            font-size: 1.25rem;
            margin: 0 0 .45rem;
        }
        .home-demo p {
            color: #506577;
            line-height: 1.6;
            margin: 0;
        }
        .home-pill {
            display: inline-block;
            margin: .25rem .35rem .1rem 0;
            padding: .3rem .65rem;
            border-radius: 999px;
            color: #145c5b;
            background: #dff5ef;
            border: 1px solid #c3e9df;
            font-size: .78rem;
            font-weight: 650;
        }
        @media (max-width: 760px) {
            .home-hero { padding: 1.6rem 1.35rem; border-radius: 18px; }
            .home-card { min-height: 0; }
        }
        </style>
        <section class="home-hero">
            <div class="home-eyebrow">Steel procurement · decision support</div>
            <h1>Forge Steel RFP AI</h1>
            <p>
                Bring supplier quotations and your RFP into one evidence-led review.
                Compare TMT reinforcement-steel offers with clear criteria, transparent
                scoring, and source citations—while keeping the procurement decision
                with your team.
            </p>
        </section>
        <div class="home-section-title">About the application</div>
        <p class="home-section-subtitle">
            A practical workspace for organizing steel bids and reviewing what each offer actually supports.
        </p>
        """,
        unsafe_allow_html=True,
    )
    overview_cards = [
        (
            "01",
            "RFP-grounded review",
            "Use buyer-provided requirements and steel documents to assess grade, quantity, delivery, quality records, and commercial terms.",
        ),
        (
            "02",
            "Evidence you can inspect",
            "See extracted findings alongside document names, page references, risks, and clarification points.",
        ),
        (
            "03",
            "Transparent comparison",
            "AI proposes criterion scores; Python applies configured weights, peer benchmarks, and deterministic ranking.",
        ),
    ]
    for row_start in range(0, len(overview_cards), 2):
        overview_columns = st.columns(2, gap="medium")
        for column, (icon, title, description) in zip(
            overview_columns, overview_cards[row_start : row_start + 2]
        ):
            with column:
                st.markdown(
                    f"""
                    <div class="home-card">
                        <div class="home-card-icon">{icon}</div>
                        <h3>{title}</h3>
                        <p>{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    st.markdown(
        """
        <div class="home-section-title">From uploaded files to the final score</div>
        <p class="home-section-subtitle">
            The app prepares each supplier's evidence, asks before sending it to the selected
            provider, then calculates and ranks results locally.
        </p>
        """,
        unsafe_allow_html=True,
    )
    evaluation_steps = [
        (
            "01",
            "Upload & extract",
            "Add the buyer RFP and each supplier's PDFs. Text and page references are extracted and checked locally; unreadable or unsupported files stop with an error.",
        ),
        (
            "02",
            "Prepare & confirm",
            "The app prepares a separate supplier prompt, shows approximate input/output token estimates, and waits for your confirmation before any evaluation call.",
        ),
        (
            "03",
            "Assess & verify",
            "The selected LLM proposes structured findings and citations. The app validates the response and checks citations against the uploaded documents.",
        ),
        (
            "04",
            "Score & compare",
            "Python calculates weighted scores, applicable peer metrics, and deterministic ranks. You review the evidence and decide what to do next.",
        ),
    ]
    for row_start in range(0, len(evaluation_steps), 2):
        evaluation_columns = st.columns(2, gap="medium")
        for column, (number, title, description) in zip(
            evaluation_columns, evaluation_steps[row_start : row_start + 2]
        ):
            with column:
                st.markdown(
                    f"""
                    <div class="home-card">
                        <div class="home-card-icon">{number}</div>
                        <h3>{title}</h3>
                        <p>{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    st.markdown(
        """
        <div class="home-section-title">The StateGraph behind the process</div>
        <p class="home-section-subtitle">
            Each supplier follows its own path through these nodes. The graph records progress
            and handles cached results and errors without mixing supplier evidence.
        </p>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        graph_nodes = [
            (
                "Prepare",
                "Collect provider/model, criterion, requirements, and document counts. A validated cached result can skip the provider call.",
            ),
            (
                "Evaluate",
                "Send the prepared evidence for this supplier to the selected LLM and receive its structured assessment.",
            ),
            (
                "Validate",
                "Check the response against the strict evaluation schema and active criteria.",
            ),
            (
                "Verify evidence",
                "Check evidence citations against the uploaded pages and record verification warnings.",
            ),
            (
                "Score",
                "Calculate the weighted score deterministically in Python from the validated criterion results.",
            ),
            (
                "Persist",
                "Assemble the supplier result, warnings, stage trace, and audit events for run history.",
            ),
        ]
        for row_start in range(0, len(graph_nodes), 2):
            node_columns = st.columns(2, gap="medium")
            for index, (node, function) in enumerate(
                graph_nodes[row_start : row_start + 2], start=row_start
            ):
                with node_columns[index - row_start]:
                    st.markdown(
                        f"""
                        <div class="home-card">
                            <div class="home-card-icon">{index + 1:02d}</div>
                            <h3>{node}</h3>
                            <p>{function}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.info(
            "**Normal path:** Prepare → Evaluate → Validate → Verify evidence → Score → Persist. "
            "A validated cached result skips Evaluate. Evaluation or validation errors route "
            "to Persist. An unverified citation is recorded as a warning and scoring continues; "
            "a scoring error routes to Persist. A failed "
            "supplier does not stop the remaining suppliers."
        )
    st.warning(
        "Decision-support only: the app does not award contracts or make legal decisions. "
        "AI-proposed scores are not an approval, and starter criteria/weights are examples—not procurement policy."
    )
    st.divider()
    st.markdown(
        """
        <div class="home-section-title">Explore the interactive demo</div>
        <p class="home-section-subtitle">
            See the full review experience using a ready-made, fictional steel procurement run.
        </p>
        <div class="home-demo">
            <h3>A complete sample comparison, ready to explore</h3>
            <p>
                Review four fictional suppliers—Apex Steels, Bright Metals, Nexa Steelworks,
                and Orbit Alloys—with sample criteria scores, source-style evidence, steel
                offer details, commercial comparisons, findings, and ranking.
                The demo is generated locally and does not call an LLM.
            </p>
        </div>
        <span class="home-pill">4 fictional suppliers</span>
        <span class="home-pill">No API key needed</span>
        <span class="home-pill">Interactive leaderboard</span>
        <span class="home-pill">Evidence and findings</span>
        <span class="home-pill">What-if analysis</span>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open the completed demo", type="primary", width="stretch"):
        criteria = st.session_state.get("criteria", STARTER_CRITERIA)
        demo = create_demo_run(criteria)
        st.session_state.active_run_id = demo["run_id"]
        st.session_state.page = "Leaderboard"
        st.session_state.pending_workspace_page = "Leaderboard"
        st.rerun()
    st.caption(
        "All demo suppliers, evidence, scores, and prices are synthetic; they are not real quotations or procurement advice."
    )


def render_new_evaluation() -> None:
    st.title("New steel RFP evaluation")
    st.caption("Create one run per steel procurement RFP. The buyer's uploaded documents are authoritative; do not assume unstated grade, standards, quantity, dates, or terms.")
    criteria = st.session_state.get("criteria", STARTER_CRITERIA)
    errors = validate_criteria(criteria)
    if errors:
        st.error("Criteria must be valid before a run can start: " + " ".join(errors))
        return
    provider = selected_llm_provider()
    api_key = st.session_state.get(f"api_key_session_{provider}") or configured_api_key(provider)
    available_models = model_options(provider)
    model = selected_llm_model(provider, available_models)
    if not api_key:
        st.error(f"{provider} API key is required to run a live evaluation. Configure it under LLM Settings. The demo remains available without a key.")
        return
    st.info(f"This run will use **{provider}** · `{model}`. No automatic provider fallback is used.")
    mode = "RFP Proposal Evaluation"
    run_criteria = [dict(item) for item in criteria]
    run_criteria_errors = validate_criteria(run_criteria)
    if run_criteria_errors:
        st.error("This run's criteria are invalid: " + " ".join(run_criteria_errors))
        return
    use_rag = st.toggle("Use local evidence retrieval", value=True, help="Local keyword retrieval scopes excerpts to the RFP and current supplier. No public internet is searched.")
    st.caption(f"Before continuing, you will be asked to confirm that steel quotation and RFP excerpts are sent to {provider} for evaluation.")
    st.caption("Extracted PDF text is stored in this local run database so the Q&A Bot can answer questions about saved runs. Original PDFs and API keys are not stored.")
    supplier_count = st.number_input("Number of steel suppliers", min_value=1, max_value=8, value=4, step=1)
    with st.form("new_evaluation_form"):
        st.subheader("Buyer RFP and supporting documents")
        shared_files = st.file_uploader(
            "Upload the procurement RFP/enquiry, steel specification, BOQ, delivery schedule, and commercial terms (PDF)",
            type=["pdf"], accept_multiple_files=True, key="shared_pdf",
        )
        req_col1, req_col2, req_col3 = st.columns(3)
        with req_col1:
            required_grade = st.text_input("RFP required steel grade (optional)")
            required_quantity_mt = st.number_input(
                "RFP required quantity (MT; 0 if not specified)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )
        with req_col2:
            delivery_location = st.text_input("RFP delivery location (optional)")
            required_delivery_date = st.text_input(
                "RFP required delivery date/period (optional)",
                placeholder="Use the date or period stated in the RFP",
            )
        with req_col3:
            buyer_currency = st.text_input("RFP currency (optional)", placeholder="e.g. INR")
            material_description = st.text_input(
                "Material / product (optional)",
                value="TMT reinforcement steel",
            )
        supplier_inputs = []
        for index in range(int(supplier_count)):
            st.markdown(f"#### Steel supplier {index + 1}")
            cols = st.columns([2, 1, 1])
            with cols[0]:
                name = st.text_input("Supplier name", key=f"supplier_name_{index}")
            with cols[1]:
                submitted = st.date_input("Submission date", value=date.today(), key=f"submission_date_{index}")
            with cols[2]:
                experience = st.number_input(
                    "Experience rating (0-5)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.5,
                    key=f"experience_{index}",
                )
            files = st.file_uploader(
                f"{name or f'Supplier {index + 1}'} technical offer, quotation, MTC, certificates, and supporting PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"supplier_files_{index}",
                help="Include the quotation breakdown, offered grade/quantity, quality documents, availability, delivery, experience, warranty, and deviations where provided.",
            )
            notes = st.text_input("Notes (optional)", key=f"supplier_notes_{index}")
            supplier_inputs.append((name.strip(), submitted.isoformat(), experience, files, notes))
        confirmed = st.checkbox(
            f"I authorize sending the uploaded document excerpts and supplier metadata to {provider} for this evaluation. "
            "The API key is not saved in the run or SQLite.",
            key=f"llm_send_consent_{provider}",
        )
        prices_comparable = st.checkbox(
            "The RFP permits lowest-total-price scoring, and I verified that quotes use the same currency, quantity, delivery basis, freight/tax treatment, and scope.",
            value=False,
            help="This attestation alone is not enough: extracted quote totals, currencies, quantities, delivery bases, and supplier comparability statuses must also agree.",
        )
        submitted_form = st.form_submit_button(f"Evaluate with {provider}", type="primary", width="stretch")
    pending = st.session_state.get("pending_evaluation")
    if pending is None:
        if not submitted_form:
            return
        if not confirmed:
            st.error(f"Confirm the {provider} data-sharing notice before starting.")
            return
        names = [entry[0].casefold() for entry in supplier_inputs]
        if any(not name for name in names) or len(names) != len(set(names)):
            st.error("Enter a unique, non-empty name for every supplier.")
            return
        buyer_requirements = {
            "material": material_description.strip() or None,
            "required_grade": required_grade.strip() or None,
            "required_quantity_mt": required_quantity_mt or None,
            "quantity_unit": "MT" if required_quantity_mt else None,
            "delivery_location": delivery_location.strip() or None,
            "required_delivery_date": required_delivery_date.strip() or None,
            "currency": buyer_currency.strip().upper() or None,
        }
        try:
            pending = prepare_pending_evaluation(
                shared_files=shared_files,
                supplier_inputs=supplier_inputs,
                provider=provider,
                model=model,
                criteria=run_criteria,
                use_rag=use_rag,
                prices_comparable=prices_comparable,
                buyer_requirements=buyer_requirements,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state["pending_evaluation"] = pending
        st.rerun()

    pending_run = pending["run"]
    st.subheader("Pre-evaluation estimate")
    st.caption(
        f"Approximate token estimates for {pending_run['provider']} · "
        f"`{pending_run['model']}`. This is a local estimate only; no provider call "
        "has been made."
    )
    st.dataframe(
        [
            {
                "Supplier": item["supplier"],
                "Estimated input tokens": item["input_tokens"],
                "Estimated output tokens": item["output_tokens"],
                "Estimated total tokens": item["input_tokens"] + item["output_tokens"],
            }
            for item in pending["token_estimates"]
        ],
        hide_index=True,
        width="stretch",
    )
    total_estimated_tokens = sum(
        item["input_tokens"] for item in pending["token_estimates"]
    )
    total_estimated_output_tokens = sum(
        item["output_tokens"] for item in pending["token_estimates"]
    )
    estimate_columns = st.columns(3)
    estimate_columns[0].metric("Estimated input tokens", f"~{total_estimated_tokens:,}")
    estimate_columns[1].metric(
        "Estimated output tokens", f"~{total_estimated_output_tokens:,}"
    )
    estimate_columns[2].metric(
        "Estimated total tokens",
        f"~{total_estimated_tokens + total_estimated_output_tokens:,}",
    )
    st.caption(
        "Input estimate uses a rough four-characters-per-token prompt heuristic; output "
        "estimate uses an approximate per-criterion JSON allowance. Actual tokenization "
        "and output length vary. Estimated cost is unavailable because a verified current "
        "rate for this exact provider/model is not configured. These estimates do not "
        "limit document text or model output."
    )
    proceed_col, cancel_col = st.columns(2)
    with proceed_col:
        proceed = st.button(
            f"Proceed with {pending_run['provider']} evaluation",
            type="primary",
            width="stretch",
        )
    with cancel_col:
        cancel = st.button("Discard pending evaluation", width="stretch")
    if cancel:
        st.session_state.pop("pending_evaluation", None)
        st.rerun()
    if not proceed:
        return

    run = pending_run
    suppliers = pending["suppliers"]
    shared_docs = pending["shared_docs"]
    provider = run["provider"]
    model = run["model"]
    use_rag = run["use_rag"]
    run_criteria = run["criteria"]
    api_key = (
        st.session_state.get(f"api_key_session_{provider}")
        or configured_api_key(provider)
    )
    if not api_key:
        st.error(f"{provider} API key is no longer available. Configure it under LLM Settings.")
        return
    st.session_state.pop("pending_evaluation", None)
    progress = st.progress(0, text="Validating files and extracting PDF text…")
    errors_by_supplier = []
    rag_documents = [dict(document) for document in shared_docs]
    for index, supplier in enumerate(suppliers):
        progress.progress(
            0.15 + 0.75 * index / max(len(suppliers), 1),
            text=f"Preparing {supplier['name']} for {provider} ({model})…",
        )
        try:
            supplier_docs = supplier.pop("_docs")
            progress.progress(
                0.15 + 0.75 * index / max(len(suppliers), 1),
                text=f"Requesting {provider} model {model} for {supplier['name']}…",
            )
            workflow_record = run_supplier_evaluation(
                run_id=run["run_id"],
                api_key=api_key,
                model=model,
                provider=provider,
                supplier={key: value for key, value in supplier.items() if key != "documents"},
                criteria=run_criteria,
                shared_documents=shared_docs,
                supplier_documents=supplier_docs,
                use_rag=use_rag,
                buyer_requirements=run["buyer_requirements"],
            )
            run["workflow_traces"].extend(workflow_record["traces"])
            run["audit_events"].extend(workflow_record["audit_events"])
            if workflow_record["status"] != "complete":
                raise EvaluationError(workflow_record["error"] or "Supplier evaluation failed.")
            result = workflow_record["evaluation"]
            progress.progress(
                0.15 + 0.75 * index / max(len(suppliers), 1),
                text=f"Response received from {provider} for {supplier['name']}; validating evidence…",
            )
            results = []
            for item in result["criteria"]:
                criterion = next(c for c in run_criteria if c["id"] == item["criterion_id"])
                source_doc = next((d for d in shared_docs + supplier_docs if d["id"] == item.get("source_id")), None)
                results.append({
                    **item,
                    "criterion_name": criterion["name"],
                    "ai_score": float(item["score"]),
                    "final_score": float(item["score"]),
                    "source_file": source_doc["filename"] if source_doc else None,
                })
            source_names = {
                document["id"]: document["filename"]
                for document in shared_docs + supplier_docs
            }
            technical_offer = result["technical_offer"]
            commercial_quote = result["commercial_quote"]
            delivery = result["delivery"]
            for section in (technical_offer, commercial_quote, delivery):
                for citation in section.get("evidence", []):
                    citation["source_file"] = source_names.get(citation["source_id"])
            quality_documents = result["quality_documents"]
            for document in quality_documents:
                document["source_file"] = source_names.get(document.get("source_id"))
            saved_supplier = {key: value for key, value in supplier.items() if key != "_docs"}
            saved_supplier.update({
                "criteria_results": results,
                "summary": result["summary"],
                "technical_offer": technical_offer,
                "quality_documents": quality_documents,
                "commercial_quote": commercial_quote,
                "delivery": delivery,
                "price": commercial_quote.get("total_quoted_amount"),
                "currency": commercial_quote.get("currency"),
                "provider": provider,
                "workflow_weighted_score": workflow_record["weighted_score"],
                "evidence_checks": workflow_record["evidence_checks"],
                "token_usage": workflow_record.get("token_usage", {}),
                "estimated_input_tokens": next(
                    item["input_tokens"]
                    for item in run["token_estimates"]
                    if item["supplier_id"] == supplier["supplier_id"]
                ),
                "estimated_output_tokens": next(
                    item["output_tokens"]
                    for item in run["token_estimates"]
                    if item["supplier_id"] == supplier["supplier_id"]
                ),
            })
            run["suppliers"].append(saved_supplier)
            run["warnings"].extend(
                f"{supplier['name']}: {warning}"
                for warning in (
                    result.get("validation_warnings", [])
                    + workflow_record["warnings"]
                )
            )
            rag_documents.extend(
                {**document, "supplier_id": supplier["supplier_id"]}
                for document in supplier_docs
            )
        except EvaluationError as exc:
            error_message = str(exc)
            errors_by_supplier.append(f"{supplier['name']}: {error_message}")
            failed_supplier = {key: value for key, value in supplier.items() if key != "_docs"}
            failed_supplier.update({
                "criteria_results": [],
                "summary": "Evaluation failed; no model assessment was saved.",
                "evaluation_error": error_message,
            })
            run["suppliers"].append(failed_supplier)
        except Exception as exc:
            error_message = f"Evaluation stopped ({type(exc).__name__})."
            errors_by_supplier.append(f"{supplier['name']}: {error_message}")
            failed_supplier = {key: value for key, value in supplier.items() if key != "_docs"}
            failed_supplier.update({
                "criteria_results": [],
                "summary": error_message,
                "evaluation_error": error_message,
            })
            run["suppliers"].append(failed_supplier)
    progress.progress(1.0, text="Evaluation and schema checks complete.")
    if errors_by_supplier:
        run["warnings"].extend(errors_by_supplier)
        run["status"] = (
            "Partially evaluated"
            if any(not supplier.get("evaluation_error") for supplier in run["suppliers"])
            else "Evaluation failed"
        )
    else:
        run["status"] = "Evaluation complete"
    if run["suppliers"]:
        apply_commercial_price_scores(run, len(run["suppliers"]) == len(suppliers))
        update_rankings(run)
        save_run(run)
        save_rag_corpus(run["run_id"], rag_documents)
        st.session_state.active_run_id = run["run_id"]
        st.session_state.page = "Leaderboard"
        st.session_state.pending_workspace_page = "Leaderboard"
        completed_count = sum(not supplier.get("evaluation_error") for supplier in run["suppliers"])
        st.success(f"Saved {completed_count} completed supplier evaluation(s) in {run['run_id']}.")
        if errors_by_supplier:
            st.warning("Some supplier evaluations failed; completed results were preserved. " + " ".join(errors_by_supplier))
        st.rerun()
    else:
        st.error("No supplier evaluation completed. " + " ".join(errors_by_supplier))


def render_leaderboard(run: dict[str, Any]) -> None:
    st.title("Leaderboard")
    render_run_badges(run)
    failed_suppliers = [supplier for supplier in run["suppliers"] if supplier.get("evaluation_error")]
    incomplete = (
        len(run["suppliers"]) < run.get("expected_supplier_count", len(run["suppliers"]))
        or bool(failed_suppliers)
    )
    if incomplete:
        st.error("This run is incomplete because one or more supplier evaluations failed. Do not select a recommendation from this comparison. See the failed supplier details and start a new run after correcting the issue.")
    winner = None if incomplete else next(iter(run["suppliers"]), None)
    if winner:
        st.success(
            f"Highest-ranked supplier: **{winner['name']}** — "
            f"{winner['ai_score']:.2f}% AI weighted score. This is decision support, not an award decision."
        )
    else:
        st.info("No completed supplier evaluation is available to rank.")
    rows = []
    for supplier in run["suppliers"]:
        rows.append({
            "Rank": supplier.get("rank"),
            "Supplier": supplier["name"],
            "Evaluation": "Failed" if supplier.get("evaluation_error") else "Completed",
            "Estimated input tokens": next(
                (
                    item["input_tokens"]
                    for item in run.get("token_estimates", [])
                    if item.get("supplier_id") == supplier.get("supplier_id")
                ),
                None,
            ),
            "Estimated output tokens": next(
                (
                    item["output_tokens"]
                    for item in run.get("token_estimates", [])
                    if item.get("supplier_id") == supplier.get("supplier_id")
                ),
                None,
            ),
            "Actual input tokens": (supplier.get("token_usage") or {}).get("input_tokens"),
            "Actual output tokens": (supplier.get("token_usage") or {}).get("output_tokens"),
            "AI weighted score": supplier.get("ai_score"),
            "PPI (peer benchmark)": supplier.get("ppi"),
            "Submitted": supplier.get("submission_date"),
            "Experience": supplier.get("experience"),
            "Quoted total": (
                f"{supplier.get('currency', '')} {supplier['price']:,.2f}"
                if isinstance(supplier.get("price"), (int, float))
                else "—"
            ),
            "Grade": supplier.get("technical_offer", {}).get("offered_grade") or "Not provided",
            "Quantity (MT)": supplier.get("technical_offer", {}).get("offered_quantity_mt"),
            "MTCs provided": sum(
                item.get("provided") is True
                for item in supplier.get("quality_documents", [])
                if "mill test" in item.get("document_type", "").casefold()
            ),
            "Evaluation error": supplier.get("evaluation_error"),
            "Missing info": any(
                result.get("ai_status") in ("NOT_PROVIDED", "NOT_VERIFIABLE", "PENDING_REVIEW")
                for result in supplier.get("criteria_results", [])
            ),
        })
    frame = pd.DataFrame(rows)
    sort_by = st.selectbox("Sort by", ["Rank", "Supplier", "AI weighted score", "PPI (peer benchmark)"])
    filtered = frame.copy()
    if sort_by != "Rank":
        filtered = filtered.sort_values(sort_by, ascending=sort_by == "Supplier", na_position="last")
    st.dataframe(filtered, width="stretch", hide_index=True)
    st.caption(f"Ranking policy: {run['ranking_policy']}")
    csv_col, pdf_col, json_col = st.columns(3)
    with csv_col:
        st.download_button(
            "Download leaderboard CSV",
            data=filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{run['run_id']}_leaderboard.csv",
            mime="text/csv",
            width="stretch",
        )
    with pdf_col:
        leaderboard_entries = [
            " | ".join(
                f"{column}: {row.get(column, '—')}"
                for column in [
                    "Rank",
                    "Supplier",
                    "AI weighted score",
                    "PPI (peer benchmark)",
                    "Quoted total",
                    "Grade",
                    "Quantity (MT)",
                ]
            )
            for row in filtered.to_dict(orient="records")
        ]
        st.download_button(
            "Download leaderboard PDF",
            data=build_report_pdf(
                "Forge Steel RFP AI - Supplier Leaderboard",
                [
                    f"Evaluation: {run['run_id']}",
                    f"Ranking policy: {run['ranking_policy']}",
                ],
                [("Supplier rankings", leaderboard_entries or ["No supplier results."])],
            ),
            file_name=f"{run['run_id']}_leaderboard.pdf",
            mime="application/pdf",
            width="stretch",
        )
    with json_col:
        st.download_button(
            "Download complete run JSON",
            data=json.dumps(run, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"{run['run_id']}.json",
            mime="application/json",
            width="stretch",
        )
    if run.get("warnings"):
        with st.expander("Run warnings"):
            for warning in run["warnings"]:
                st.warning(warning)


def render_comparison(run: dict[str, Any]) -> None:
    st.title("Compare & review")
    render_run_badges(run)
    suppliers = run["suppliers"]
    if not suppliers:
        st.info("This run has no completed supplier evaluations.")
        return
    selected_names = st.multiselect(
        "Suppliers to compare",
        [item["name"] for item in suppliers],
        default=[item["name"] for item in suppliers[: min(4, len(suppliers))]],
    )
    criteria = [item for item in run["criteria"] if item.get("active", True)]
    selected_criteria = st.multiselect(
        "Criteria",
        [item["name"] for item in criteria],
        default=[item["name"] for item in criteria],
    )
    comparison = []
    for supplier in suppliers:
        if supplier["name"] not in selected_names:
            continue
        for criterion in criteria:
            if criterion["name"] not in selected_criteria:
                continue
            result = next((r for r in supplier.get("criteria_results", []) if r["criterion_id"] == criterion["id"]), None)
            if result:
                comparison.append({
                    "Supplier": supplier["name"],
                    "Criterion": criterion["name"],
                    "AI score": (
                        f"{result['ai_score']:g}/{criterion['max_score']}"
                        if result.get("ai_score") is not None else "Not scored"
                    ),
                    "Weight": f"{criterion['weight']:g}%",
                    "Weighted contribution": (
                        f"{float(result['ai_score']) / float(criterion['max_score']) * float(criterion['weight']):.2f}"
                        if result.get("ai_score") is not None else "—"
                    ),
                    "Benchmark": (
                        f"{result['benchmark']:g}/{criterion['max_score']}"
                        if result.get("benchmark") is not None else "—"
                    ),
                    "Criterion gap": (
                        f"{result['criterion_gap']:+g}"
                        if result.get("criterion_gap") is not None else "—"
                    ),
                    "Relative performance": (
                        f"{result['relative_performance_pct']:g}%"
                        if result.get("relative_performance_pct") is not None else "Undefined (benchmark is zero or unavailable)"
                    ),
                    "Status": result.get("ai_status"),
                    "Evidence": result.get("evidence") or "This information does not appear in the uploaded documents.",
                    "Source": f"{result.get('source_file') or '—'}{', page ' + str(result['page']) if result.get('page') else ''}",
                    "AI explanation": result.get("justification"),
                })
    st.dataframe(pd.DataFrame(comparison), width="stretch", hide_index=True)
    st.subheader("What-if analysis")
    st.caption(
        "Hypothetical only: adjust relative criterion weights to explore ranking "
        "sensitivity. This does not change or save the official run."
    )
    weight_rows = [
        {"Criterion": item["name"], "Weight (%)": float(item["weight"])}
        for item in criteria
    ]
    edited_weights = st.data_editor(
        pd.DataFrame(weight_rows),
        hide_index=True,
        disabled=["Criterion"],
        column_config={
            "Weight (%)": st.column_config.NumberColumn(
                min_value=0.0, max_value=100.0, step=1.0
            )
        },
        key=f"scenario_weights_{run['run_id']}",
        width="stretch",
    )
    scenario_weight_map = {
        item["id"]: (
            float(edited_weights.iloc[index]["Weight (%)"])
            if pd.notna(edited_weights.iloc[index]["Weight (%)"])
            else 0.0
        )
        for index, item in enumerate(criteria)
    }
    try:
        scenario = evaluate_scenario(suppliers, run["criteria"], scenario_weight_map)
        sensitivity = analyze_sensitivity(
            suppliers, run["criteria"], simulations=200, seed=42
        )
    except ValueError as exc:
        st.error(f"Cannot calculate the hypothetical scenario: {exc}")
    else:
        st.dataframe(
            [
                {
                    "Scenario rank": item.get("rank"),
                    "Supplier": item.get("name"),
                    "Baseline rank": item.get("baseline_rank"),
                    "Rank movement": item.get("rank_movement"),
                    "Weighted score": item.get("absolute_score"),
                    "PPI": item.get("ppi"),
                }
                for item in scenario["ranked_suppliers"]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Under seeded ±10 percentage-point weight perturbations, the baseline "
            f"winner remained first in {sensitivity['winner_stability_pct']}% "
            f"of {sensitivity['simulations']} hypothetical trials."
        )
    st.subheader("Supplier findings")
    for supplier in suppliers:
        if supplier["name"] not in selected_names:
            continue
        with st.expander(f"{supplier['name']} — steel offer, quote, evidence, and risks"):
            st.markdown("**Supplier summary**")
            st.write(supplier.get("summary") or "This information does not appear in the uploaded documents.")
            technical = supplier.get("technical_offer", {})
            quote = supplier.get("commercial_quote", {})
            delivery = supplier.get("delivery", {})
            st.markdown("**Technical offer and delivery**")
            st.json({
                "material": technical.get("material_description"),
                "offered_grade": technical.get("offered_grade"),
                "offered_specification": technical.get("offered_specification"),
                "offered_quantity_mt": technical.get("offered_quantity_mt"),
                "technical_status": technical.get("compliance_status"),
                "deviations": technical.get("deviations", []),
                "quantity_available_mt": delivery.get("quantity_available_mt"),
                "first_delivery_date": delivery.get("first_delivery_date"),
                "completion_date": delivery.get("completion_date"),
                "delivery_location": delivery.get("delivery_location"),
            })
            extraction_evidence = []
            for section_name, section in (
                ("Technical offer", technical),
                ("Commercial quotation", quote),
                ("Delivery", delivery),
            ):
                for citation in section.get("evidence", []):
                    extraction_evidence.append({
                        "Section": section_name,
                        "Evidence quote": citation.get("quote"),
                        "Source": citation.get("source_file"),
                        "Page": citation.get("page"),
                    })
            if extraction_evidence:
                st.markdown("**Steel offer evidence citations**")
                st.dataframe(pd.DataFrame(extraction_evidence), width="stretch", hide_index=True)
            st.markdown("**Commercial quotation (unverified extraction)**")
            st.json({
                "currency": quote.get("currency"),
                "basic_rate_per_mt": quote.get("basic_rate_per_mt"),
                "freight_per_mt": quote.get("freight_per_mt"),
                "tax_amount": quote.get("tax_amount"),
                "loading_unloading_amount": quote.get("loading_unloading_amount"),
                "other_charges": quote.get("other_charges"),
                "total_quoted_amount": quote.get("total_quoted_amount"),
                "delivery_basis": quote.get("delivery_basis"),
                "quote_valid_until": quote.get("quote_valid_until"),
                "payment_terms": quote.get("payment_terms"),
                "exclusions": quote.get("exclusions", []),
                "comparability_status": quote.get("comparability_status"),
            })
            quality_rows = [
                {
                    "Document": item.get("document_type"),
                    "Provided": item.get("provided"),
                    "Heat/batch": item.get("heat_or_batch_reference"),
                    "Verification": item.get("verification_status"),
                    "Source": item.get("source_file"),
                    "Page": item.get("page"),
                    "Quote": item.get("evidence_quote"),
                }
                for item in supplier.get("quality_documents", [])
            ]
            if quality_rows:
                st.markdown("**Quality documents**")
                st.dataframe(pd.DataFrame(quality_rows), width="stretch", hide_index=True)
            findings = []
            for result in supplier.get("criteria_results", []):
                if result.get("risks") or result.get("clarifications"):
                    findings.append({
                        "Criterion": result.get("criterion_name", result["criterion_id"]),
                        "Risks": "; ".join(result.get("risks", [])) or "None stated",
                        "Clarifications": "; ".join(result.get("clarifications", [])) or "None requested",
                    })
            if findings:
                st.dataframe(pd.DataFrame(findings), width="stretch", hide_index=True)
            else:
                st.caption("No criterion-level risks or clarification requests were recorded.")
    chart_data = []
    for supplier in suppliers:
        if supplier["name"] in selected_names:
            for criterion in criteria:
                if criterion["name"] in selected_criteria:
                    result = next((r for r in supplier.get("criteria_results", []) if r["criterion_id"] == criterion["id"]), None)
                    if result and result.get("ai_score") is not None:
                        chart_data.append({
                            "Supplier": supplier["name"],
                            "Criterion": criterion["name"],
                            "Normalized score (%)": round(
                                float(result["ai_score"])
                                / criterion["max_score"] * 100,
                                2,
                            ),
                        })
    if chart_data:
        st.subheader("Normalized criterion comparison")
        st.caption("Each score is normalized as score ÷ criterion maximum × 100. Table values remain available for precise comparison.")
        st.plotly_chart(
            px.line_polar(
                pd.DataFrame(chart_data),
                r="Normalized score (%)",
                theta="Criterion",
                color="Supplier",
                line_close=True,
                range_r=[0, 100],
                markers=True,
            ),
            width="stretch",
        )
def render_criteria_settings() -> None:
    st.title("Criteria")
    st.subheader("Weighted evaluation criteria")
    st.caption(
        "The default starter template follows the project's five example categories and weights: "
        "Technical Capability (30%), Implementation Plan (20%), Commercial Value (20%), "
        "Security & Compliance (20%), and Support & Experience (10%). Edit the template below "
        "to match the approved procurement policy."
    )
    st.warning("Example weights only — review and authorize them against the project's procurement policy. Active weighted criteria must total exactly 100%.")
    st.caption(
        "AI-proposed scores are used directly in weighted totals and ranking. Missing criterion "
        "scores contribute zero to the weighted total, are flagged for review, and are excluded from PPI."
    )
    criteria = st.session_state.get("criteria", STARTER_CRITERIA)
    df = pd.DataFrame(criteria)
    edited = st.data_editor(
        df,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "active": st.column_config.CheckboxColumn("Active"),
            "weight": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0, step=1.0),
            "max_score": st.column_config.NumberColumn("Maximum score", min_value=1, max_value=100, step=1),
        },
        key="criteria_editor",
    )
    if st.button("Save criteria template"):
        edited_items = edited.to_dict(orient="records")
        try:
            for item in edited_items:
                item["weight"] = float(item["weight"])
                item["max_score"] = int(item["max_score"])
                item["active"] = bool(item["active"])
            errors = validate_criteria(edited_items)
        except (TypeError, ValueError, KeyError) as exc:
            errors = [f"Invalid criteria values: {exc}"]
            edited_items = criteria
        if errors:
            for error in errors:
                st.error(error)
        else:
            st.session_state.criteria = edited_items
            save_criteria(edited_items)
            st.success("Criteria template saved to SQLite. Each new run snapshots it.")
            st.rerun()


def render_settings() -> None:
    st.title("LLM Settings")
    st.write("Choose the provider and model for live steel quotation evaluations. Your selection is saved and used for each new run unless you change it.")
    provider = st.selectbox(
        "LLM provider",
        list(PROVIDERS),
        key="llm_provider",
        on_change=persist_selected_provider,
    )
    embedding_models_by_provider = EMBEDDING_CONFIG
    embedding_model = None
    if provider in embedding_models_by_provider and embedding_models_by_provider[provider][0]:
        embedding_models, embedding_default = embedding_models_by_provider[provider]
        embedding_key = f"rag_embedding_model_{provider}"
        if st.session_state.get(embedding_key) not in embedding_models:
            st.session_state[embedding_key] = embedding_default
        embedding_model = st.selectbox(
            f"{provider} Q&A embedding model",
            embedding_models,
            key=embedding_key,
            help="Uses the same API key configured for evaluations with this provider.",
        )
        st.caption(
            f"Q&A retrieval uses {provider} `{embedding_model}` with BM25 and semantic "
            "similarity fused using reciprocal rank fusion (RRF)."
        )
    else:
        st.caption(
            "Claude does not provide a first-party embeddings API. Q&A uses BM25 keyword "
            "retrieval and Claude generates the answer with the same API key."
        )
    models = model_options(provider)
    model_key = f"model_{provider}"
    if st.session_state.get(model_key) not in models:
        stored_model = get_app_setting(model_key)
        st.session_state[model_key] = stored_model if stored_model in models else models[0]
    model = st.selectbox(
        "Model",
        models,
        key=model_key,
        on_change=persist_selected_model,
        args=(provider,),
    )
    env_name = secret_name(provider)
    st.caption(
        f"Create a key from the provider account, then enter it below. "
        f"For deployment, configure `{env_name}` in Streamlit secrets. "
        "Keys typed here stay in this browser session and are never saved in run data or SQLite."
    )
    st.text_input(
        f"{provider} API key",
        type="password",
        value=st.session_state.get(f"api_key_session_{provider}", ""),
        key=f"api_key_input_{provider}",
        on_change=persist_api_key,
        args=(provider, st.session_state),
        help="Never share your provider API key or commit it to source control.",
    )
    if os.getenv(env_name):
        st.caption("An API key is loaded from the local environment; its value is not displayed.")
    st.link_button(f"Get a {provider} API key", str(PROVIDERS[provider]["link"]))
    if st.button(f"Test {provider} connection"):
        key = st.session_state.get(f"api_key_session_{provider}") or configured_api_key(provider)
        if not key:
            st.error(f"Not configured. Add a {provider} API key.")
        else:
            with st.spinner(f"Testing {provider} connection…"):
                try:
                    test_connection(provider, key, model)
                    st.success(f"Connected to {provider} using {model}.")
                except Exception as exc:
                    st.error(str(exc))
    st.info(
        f"Live evaluations transmit the confirmed uploaded PDF text excerpts and supplier metadata "
        f"to {provider}. Demo mode makes no provider request."
    )
    st.caption("The local SQLite database stores evaluation outputs, extracted page text for the Q&A Bot, cited snippets, and audit events. Original PDFs and API keys are not stored.")


def render_qa_bot(run: dict[str, Any]) -> None:
    st.title("Q&A Bot")
    runs = list_runs()
    if runs:
        run_ids = [item["run_id"] for item in runs]
        active_run_id = st.session_state.get("active_run_id", run["run_id"])
        if st.session_state.get("qa_selected_run_id") != active_run_id:
            st.session_state.qa_selected_run_id = active_run_id
        selected_run_id = st.selectbox(
            "Select evaluation",
            run_ids,
            format_func=lambda run_id: next(
                (
                    f"{item['run_id']} · {'Demo' if item['is_demo'] else item['mode']}"
                    for item in runs
                    if item["run_id"] == run_id
                ),
                run_id,
            ),
            key="qa_selected_run_id",
        )
        if selected_run_id != active_run_id:
            st.session_state.active_run_id = selected_run_id
            st.rerun()
        selected_run = load_run(selected_run_id)
        if selected_run is None:
            st.error("The selected evaluation could not be loaded.")
            return
        run = selected_run

    st.info(
        "**How answers are made:** We search the selected evaluation's shared RFP "
        "and, if selected, that supplier's documents. BM25 keyword search is combined "
        "with semantic embeddings (Claude uses BM25 only). The selected language model "
        "then writes an answer with document and page citations. Other suppliers and "
        "the public internet are not searched."
    )
    if run.get("is_demo"):
        st.info("The completed demo contains synthetic review examples, not a document corpus. Start a live run to use document Q&A.")
        return

    documents = load_rag_corpus(run["run_id"])
    if not documents:
        st.info("No extracted document text is available for this run.")
        return
    suppliers = run.get("suppliers", [])
    scope_labels = ["Project requirements only"] + [item["name"] for item in suppliers]
    selected_scope = st.selectbox("Chat about", scope_labels)
    selected_supplier = next(
        (item for item in suppliers if item["name"] == selected_scope),
        None,
    )
    supplier_id = selected_supplier["supplier_id"] if selected_supplier else None
    scoped_docs = scope_documents(documents, supplier_id)
    chunks = chunk_documents(scoped_docs)
    if not chunks:
        st.info("There is no readable evidence in the selected document scope.")
        return

    provider = selected_llm_provider()
    model = selected_llm_model(provider, model_options(provider))
    if provider not in PROVIDERS or not model:
        st.error("This run does not have a supported language model configured for Q&A.")
        return
    provider_key = st.session_state.get(f"api_key_session_{provider}") or configured_api_key(provider)
    embedding_provider = provider
    embedding_key = provider_key
    embedding_model = st.session_state.get(
        f"rag_embedding_model_{provider}",
        EMBEDDING_CONFIG.get(provider, ([], DEFAULT_EMBEDDING_MODEL))[1],
    )
    if embedding_provider == "Claude":
        embedding_model = ""
    elif embedding_model is None:
        embedding_model = (
            EMBEDDING_CONFIG.get(embedding_provider, ([], DEFAULT_EMBEDDING_MODEL))[1]
            or ""
        )
    st.caption(
        f"Answer model: {provider} · `{model}`  |  Evaluation model: "
        f"{run.get('provider', 'unknown')} · `{run.get('model', 'unknown')}`  |  "
        f"{len(scoped_docs)} documents in this chat"
    )
    with st.expander("Documents in this Q&A scope"):
        for document in scoped_docs:
            pages = ", ".join(str(page["page"]) for page in document["pages"])
            supplier_label = (
                f" · supplier: {selected_supplier['name']}"
                if selected_supplier and document.get("scope") != "shared"
                else " · shared requirements"
            )
            st.caption(f"{document['filename']}{supplier_label} · pages {pages}")
    if embedding_provider == "Claude":
        consent_text = (
            f"I authorize sending the question and retrieved excerpts to {provider} ({model}) "
            "to generate an answer. BM25 keyword retrieval runs locally."
        )
    else:
        consent_text = (
            f"I authorize sending extracted document chunks and each question to "
            f"{embedding_provider} ({embedding_model}) for embeddings, and the question plus "
            f"top retrieved excerpts to {provider} ({model}) to generate an answer."
        )
    consent = st.checkbox(
        consent_text,
        key=f"qa_consent_{run['run_id']}_{supplier_id or 'shared'}",
    )
    if not embedding_key:
        st.warning(f"Configure the {embedding_provider} API key in LLM Settings for Q&A.")
    if not provider_key:
        st.warning(f"Configure the {provider} API key in LLM Settings for answer generation.")

    history_key = f"qa_history_{run['run_id']}_{supplier_id or 'shared'}"
    history = st.session_state.setdefault(history_key, [])
    if history:
        csv_column, pdf_column = st.columns(2)
        export_stem = f"{run['run_id']}_{supplier_id or 'requirements'}_qa"
        with csv_column:
            st.download_button(
                "Export chat as CSV",
                data=build_qa_csv(history),
                file_name=f"{export_stem}.csv",
                mime="text/csv",
                width="stretch",
            )
        with pdf_column:
            st.download_button(
                "Export chat as PDF",
                data=build_qa_pdf(
                    history,
                    run_id=run["run_id"],
                    scope=selected_scope,
                ),
                file_name=f"{export_stem}.pdf",
                mime="application/pdf",
                width="stretch",
            )
    else:
        st.caption("Your chat exports will appear here after the first answer.")
    for exchange in history:
        with st.chat_message("user"):
            st.markdown(exchange["question"])
        with st.chat_message("assistant"):
            st.markdown(exchange["answer"])
            for index, source in enumerate(exchange.get("sources", []), start=1):
                st.caption(
                    f"[S{index}] {source['filename']} · page {source['page']}"
                )

    question = st.chat_input(
        "Ask about requirements, evidence, deviations, or missing information",
        disabled=not (consent and embedding_key and provider_key),
    )
    if not question:
        return

    cache_material = "\n".join(
        f"{chunk['source_id']}:{chunk['page']}:{chunk['text']}" for chunk in chunks
    )
    corpus_key = hashlib.sha256(
        f"{run['run_id']}:{supplier_id}:{embedding_provider}:{embedding_model}:{cache_material}".encode("utf-8")
    ).hexdigest()
    embedding_cache = st.session_state.setdefault("rag_embedding_cache", {})
    try:
        retrieval_status = (
            "Searching evidence with BM25 keyword retrieval…"
            if embedding_provider == "Claude"
            else "Retrieving evidence with BM25 + the bi-encoder, then fusing ranks…"
        )
        with st.spinner(retrieval_status):
            document_embeddings = embedding_cache.get(corpus_key)
            if document_embeddings is None and embedding_provider != "Claude":
                document_embeddings = embed_texts(
                    [chunk["text"] for chunk in chunks],
                    api_key=embedding_key,
                    model=embedding_model,
                    input_type="search_document",
                    provider=embedding_provider,
                )
                embedding_cache[corpus_key] = document_embeddings
            retrieved = retrieve_fused(
                question,
                chunks,
                api_key=embedding_key,
                model=embedding_model,
                embedding_provider=embedding_provider,
                document_embeddings=document_embeddings,
                limit=6,
            )
            if retrieved:
                answer = call_llm_text(
                    provider,
                    provider_key,
                    model,
                    build_qa_prompt(question, retrieved),
                )
                validate_answer_citations(answer, retrieved)
            else:
                answer = "This information does not appear in the uploaded documents for the selected scope."
    except (RuntimeError, ValueError) as exc:
        st.error(f"Could not answer this question: {exc}")
        return

    history.append(
        {
            "question": question,
            "answer": answer,
            "sources": retrieved,
        }
    )
    st.rerun()


def configured_api_key(provider: str) -> str | None:
    from streamlit.errors import StreamlitSecretNotFoundError

    environment_value = os.getenv(secret_name(provider))
    if environment_value:
        return environment_value
    try:
        value = st.secrets.get(secret_name(provider))
        return str(value) if value else None
    except StreamlitSecretNotFoundError:
        return None


def selected_llm_provider() -> str:
    provider = st.session_state.get("llm_provider") or get_app_setting("llm_provider")
    if provider not in PROVIDERS:
        provider = "Cohere.ai"
        st.session_state["llm_provider"] = provider
    return provider


def selected_llm_model(provider: str, models: list[str]) -> str:
    model_key = f"model_{provider}"
    selected = st.session_state.get(model_key)
    if selected not in models:
        stored_model = get_app_setting(model_key)
        selected = stored_model if stored_model in models else models[0]
        st.session_state[model_key] = selected
    return selected


def persist_selected_provider() -> None:
    provider = st.session_state.get("llm_provider")
    if provider in PROVIDERS:
        save_app_setting("llm_provider", provider)


def persist_selected_model(provider: str) -> None:
    model = st.session_state.get(f"model_{provider}")
    if model in model_options(provider):
        save_app_setting(f"model_{provider}", model)


def render_run_badges(run: dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("RFP_RUN_ID", run["run_id"])
    c2.metric("Mode", run["mode"])
    c3.metric("Status", run["status"])
    if run.get("is_demo"):
        st.warning("SYNTHETIC DEMO DATA — fictional suppliers, evidence, scores, and prices. Not procurement advice.")
    else:
        st.caption(f"Provider: {run.get('provider')} · Model: {run.get('model')} · Created: {run.get('created_at')}")


def render_history(run: dict[str, Any]) -> None:
    st.title("Run history")
    st.caption("Inspect saved evaluations, criteria, warnings, LangGraph traces, and audit events. This page reads SQLite only and never calls an LLM.")
    render_run_badges(run)
    runs = list_runs()
    st.subheader("Saved runs")
    st.dataframe(
        [
            {
                "Run ID": item["run_id"],
                "Created": item["created_at"],
                "Mode": item["mode"],
                "Status": item["status"],
                "Synthetic demo": item["is_demo"],
            }
            for item in runs
        ],
        hide_index=True,
        width="stretch",
    )
    run_export_rows = []
    criterion_report_entries = []
    for supplier in run.get("suppliers", []):
        results = supplier.get("criteria_results", [])
        if not results:
            run_export_rows.append(
                {
                    "Run ID": run["run_id"],
                    "Supplier": supplier.get("name"),
                    "Rank": supplier.get("rank"),
                    "Supplier status": "Failed" if supplier.get("evaluation_error") else "Complete",
                    "Weighted score": supplier.get("absolute_score"),
                    "PPI": supplier.get("ppi"),
                    "Criterion": "",
                    "AI score": "",
                    "Maximum score": "",
                    "Criterion status": "",
                    "Evidence": "",
                    "Source": "",
                    "Page": "",
                    "Explanation": "",
                }
            )
        for result in results:
            source = result.get("source_file") or ""
            run_export_rows.append(
                {
                    "Run ID": run["run_id"],
                    "Supplier": supplier.get("name"),
                    "Rank": supplier.get("rank"),
                    "Supplier status": "Failed" if supplier.get("evaluation_error") else "Complete",
                    "Weighted score": supplier.get("absolute_score"),
                    "PPI": supplier.get("ppi"),
                    "Criterion": result.get("criterion_name") or result.get("criterion_id"),
                    "AI score": result.get("ai_score"),
                    "Maximum score": result.get("max_score"),
                    "Criterion status": result.get("ai_status"),
                    "Evidence": result.get("evidence"),
                    "Source": source,
                    "Page": result.get("page"),
                    "Explanation": result.get("justification"),
                }
            )
            evidence = result.get("evidence") or "No evidence was recorded."
            citation = source
            if result.get("page"):
                citation += f", page {result['page']}"
            criterion_report_entries.append(
                f"{supplier.get('name')} | {result.get('criterion_name') or result.get('criterion_id')} "
                f"| score {result.get('ai_score', '—')}/{result.get('max_score', '—')} "
                f"| {result.get('ai_status', 'UNKNOWN')} | {evidence} "
                f"| Source: {citation or '—'}"
            )

    history_csv = pd.DataFrame(run_export_rows).to_csv(index=False).encode("utf-8-sig")
    supplier_report_entries = [
        f"Rank {supplier.get('rank', '—')} | {supplier.get('name', 'Unknown supplier')} "
        f"| {'Failed' if supplier.get('evaluation_error') else 'Complete'} "
        f"| Weighted score: {supplier.get('absolute_score', '—')} "
        f"| PPI: {supplier.get('ppi', '—')}"
        + (
            f" | Error: {supplier['evaluation_error']}"
            if supplier.get("evaluation_error")
            else ""
        )
        for supplier in run.get("suppliers", [])
    ]
    history_pdf = build_report_pdf(
        "Forge Steel RFP AI - Evaluation Report",
        [
            f"Evaluation: {run['run_id']}",
            f"Mode: {run.get('mode', '—')} | Status: {run.get('status', '—')}",
            f"Created: {run.get('created_at', '—')}",
            f"Provider/model: {run.get('provider', '—')} / {run.get('model', '—')}",
        ],
        [
            ("Supplier scorecards", supplier_report_entries or ["No suppliers were recorded."]),
            ("Criterion evidence", criterion_report_entries or ["No criterion results were recorded."]),
        ],
    )
    csv_col, pdf_col, json_col = st.columns(3)
    with csv_col:
        st.download_button(
            "Export selected run as CSV",
            data=history_csv,
            file_name=f"{run['run_id']}.csv",
            mime="text/csv",
            width="stretch",
        )
    with pdf_col:
        st.download_button(
            "Export selected run as PDF",
            data=history_pdf,
            file_name=f"{run['run_id']}.pdf",
            mime="application/pdf",
            width="stretch",
        )
    with json_col:
        st.download_button(
            "Export selected run as JSON",
            data=json.dumps(run, ensure_ascii=False, indent=2),
            file_name=f"{run['run_id']}.json",
            mime="application/json",
            width="stretch",
        )
    st.subheader("Supplier scorecards")
    st.dataframe(
        [
            {
                "Rank": supplier.get("rank"),
                "Supplier": supplier.get("name"),
                "Status": "Failed" if supplier.get("evaluation_error") else "Complete",
                "Weighted score": supplier.get("absolute_score"),
                "PPI": supplier.get("ppi"),
                "Error": supplier.get("evaluation_error"),
            }
            for supplier in run.get("suppliers", [])
        ],
        hide_index=True,
        width="stretch",
    )
    warnings = evaluation_warnings(run["run_id"])
    with st.expander(f"Warnings ({len(warnings)})", expanded=bool(warnings)):
        if warnings:
            st.dataframe(warnings, hide_index=True, width="stretch")
        else:
            st.caption("No warnings were recorded.")

    traces = workflow_traces(run["run_id"])
    with st.expander(f"Workflow trace ({len(traces)} stages)"):
        if traces:
            st.dataframe(
                [
                    {
                        **{key: value for key, value in item.items() if key != "details"},
                        "details": json.dumps(item["details"], ensure_ascii=False),
                    }
                    for item in traces
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("This run has no LangGraph trace (for example, an older or synthetic run).")

    events = audit_events(run["run_id"])
    with st.expander(f"Audit events ({len(events)})"):
        if events:
            st.dataframe(
                [
                    {
                        **{key: value for key, value in item.items() if key != "details"},
                        "details": json.dumps(item["details"], ensure_ascii=False),
                    }
                    for item in events
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No workflow audit events were recorded.")

    chain_valid, invalid_sequence = verify_audit_chain()
    if chain_valid:
        st.success("Tamper-evident audit chain verified against saved run snapshots.")
    else:
        st.error(
            "Audit-chain verification failed"
            + (f" at sequence {invalid_sequence}." if invalid_sequence else ".")
        )
    with st.expander("Tamper-evident audit chain"):
        chain_entries = list_audit_chain(run["run_id"])
        if chain_entries:
            st.caption(
                "Hashes make later changes detectable; they do not prevent an "
                "administrator from replacing the local SQLite database."
            )
            st.dataframe(chain_entries, hide_index=True, width="stretch")
        else:
            st.caption("No hash-chain entries were recorded for this run.")

    criteria = criterion_evaluations(run["run_id"])
    with st.expander(f"Criterion evaluations ({len(criteria)})"):
        if criteria:
            st.dataframe(
                [
                    {
                        "Supplier ID": item["supplier_id"],
                        "Criterion": item["criterion_id"],
                        "Score": item["score"],
                        "Status": item["status"],
                        "Justification": item["evaluation"].get("justification"),
                        "Source": item["evaluation"].get("source_file"),
                        "Page": item["evaluation"].get("page"),
                    }
                    for item in criteria
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No per-criterion records were saved for this run.")


def main() -> None:
    st.session_state.criteria = load_criteria() or [dict(item) for item in STARTER_CRITERIA]
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = get_app_setting("llm_provider") or "Cohere.ai"
    pages = ["Home", "New evaluation", "Leaderboard", "Compare & review", "History", "Q&A Bot", "Criteria", "LLM Settings"]
    pending_page = st.session_state.pop("pending_workspace_page", None)
    if pending_page in pages:
        st.session_state["workspace_page"] = pending_page
    current_page = st.session_state.get("workspace_page", st.session_state.get("page", "Home"))
    if current_page not in pages:
        current_page = "Home"
    with st.sidebar:
        st.title("Forge Steel RFP AI")
        st.caption("RFP procurement review workspace")
        page = st.radio(
            "Workspace",
            pages,
            index=pages.index(current_page),
            label_visibility="collapsed",
            key="workspace_page",
        )
        runs = list_runs()
        if runs:
            run_ids = [item["run_id"] for item in runs]
            default_index = run_ids.index(st.session_state.active_run_id) if st.session_state.get("active_run_id") in run_ids else 0
            selected = st.selectbox(
                "Saved runs",
                runs,
                index=default_index,
                format_func=lambda item: f"{item['run_id']} · {'Demo' if item['is_demo'] else item['mode']}",
            )
            st.session_state.active_run_id = selected["run_id"]
        else:
            st.caption("No saved evaluations yet.")
        st.divider()
        st.caption("AI scores and findings are decision support; the application does not award contracts.")
    if page == "Home":
        render_home()
    elif page == "New evaluation":
        render_new_evaluation()
    elif page == "Criteria":
        render_criteria_settings()
    elif page == "LLM Settings":
        render_settings()
    else:
        run_id = st.session_state.get("active_run_id")
        run = load_run(run_id) if run_id else None
        if run is not None:
            update_rankings(run)
        if run is None:
            st.title(page)
            st.info("Create an evaluation or explore the completed demo from Home.")
        elif page == "Leaderboard":
            render_leaderboard(run)
        elif page == "History":
            render_history(run)
        elif page == "Q&A Bot":
            render_qa_bot(run)
        else:
            render_comparison(run)


if __name__ == "__main__":
    main()
