"""Deterministic validation, scoring, and ranking helpers."""

from __future__ import annotations

import math
from typing import Any


STARTER_CRITERIA: list[dict[str, Any]] = [
    {
        "id": "C01",
        "name": "Technical Capability",
        "description": "For this steel-procurement example, assess the offered product's technical fit against the buyer's stated grade, specification, dimensions, and requirements.",
        "weight": 30.0,
        "max_score": 10,
        "rubric": "Assess only technical fit and capability supported by supplier evidence. Missing facts are NOT_PROVIDED, not assumed non-compliant.",
        "active": True,
    },
    {
        "id": "C02",
        "name": "Implementation Plan",
        "description": "For this steel-procurement example, assess delivery timeline, quantities, milestones or phases, staffing/capacity, and stated delivery risks.",
        "weight": 20.0,
        "max_score": 10,
        "rubric": "Assess stated timeline, milestones, resourcing/capacity, and risk plan against buyer requirements. Do not infer missing commitments.",
        "active": True,
    },
    {
        "id": "C03",
        "name": "Commercial Value",
        "description": "Comparable price per MT, total quoted amount, freight, taxes, unloading, other charges, exclusions, and payment terms.",
        "weight": 20.0,
        "max_score": 10,
        "rubric": "Assess pricing clarity, total cost, assumptions, taxes, freight, and exclusions. Compare totals only when scope is comparable; do not infer missing charges.",
        "active": True,
    },
    {
        "id": "C04",
        "name": "Security & Compliance",
        "description": "Assess supplier evidence for relevant controls, certifications, privacy, and auditability; include only compliance evidence relevant to the stated procurement requirements.",
        "weight": 20.0,
        "max_score": 10,
        "rubric": "Score documented controls, certifications, privacy, and auditability relevant to the request. Mark unsupported or absent claims unknown; do not treat document presence as independent verification.",
        "active": True,
    },
    {
        "id": "C05",
        "name": "Support & Experience",
        "description": "Assess supplier support model, warranty or replacement support, and relevant project references.",
        "weight": 10.0,
        "max_score": 10,
        "rubric": "Assess documented support, warranty/replacement terms, and relevant references. Supplier claims are not independently verified unless supporting evidence is provided.",
        "active": True,
    },
]


def validate_criteria(criteria: list[dict[str, Any]]) -> list[str]:
    active = [criterion for criterion in criteria if criterion.get("active", True)]
    errors: list[str] = []
    if not active:
        errors.append("At least one weighted criterion must be active.")
    total = sum(float(item.get("weight", 0)) for item in active)
    if abs(total - 100.0) > 1e-6:
        errors.append(f"Active weighted criteria must total exactly 100% (currently {total:g}%).")
    seen: set[str] = set()
    for item in criteria:
        criterion_id = str(item.get("id", "")).strip()
        if not criterion_id or criterion_id in seen:
            errors.append("Every criterion must have a unique, non-empty ID.")
            break
        seen.add(criterion_id)
        if float(item.get("weight", 0)) < 0:
            errors.append(f"{item.get('name', criterion_id)} has a negative weight.")
        if int(item.get("max_score", 0)) <= 0:
            errors.append(f"{item.get('name', criterion_id)} must have a maximum score above zero.")
        if not str(item.get("rubric", "")).strip():
            errors.append(f"{item.get('name', criterion_id)} needs a written scoring rubric.")
    return errors


def weighted_score(
    criteria: list[dict[str, Any]],
    scores: dict[str, float | None],
    *,
    missing_as_zero: bool = True,
) -> float | None:
    """Return a 0–100 score; missing values are explicit and configurable."""
    active = [item for item in criteria if item.get("active", True)]
    if not active:
        return None
    contributions = 0.0
    for item in active:
        score = scores.get(item["id"])
        if score is None:
            if not missing_as_zero:
                return None
            score = 0.0
        maximum = float(item["max_score"])
        if not 0 <= float(score) <= maximum:
            raise ValueError(f"Score for {item['id']} must be between 0 and {maximum:g}.")
        contributions += float(score) / maximum * float(item["weight"])
    return round(contributions, 2)


def rank_suppliers(suppliers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank suppliers by peer-relative performance, then deterministic tie-breaks."""
    ordered = sorted(
        suppliers,
        key=lambda item: (
            -(float(item["ppi"]) if item.get("ppi") is not None else float("-inf")),
            str(item.get("submission_date") or "9999-12-31"),
            -float(item.get("experience") or 0),
            str(item.get("name", "")).casefold(),
        ),
    )
    for index, supplier in enumerate(ordered, start=1):
        supplier["rank"] = index
    return ordered


def calculate_ppi(
    criteria: list[dict[str, Any]],
    scores: dict[str, float | None],
    benchmarks: dict[str, float | None],
) -> float | None:
    """Calculate weighted relative performance against each criterion's peer benchmark."""
    active = [item for item in criteria if item.get("active", True)]
    relative_sum = 0.0
    weight_sum = 0.0
    for criterion in active:
        score = scores.get(criterion["id"])
        benchmark = benchmarks.get(criterion["id"])
        if score is None or benchmark in (None, 0):
            continue
        relative_sum += (float(score) / float(benchmark) * 100) * float(criterion["weight"])
        weight_sum += float(criterion["weight"])
    return round(relative_sum / weight_sum, 2) if weight_sum else None


def apply_peer_metrics(
    criteria: list[dict[str, Any]],
    suppliers: list[dict[str, Any]],
) -> None:
    """Attach criterion benchmarks, score gaps, relative percentages, and weighted PPI."""
    active = [item for item in criteria if item.get("active", True)]
    for criterion in active:
        numeric_scores = [
            result.get("final_score")
            for supplier in suppliers
            for result in supplier.get("criteria_results", [])
            if result.get("criterion_id") == criterion["id"]
            and isinstance(result.get("final_score"), (int, float))
        ]
        benchmark = max(numeric_scores) if numeric_scores else None
        for supplier in suppliers:
            result = next(
                (
                    item
                    for item in supplier.get("criteria_results", [])
                    if item.get("criterion_id") == criterion["id"]
                ),
                None,
            )
            if result is None:
                continue
            score = result.get("final_score")
            result["benchmark"] = benchmark
            result["criterion_gap"] = (
                round(float(score) - float(benchmark), 2)
                if score is not None and benchmark is not None else None
            )
            result["relative_performance_pct"] = (
                round(float(score) / float(benchmark) * 100, 2)
                if score is not None and benchmark not in (None, 0) else None
            )

    benchmarks = {
        criterion["id"]: next(
            (
                result.get("benchmark")
                for supplier in suppliers
                for result in supplier.get("criteria_results", [])
                if result.get("criterion_id") == criterion["id"]
            ),
            None,
        )
        for criterion in active
    }
    for supplier in suppliers:
        supplier["ppi"] = calculate_ppi(
            criteria,
            {
                result["criterion_id"]: result.get("final_score")
                for result in supplier.get("criteria_results", [])
            },
            benchmarks,
        )


def comparable_price_score(
    lowest_price: float | None,
    supplier_price: float | None,
    max_score: float,
) -> float | None:
    """Score only positive comparable prices; callers validate scope and currency."""
    if (
        lowest_price is None
        or supplier_price is None
        or lowest_price <= 0
        or supplier_price <= 0
        or max_score <= 0
    ):
        return None
    return round(float(lowest_price) / float(supplier_price) * float(max_score), 2)


def steel_quotes_comparable(
    suppliers: list[dict[str, Any]],
    *,
    required_quantity: float | None = None,
    buyer_currency: str | None = None,
    reviewer_confirmed: bool = False,
    all_suppliers_evaluated: bool = True,
) -> tuple[bool, list[str]]:
    """Check the minimum safe-comparison conditions for extracted steel quotations."""
    reasons = []
    quotes = [supplier.get("commercial_quote", {}) for supplier in suppliers]
    quantities = [
        supplier.get("technical_offer", {}).get("offered_quantity_mt")
        for supplier in suppliers
    ]
    currencies = {
        str(quote.get("currency")).strip().upper()
        if quote.get("currency") is not None
        else None
        for quote in quotes
    }
    delivery_bases = {
        str(quote.get("delivery_basis") or "").strip().casefold()
        for quote in quotes
    }
    totals = [quote.get("total_quoted_amount") for quote in quotes]

    if not reviewer_confirmed:
        reasons.append("reviewer has not confirmed the RFP-authorized comparable-price method and common cost scope")
    if not all_suppliers_evaluated:
        reasons.append("one or more supplier evaluations failed")
    if not suppliers:
        reasons.append("no supplier quotations are available")
    if (
        len(currencies) != 1
        or None in currencies
        or (buyer_currency and currencies != {buyer_currency.strip().upper()})
    ):
        reasons.append("quote currencies differ, are missing, or conflict with the RFP currency")
    if any(
        not isinstance(total, (int, float))
        or isinstance(total, bool)
        or not math.isfinite(float(total))
        or total <= 0
        for total in totals
    ):
        reasons.append("a positive total quoted amount is missing")
    if any(
        not isinstance(quantity, (int, float))
        or isinstance(quantity, bool)
        or not math.isfinite(float(quantity))
        or quantity <= 0
        for quantity in quantities
    ):
        reasons.append("an offered quantity is missing or invalid")
    elif len(set(quantities)) != 1:
        reasons.append("offered quantities differ")
    if required_quantity is not None and any(
        isinstance(quantity, (int, float)) and float(quantity) != float(required_quantity)
        for quantity in quantities
    ):
        reasons.append("offered quantities do not match the RFP quantity")
    if len(delivery_bases) != 1 or "" in delivery_bases:
        reasons.append("delivery bases differ or are missing")
    if any(quote.get("comparability_status") != "COMPARABLE" for quote in quotes):
        reasons.append("one or more supplier quotations are not confirmed comparable")
    return not reasons, reasons
