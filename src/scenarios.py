"""Non-official, deterministic ranking scenarios for saved evaluations."""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any

from src.scoring import apply_peer_metrics, rank_suppliers, weighted_score


def _normalized_weights(
    criteria: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    active = [item for item in criteria if item.get("active", True)]
    base = {str(item["id"]): float(item["weight"]) for item in active}
    requested = {**base, **(weights or {})}
    unknown = set(requested) - set(base)
    if unknown:
        raise ValueError(f"Unknown criterion weight keys: {sorted(unknown)}")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        for value in requested.values()
    ):
        raise ValueError("Criterion weights must be nonnegative numbers.")
    total = sum(float(value) for value in requested.values())
    if not active or total <= 0:
        raise ValueError("At least one active criterion must have positive weight.")
    normalized = {
        key: round(float(value) * 100.0 / total, 6)
        for key, value in requested.items()
    }
    last_key = next(reversed(normalized))
    normalized[last_key] = round(100.0 - sum(
        value for key, value in normalized.items() if key != last_key
    ), 6)
    scenario_criteria = [
        {**item, "weight": normalized[str(item["id"])]}
        for item in active
    ]
    return scenario_criteria, normalized


def evaluate_scenario(
    suppliers: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Recalculate rankings with proposed weights without mutating saved results."""
    scenario_criteria, normalized = _normalized_weights(criteria, weights)
    evaluated = deepcopy(suppliers)
    valid_suppliers = [
        supplier for supplier in evaluated if not supplier.get("evaluation_error")
    ]
    for supplier in valid_suppliers:
        supplier["absolute_score"] = weighted_score(
            scenario_criteria,
            {
                str(result["criterion_id"]): result.get("final_score")
                for result in supplier.get("criteria_results", [])
            },
        )
    apply_peer_metrics(scenario_criteria, valid_suppliers)
    ranked = rank_suppliers(valid_suppliers)
    baseline = {
        str(supplier["supplier_id"]): supplier.get("rank")
        for supplier in suppliers
    }
    for supplier in ranked:
        old_rank = baseline.get(str(supplier["supplier_id"]))
        supplier["baseline_rank"] = old_rank
        supplier["rank_movement"] = (
            int(old_rank) - int(supplier["rank"])
            if old_rank is not None else None
        )
    return {
        "official": False,
        "weights": normalized,
        "ranked_suppliers": ranked,
        "winner": ranked[0]["name"] if ranked else None,
    }


def analyze_sensitivity(
    suppliers: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
    *,
    perturbation: float = 10.0,
    simulations: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate winner stability under seeded random weight perturbations."""
    if perturbation < 0 or simulations < 1:
        raise ValueError("Perturbation must be nonnegative and simulations positive.")
    active = [item for item in criteria if item.get("active", True)]
    baseline_weights = {str(item["id"]): float(item["weight"]) for item in active}
    baseline = evaluate_scenario(suppliers, criteria, baseline_weights)
    rng = random.Random(seed)
    winner_counts: dict[str, int] = {}
    for _ in range(simulations):
        perturbed = {
            criterion_id: max(
                0.0, weight + rng.uniform(-perturbation, perturbation)
            )
            for criterion_id, weight in baseline_weights.items()
        }
        if sum(perturbed.values()) == 0:
            perturbed = dict(baseline_weights)
        winner = evaluate_scenario(suppliers, criteria, perturbed)["winner"]
        if winner is not None:
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
    stable_count = winner_counts.get(str(baseline["winner"]), 0)
    return {
        "official": False,
        "baseline_winner": baseline["winner"],
        "winner_counts": winner_counts,
        "winner_stability_pct": round(stable_count * 100 / simulations, 2),
        "simulations": simulations,
        "perturbation_points": perturbation,
        "seed": seed,
    }
