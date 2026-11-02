"""Validate and serialize normalized evaluation data."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.schemas import EvaluationOutput


def validate_evaluation(
    result: dict[str, Any],
    criteria: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply strict shape, criterion-set, and criterion-specific score checks."""
    validated = EvaluationOutput.model_validate(result, strict=True)
    active = {
        str(item["id"]): float(item["max_score"])
        for item in criteria
        if item.get("active", True)
    }
    evaluated_ids = [item.criterion_id for item in validated.criteria]
    if len(evaluated_ids) != len(set(evaluated_ids)):
        raise ValueError("The normalized response contains duplicate criterion results.")
    if set(evaluated_ids) != set(active):
        raise ValueError("The normalized response does not match the active criterion set.")
    for item in validated.criteria:
        if item.score > active[item.criterion_id]:
            raise ValueError(
                f"Score for criterion {item.criterion_id!r} exceeds its configured maximum."
            )
    return validated.model_dump(mode="python")


__all__ = ["ValidationError", "validate_evaluation"]
