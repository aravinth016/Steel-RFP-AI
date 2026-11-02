"""Independent evidence-citation checks for normalized evaluations."""

from __future__ import annotations

from typing import Any

from src.evaluator import _evidence_quote_matches


def verify_evidence(
    result: dict[str, Any],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pages = {
        (str(document["id"]), int(page["page"])): str(page.get("text", ""))
        for document in documents
        for page in document.get("pages", [])
    }
    checks: list[dict[str, Any]] = []

    def check(
        kind: str,
        target: str,
        source_id: Any,
        page: Any,
        quote: Any,
    ) -> None:
        status = "NOT_PROVIDED"
        if isinstance(source_id, str) and isinstance(page, int) and not isinstance(page, bool):
            page_text = pages.get((source_id, page))
            status = (
                "VERIFIED"
                if page_text is not None
                and isinstance(quote, str)
                and _evidence_quote_matches(quote, page_text)
                else "NOT_VERIFIED"
            )
        checks.append({
            "evidence_type": kind,
            "target_id": target,
            "source_id": source_id if isinstance(source_id, str) else None,
            "page": page if isinstance(page, int) and not isinstance(page, bool) else None,
            "status": status,
        })

    for item in result.get("criteria", []):
        check(
            "criterion",
            item["criterion_id"],
            item.get("source_id"),
            item.get("page"),
            item.get("evidence"),
        )
    for item in result.get("gates", []):
        check("gate", item["id"], item.get("source_id"), item.get("page"), item.get("evidence"))
    for section_name in ("technical_offer", "commercial_quote", "delivery"):
        for index, citation in enumerate(result[section_name]["evidence"], start=1):
            check(
                section_name,
                str(index),
                citation["source_id"],
                citation["page"],
                citation["quote"],
            )
    for index, item in enumerate(result["quality_documents"], start=1):
        check(
            "quality_document",
            str(index),
            item.get("source_id"),
            item.get("page"),
            item.get("evidence_quote"),
        )
    return checks


__all__ = ["verify_evidence"]
