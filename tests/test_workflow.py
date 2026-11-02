import unittest
from unittest.mock import patch

from pydantic import ValidationError

from src.evaluator import EvaluationError, _normalize_result
from src.normalization import validate_evaluation
from src.workflow import run_supplier_evaluation


CRITERIA = [
    {
        "id": "technical",
        "name": "Technical fit",
        "description": "Evaluate supported technical fit.",
        "rubric": "Use only cited evidence.",
        "weight": 100,
        "max_score": 5,
        "active": True,
    }
]

SUPPLIER = {"supplier_id": "supplier-a", "name": "Supplier A"}
DOCUMENT = {
    "id": "offer-a",
    "filename": "offer.pdf",
    "pages": [{"page": 1, "text": "The offered capacity is 20 tonnes per day."}],
}


def valid_evaluation():
    result, _ = _normalize_result(
        {
            "summary": "The proposal states a capacity.",
            "gates": [],
            "criteria": [
                {
                    "criterion_id": "technical",
                    "score": 4,
                    "status": "Compliant",
                    "justification": "The capacity is documented.",
                    "evidence": "offered capacity is 20 tonnes per day",
                    "source_id": "offer-a",
                    "page": 1,
                    "risks": [],
                    "clarifications": [],
                }
            ],
        },
        CRITERIA,
    )
    return result


class EvaluationWorkflowTests(unittest.TestCase):
    def test_strict_schema_rejects_string_scores(self):
        result = valid_evaluation()
        result["criteria"][0]["score"] = "4"
        with self.assertRaises(ValidationError):
            validate_evaluation(result, CRITERIA)

    def test_workflow_validates_evidence_scores_and_records_safe_trace(self):
        evaluated_result = {
            **valid_evaluation(),
            "validation_warnings": ["Provider response normalized."],
            "provider": "Mock",
            "model": "mock-model",
            "token_usage": {"input_tokens": 72, "output_tokens": 29},
        }
        with patch("src.workflow.evaluate_supplier", return_value=evaluated_result) as evaluate:
            record = run_supplier_evaluation(
                run_id="RUN-1",
                api_key="must-not-appear-in-trace",
                model="mock-model",
                provider="Mock",
                supplier=SUPPLIER,
                criteria=CRITERIA,
                shared_documents=[],
                supplier_documents=[DOCUMENT],
                use_rag=True,
                buyer_requirements={},
            )

        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["weighted_score"], 80.0)
        self.assertEqual(
            record["token_usage"], {"input_tokens": 72, "output_tokens": 29}
        )
        self.assertEqual(
            record["evaluation"]["validation_warnings"],
            ["Provider response normalized."],
        )
        self.assertEqual(record["evidence_checks"][0]["status"], "VERIFIED")
        self.assertEqual(
            [trace["node"] for trace in record["traces"]],
            ["prepare", "evaluate", "validate", "verify_evidence", "score", "persist"],
        )
        self.assertNotIn(
            "must-not-appear-in-trace",
            str(record["traces"]) + str(record["audit_events"]),
        )
        evaluate.assert_called_once()

    def test_failed_supplier_isolated_with_trace_and_error(self):
        with patch(
            "src.workflow.evaluate_supplier",
            side_effect=EvaluationError("provider failed"),
        ):
            record = run_supplier_evaluation(
                run_id="RUN-2",
                api_key="secret",
                model="mock-model",
                provider="Mock",
                supplier=SUPPLIER,
                criteria=CRITERIA,
                shared_documents=[],
                supplier_documents=[DOCUMENT],
                use_rag=False,
                buyer_requirements={},
            )

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "provider failed")
        self.assertEqual(
            [trace["node"] for trace in record["traces"]],
            ["prepare", "evaluate", "persist"],
        )
        self.assertEqual(record["traces"][1]["status"], "failed")

    def test_reuses_cached_evaluation_without_calling_provider(self):
        with patch("src.workflow.evaluate_supplier") as evaluate:
            record = run_supplier_evaluation(
                run_id="RUN-3",
                api_key="secret",
                model="mock-model",
                provider="Mock",
                supplier=SUPPLIER,
                criteria=CRITERIA,
                shared_documents=[],
                supplier_documents=[DOCUMENT],
                use_rag=True,
                buyer_requirements={},
                cached_evaluation=valid_evaluation(),
            )

        self.assertEqual(record["status"], "complete")
        self.assertEqual(
            [trace["node"] for trace in record["traces"]],
            ["prepare", "evaluate", "validate", "verify_evidence", "score", "persist"],
        )
        self.assertEqual(record["traces"][1]["status"], "skipped")
        evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
