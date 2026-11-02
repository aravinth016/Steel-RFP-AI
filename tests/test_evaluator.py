import unittest
import json
from unittest.mock import patch

from src.evaluator import (
    EvaluationError,
    _build_prompt,
    _normalize_result,
    _parse_json_response,
    _retrieve_pages,
    _validate_result,
    estimate_output_tokens,
    estimate_prompt_tokens,
    evaluate_supplier,
)
from src.llm_providers import LLMGeneration


CRITERIA = [
    {
        "id": "technical",
        "name": "Technical fit",
        "description": "Check pump capacity",
        "rubric": "Score documented pump capacity",
        "weight": 100,
        "max_score": 5,
        "active": True,
    }
]


def document(document_id, filename, text):
    return {
        "id": document_id,
        "filename": filename,
        "pages": [{"page": 1, "text": text}],
    }


class EvaluatorTests(unittest.TestCase):
    def test_prompt_token_estimate_uses_extracted_prompt_content_without_calling_provider(self):
        with patch("src.evaluator.call_llm_with_usage") as call_llm:
            baseline = estimate_prompt_tokens(
                supplier={"name": "Supplier A"},
                criteria=CRITERIA,
                shared_documents=[document("rfp", "rfp.pdf", "Required capacity is 20.")],
                supplier_documents=[document("offer", "offer.pdf", "Capacity is 20.")],
                use_rag=False,
            )
            expanded = estimate_prompt_tokens(
                supplier={"name": "Supplier A"},
                criteria=CRITERIA,
                shared_documents=[
                    document("rfp", "rfp.pdf", "Required capacity is 20." * 100)
                ],
                supplier_documents=[document("offer", "offer.pdf", "Capacity is 20.")],
                use_rag=False,
            )
        self.assertGreater(expanded, baseline)
        call_llm.assert_not_called()

    def test_parses_json_object_after_model_commentary(self):
        response = (
            "The model may include commentary first.\n"
            '{"summary":"Checked.","gates":[],"criteria":[]}\n'
        )
        self.assertEqual(
            _parse_json_response(response),
            {"summary": "Checked.", "gates": [], "criteria": []},
        )

    def test_rejects_response_without_evaluation_json(self):
        with self.assertRaisesRegex(ValueError, "valid JSON evaluation object"):
            _parse_json_response("The supplier response is being evaluated.")

    def test_normalizes_missing_and_invalid_score_rows_with_warnings(self):
        result, warnings = _normalize_result(
            {
                "criteria": [
                    {
                        "criterion_id": "technical",
                        "score": "not-a-number",
                        "status": "Unknown",
                        "justification": "",
                    }
                ]
            },
            CRITERIA,
        )
        self.assertEqual(len(result["criteria"]), 1)
        self.assertEqual(result["criteria"][0]["score"], 0)
        self.assertEqual(result["criteria"][0]["status"], "NOT_VERIFIABLE")
        self.assertIsNone(result["criteria"][0]["evidence"])
        self.assertTrue(any("Malformed score" in warning for warning in warnings))
        self.assertTrue(any("summary" in warning for warning in warnings))

    def test_clips_out_of_range_scores_and_fills_missing_active_criteria(self):
        criteria = CRITERIA + [
            {
                "id": "delivery",
                "name": "Delivery",
                "description": "Timeline",
                "rubric": "Score timeline",
                "weight": 0,
                "max_score": 5,
                "active": True,
            }
        ]
        result, warnings = _normalize_result(
            {
                "summary": "Summary",
                "gates": [],
                "criteria": [
                    {
                        "criterion_id": "technical",
                        "score": 8,
                        "status": "Compliant",
                        "justification": "Evidence supports the score.",
                        "evidence": None,
                        "source_id": None,
                        "page": None,
                    }
                ],
            },
            criteria,
        )
        self.assertEqual([item["score"] for item in result["criteria"]], [5, 0])
        self.assertTrue(any("clipped to 5" in warning for warning in warnings))
        self.assertTrue(any("Missing result" in warning for warning in warnings))

    def test_retrieval_is_scoped_to_provided_supplier_documents(self):
        source_a = document("supplier-a", "a.pdf", "Pump capacity is 20 litres per second.")
        source_b = document("supplier-b", "b.pdf", "Pump capacity is 10 litres per second.")
        result = _retrieve_pages("pump capacity", [source_a])
        self.assertEqual([item["source_id"] for item in result], ["supplier-a"])
        self.assertNotIn("supplier-b", [item["source_id"] for item in result])

    def test_no_relevant_results_stays_empty(self):
        source = document("source", "proposal.pdf", "The shipment arrives next month.")
        self.assertEqual(_retrieve_pages("pump capacity", [source]), [])

    def test_rag_prompt_and_citation_allow_only_retrieved_pages(self):
        requirements = [document("req", "spec.pdf", "Pump capacity must be 20 litres per second.")]
        submissions = [document("offer", "offer.pdf", "Pump capacity is 20 litres per second.")]
        prompt, permitted = _build_prompt(
            {"name": "Supplier A"}, CRITERIA, requirements, submissions, True
        )
        self.assertEqual(permitted, {"req": {1}, "offer": {1}})
        self.assertIn("evidence is either a plain string quote or null", prompt)
        self.assertIn("source_id is the exact source_id", prompt)
        self.assertIn(
            "COMPARABLE, NOT_COMPARABLE, PENDING_REVIEW, or NOT_PROVIDED",
            prompt,
        )
        result = {
            "summary": "The proposal states a pump capacity.",
            "gates": [],
            "criteria": [
                {
                    "criterion_id": "technical",
                    "score": 5,
                    "status": "Compliant",
                    "justification": "The document states the required capacity.",
                    "evidence": "Pump capacity is 20 litres per second.",
                    "source_id": "offer",
                    "page": 1,
                }
            ],
        }
        _validate_result(result, CRITERIA, submissions, requirements, permitted)

    def test_normalizes_human_readable_commercial_comparability_status(self):
        result, warnings = _normalize_result(
            {
                "summary": "Summary.",
                "gates": [],
                "criteria": [],
                "commercial_quote": {"comparability_status": "Comparable"},
            },
            CRITERIA,
        )

        self.assertEqual(
            result["commercial_quote"]["comparability_status"], "COMPARABLE"
        )
        self.assertFalse(
            any("Malformed commercial comparability status" in item for item in warnings)
        )

    def test_citation_to_unretrieved_page_is_rejected(self):
        requirements = [document("req", "spec.pdf", "Pump capacity is specified.")]
        submissions = [document("offer", "offer.pdf", "Pump capacity is offered.")]
        result = {
            "summary": "Summary.",
            "gates": [],
            "criteria": [
                {
                    "criterion_id": "technical",
                    "score": 3,
                    "status": "Partially compliant",
                    "justification": "Some information is present.",
                    "evidence": "Pump capacity is offered.",
                    "source_id": "offer",
                    "page": 9,
                }
            ],
        }
        with self.assertRaises(EvaluationError):
            _validate_result(result, CRITERIA, submissions, requirements, {"offer": {1}, "req": {1}})

    def test_invented_evidence_quote_is_rejected(self):
        requirements = [document("req", "spec.pdf", "Pump capacity is specified.")]
        submissions = [document("offer", "offer.pdf", "The supplier submitted a proposal.")]
        result = {
            "summary": "Summary.",
            "gates": [],
            "criteria": [
                {
                    "criterion_id": "technical",
                    "score": 3,
                    "status": "Partially compliant",
                    "justification": "The evidence is incomplete.",
                    "evidence": "Capacity is 100 litres per second.",
                    "source_id": "offer",
                    "page": 1,
                }
            ],
        }
        with self.assertRaises(EvaluationError):
            _validate_result(result, CRITERIA, submissions, requirements, {"offer": {1}, "req": {1}})

    def test_evidence_matching_allows_pdf_line_breaks_and_punctuation_differences(self):
        requirements = [document("req", "spec.pdf", "Required: 20.5 mm diameter.")]
        submissions = [
            document(
                "offer",
                "offer.pdf",
                "The proposed rein-\nforcement has a 20.5 mm diameter.",
            )
        ]
        result = {
            "summary": "Summary.",
            "gates": [],
            "criteria": [
                {
                    "criterion_id": "technical",
                    "score": 5,
                    "status": "Compliant",
                    "justification": "The proposal states the diameter.",
                    "evidence": "reinforcement has a 20.5 mm diameter",
                    "source_id": "offer",
                    "page": 1,
                }
            ],
        }
        _validate_result(result, CRITERIA, submissions, requirements, {"offer": {1}, "req": {1}})

    def test_evidence_matching_does_not_accept_noncontiguous_or_paraphrased_text(self):
        requirements = [document("req", "spec.pdf", "Pump capacity is specified.")]
        submissions = [document("offer", "offer.pdf", "The pump capacity is 20 litres per second.")]
        result = {
            "summary": "Summary.",
            "gates": [],
            "criteria": [
                {
                    "criterion_id": "technical",
                    "score": 5,
                    "status": "Compliant",
                    "justification": "The proposal states capacity.",
                    "evidence": "Pump capacity of 20 litres per second",
                    "source_id": "offer",
                    "page": 1,
                }
            ],
        }
        with self.assertRaises(EvaluationError):
            _validate_result(result, CRITERIA, submissions, requirements, {"offer": {1}, "req": {1}})

    def test_evaluate_supplier_uses_selected_provider(self):
        supplier_doc = document("offer", "offer.pdf", "Pump capacity is 20 litres per second.")
        provider_result = {
            "summary": "The pump capacity is stated.",
            "gates": [],
            "criteria": [
                {
                    "criterion_id": "technical",
                    "score": 5,
                    "status": "Compliant",
                    "justification": "The proposal states the example capacity.",
                    "evidence": "Pump capacity is 20 litres per second.",
                    "source_id": "offer",
                    "page": 1,
                    "risks": [],
                    "clarifications": [],
                }
            ],
        }
        with patch(
            "src.evaluator.call_llm_with_usage",
            return_value=LLMGeneration(
                json.dumps(provider_result), input_tokens=84, output_tokens=26
            ),
        ) as call:
            result = evaluate_supplier(
                api_key="key",
                model="claude-model",
                supplier={"name": "Supplier A"},
                criteria=CRITERIA,
                shared_documents=[],
                supplier_documents=[supplier_doc],
                use_rag=True,
                provider="Claude",
            )
        self.assertEqual(result["provider"], "Claude")
        self.assertEqual(result["model"], "claude-model")
        self.assertEqual(call.call_args.args[0], "Claude")
        self.assertEqual(
            result["token_usage"], {"input_tokens": 84, "output_tokens": 26}
        )

    def test_output_token_estimate_scales_with_active_criteria(self):
        self.assertEqual(
            estimate_output_tokens(CRITERIA + [{"id": "inactive", "active": False}]),
            340,
        )

    def test_cohere_token_limit_retries_once_with_compact_output_instructions(self):
        supplier_doc = document(
            "offer", "offer.pdf", "Pump capacity is 20 litres per second."
        )
        compact_result = {
            "summary": "Capacity is stated.",
            "gates": [],
            "criteria": [
                {
                    "criterion_id": "technical",
                    "score": 4,
                    "status": "COMPLIANT",
                    "justification": "The offer states the capacity.",
                    "evidence": "Pump capacity is 20 litres per second.",
                    "source_id": "offer",
                    "page": 1,
                }
            ],
        }
        with patch(
            "src.evaluator.call_llm_with_usage",
            side_effect=[
                RuntimeError("Cohere reached the output token limit."),
                json.dumps(compact_result),
            ],
        ) as call:
            result = evaluate_supplier(
                api_key="key",
                model="command-a-plus-05-2026",
                supplier={"name": "Supplier A"},
                criteria=CRITERIA,
                shared_documents=[],
                supplier_documents=[supplier_doc],
                use_rag=True,
                provider="Cohere.ai",
            )

        self.assertEqual(call.call_count, 2)
        self.assertIn("COMPACT RETRY", call.call_args.args[3])
        self.assertEqual(result["criteria"][0]["score"], 4)
        self.assertTrue(
            any("retried once with compact output" in warning
                for warning in result["validation_warnings"])
        )

    def test_cohere_token_limit_after_compact_retry_reports_no_assessment(self):
        with patch(
            "src.evaluator.call_llm_with_usage",
            side_effect=RuntimeError("Cohere reached the output token limit."),
        ) as call:
            with self.assertRaisesRegex(
                EvaluationError,
                "automatic compact retry also reached the output token limit",
            ):
                evaluate_supplier(
                    api_key="key",
                    model="command-a-plus-05-2026",
                    supplier={"name": "Supplier A"},
                    criteria=CRITERIA,
                    shared_documents=[],
                    supplier_documents=[
                        document("offer", "offer.pdf", "Pump capacity is stated.")
                    ],
                    use_rag=True,
                    provider="Cohere.ai",
                )

        self.assertEqual(call.call_count, 2)

    def test_marks_unverifiable_evidence_unknown_without_failing_supplier(self):
        supplier_doc = document(
            "offer", "offer.pdf", "Pump capacity is 20 litres per second."
        )
        invalid_result = {
            "summary": "The pump capacity is stated.",
            "gates": [],
            "criteria": [
                {
                    "criterion_id": "technical",
                    "score": 5,
                    "status": "Compliant",
                    "justification": "The proposal states the example capacity.",
                    "evidence": "Pump capacity reaches 20 litres per second.",
                    "source_id": "offer",
                    "page": 1,
                }
            ],
        }
        with patch("src.evaluator.call_llm_with_usage", return_value=json.dumps(invalid_result)) as call:
            result = evaluate_supplier(
                api_key="key",
                model="claude-model",
                supplier={"name": "Supplier A"},
                criteria=CRITERIA,
                shared_documents=[],
                supplier_documents=[supplier_doc],
                use_rag=True,
                provider="Claude",
            )

        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["criteria"][0]["score"], 0)
        self.assertEqual(result["criteria"][0]["status"], "NOT_VERIFIABLE")
        self.assertIsNone(result["criteria"][0]["evidence"])
        self.assertTrue(any("NOT_VERIFIABLE" in item for item in result["validation_warnings"]))

    def test_evaluate_supplier_routes_openrouter_model_to_llm_call(self):
        supplier_doc = document("offer", "offer.pdf", "Pump capacity is 20 litres per second.")
        provider_result = {
            "summary": "The pump capacity is stated.",
            "gates": [],
            "criteria": [
                {
                    "criterion_id": "technical",
                    "score": 5,
                    "status": "Compliant",
                    "justification": "The proposal states the example capacity.",
                    "evidence": "Pump capacity is 20 litres per second.",
                    "source_id": "offer",
                    "page": 1,
                    "risks": [],
                    "clarifications": [],
                }
            ],
        }
        with patch("src.evaluator.call_llm_with_usage", return_value=json.dumps(provider_result)) as call:
            result = evaluate_supplier(
                api_key="openrouter-key",
                model="openai/gpt-4.1-mini",
                supplier={"name": "Supplier A"},
                criteria=CRITERIA,
                shared_documents=[],
                supplier_documents=[supplier_doc],
                use_rag=True,
                provider="OpenRouter",
            )
        self.assertEqual(result["provider"], "OpenRouter")
        self.assertEqual(result["model"], "openai/gpt-4.1-mini")
        self.assertEqual(call.call_args.args[:3], (
            "OpenRouter",
            "openrouter-key",
            "openai/gpt-4.1-mini",
        ))


if __name__ == "__main__":
    unittest.main()
