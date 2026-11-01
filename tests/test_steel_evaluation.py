import unittest

from src.evaluator import EvaluationError, _build_prompt, _normalize_result, _validate_result
from src.scoring import steel_quotes_comparable


CRITERIA = [
    {
        "id": "C01",
        "name": "Technical and grade compliance",
        "description": "Check offered grade",
        "rubric": "Use RFP evidence",
        "weight": 100,
        "max_score": 10,
        "active": True,
    }
]


def document(document_id, filename, text):
    return {
        "id": document_id,
        "filename": filename,
        "pages": [{"page": 1, "text": text}],
    }


def quote_supplier(total, *, currency="INR", quantity=500, basis="Delivered"):
    return {
        "technical_offer": {"offered_quantity_mt": quantity},
        "commercial_quote": {
            "currency": currency,
            "total_quoted_amount": total,
            "delivery_basis": basis,
            "comparability_status": "COMPARABLE",
        },
    }


class SteelEvaluationTests(unittest.TestCase):
    def test_only_configured_rfp_gates_are_returned_and_unknowns_stay_null(self):
        result, warnings = _normalize_result(
            {
                "summary": "Steel offer extracted.",
                "gates": [{"id": "INVENTED", "status": "FAIL", "reason": "Model invented this."}],
                "criteria": [],
                "technical_offer": {"offered_grade": 7, "offered_quantity_mt": "unknown"},
                "quality_documents": [{"document_type": "Mill Test Certificate", "provided": True}],
            },
            CRITERIA,
            [{"id": "G01", "requirement": "Offer Fe 500D"}],
        )
        self.assertEqual([gate["id"] for gate in result["gates"]], ["G01"])
        self.assertEqual(result["gates"][0]["status"], "Pending")
        self.assertIsNone(result["technical_offer"]["offered_grade"])
        self.assertIsNone(result["technical_offer"]["offered_quantity_mt"])
        self.assertEqual(
            result["quality_documents"][0]["verification_status"],
            "NOT_INDEPENDENTLY_VERIFIED",
        )
        self.assertTrue(any("invented mandatory gates" in warning for warning in warnings))

    def test_steel_prompt_contains_buyer_requirements_and_gate_scope(self):
        prompt, _ = _build_prompt(
            {"name": "Steel Supplier"},
            CRITERIA,
            [document("rfp", "rfp.pdf", "Required grade Fe 500D, 500 MT.")],
            [document("offer", "offer.pdf", "Offer grade Fe 500D, 500 MT.")],
            True,
            [{"id": "G01", "requirement": "Offer Fe 500D"}],
            {"required_grade": "Fe 500D", "required_quantity_mt": 500},
        )
        self.assertIn('"required_grade":"Fe 500D"', prompt)
        self.assertIn("G01", prompt)
        self.assertIn("do not invent gates", prompt)

    def test_rag_prompt_includes_each_evidence_page_once_for_lnt_review(self):
        requirement_text = "UNIQUE-RFP-EVIDENCE Fe 500D is required."
        offer_text = "UNIQUE-OFFER-EVIDENCE Fe 500D is offered."
        prompt, permitted = _build_prompt(
            {"name": "Steel Supplier"},
            CRITERIA,
            [document("rfp", "rfp.pdf", requirement_text)],
            [document("offer", "offer.pdf", offer_text)],
            True,
            [{"id": "G01", "requirement": "Offer Fe 500D"}],
            {"required_grade": "Fe 500D"},
        )

        self.assertIn("L&T procurement review", prompt)
        self.assertIn("criterion_sources", prompt)
        self.assertIn("evidence_pages", prompt)
        self.assertEqual(prompt.count(requirement_text), 1)
        self.assertEqual(prompt.count(offer_text), 1)
        self.assertEqual(permitted, {"rfp": {1}, "offer": {1}})

    def test_nested_steel_evidence_must_match_supplier_source_page(self):
        offer = document("offer", "offer.pdf", "Offered grade Fe 500D.")
        result = {
            "summary": "Summary",
            "gates": [],
            "criteria": [],
            "technical_offer": {
                "evidence": [{"source_id": "offer", "page": 1, "quote": "Offered grade Fe 500D."}],
            },
            "commercial_quote": {"evidence": []},
            "delivery": {"evidence": []},
            "quality_documents": [],
        }
        _validate_result(result, [], [offer], [], {"offer": {1}})
        result["technical_offer"]["evidence"][0]["quote"] = "Offered grade Fe 600."
        with self.assertRaisesRegex(EvaluationError, "steel offer evidence quote"):
            _validate_result(result, [], [offer], [], {"offer": {1}})

    def test_steel_quote_comparison_requires_matching_scope_and_human_confirmation(self):
        suppliers = [quote_supplier(1000), quote_supplier(1200)]
        valid, reasons = steel_quotes_comparable(
            suppliers,
            required_quantity=500,
            buyer_currency="INR",
            reviewer_confirmed=True,
        )
        self.assertTrue(valid)
        self.assertEqual(reasons, [])

        suppliers[1]["commercial_quote"]["delivery_basis"] = "Ex-works"
        valid, reasons = steel_quotes_comparable(
            suppliers,
            required_quantity=500,
            buyer_currency="INR",
            reviewer_confirmed=True,
        )
        self.assertFalse(valid)
        self.assertTrue(any("delivery bases differ" in reason for reason in reasons))

    def test_steel_quote_comparison_rejects_missing_total_and_wrong_rfp_quantity(self):
        suppliers = [quote_supplier(None), quote_supplier(1200, quantity=400)]
        valid, reasons = steel_quotes_comparable(
            suppliers,
            required_quantity=500,
            buyer_currency="INR",
            reviewer_confirmed=True,
        )
        self.assertFalse(valid)
        self.assertTrue(any("positive total" in reason for reason in reasons))
        self.assertTrue(any("do not match the RFP quantity" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
