import unittest

from src.scoring import (
    STARTER_CRITERIA,
    apply_peer_metrics,
    calculate_ppi,
    comparable_price_score,
    rank_suppliers,
    validate_criteria,
    weighted_score,
)


class ScoringTests(unittest.TestCase):
    def test_starter_weights_total_100(self):
        self.assertEqual(validate_criteria(STARTER_CRITERIA), [])
        self.assertEqual(
            [(item["name"], item["weight"]) for item in STARTER_CRITERIA],
            [
                ("Technical Capability", 30.0),
                ("Implementation Plan", 20.0),
                ("Commercial Value", 20.0),
                ("Security & Compliance", 20.0),
                ("Support & Experience", 10.0),
            ],
        )

    def test_weights_must_total_100(self):
        criteria = [dict(item) for item in STARTER_CRITERIA]
        criteria[0]["weight"] -= 1
        self.assertTrue(any("exactly 100%" in error for error in validate_criteria(criteria)))

    def test_weighted_score_hand_calculation(self):
        criteria = [
            {"id": "a", "weight": 60, "max_score": 5, "active": True},
            {"id": "b", "weight": 40, "max_score": 10, "active": True},
        ]
        self.assertEqual(weighted_score(criteria, {"a": 4, "b": 8}), 80.0)

    def test_missing_value_policy_is_explicit(self):
        criteria = [{"id": "a", "weight": 100, "max_score": 5, "active": True}]
        self.assertEqual(weighted_score(criteria, {"a": None}), 0.0)
        self.assertIsNone(weighted_score(criteria, {"a": None}, missing_as_zero=False))

    def test_out_of_range_score_rejected(self):
        criteria = [{"id": "a", "weight": 100, "max_score": 5, "active": True}]
        with self.assertRaises(ValueError):
            weighted_score(criteria, {"a": 6})

    def test_ppi_uses_weighted_peer_relative_scores(self):
        criteria = [{"id": "a", "weight": 100, "max_score": 5, "active": True}]
        self.assertEqual(calculate_ppi(criteria, {"a": 4}, {"a": 5}), 80.0)
        self.assertIsNone(calculate_ppi(criteria, {"a": None}, {"a": 5}))
        self.assertIsNone(calculate_ppi(criteria, {"a": 0}, {"a": 0}))

    def test_peer_metrics_calculate_benchmark_gap_relative_percent_and_ppi(self):
        criteria = [
            {"id": "a", "weight": 60, "max_score": 10, "active": True},
            {"id": "b", "weight": 40, "max_score": 5, "active": True},
        ]
        suppliers = [
            {
                "name": "A",
                "criteria_results": [
                    {"criterion_id": "a", "final_score": 8},
                    {"criterion_id": "b", "final_score": 0},
                ],
            },
            {
                "name": "B",
                "criteria_results": [
                    {"criterion_id": "a", "final_score": 10},
                    {"criterion_id": "b", "final_score": 0},
                ],
            },
        ]
        apply_peer_metrics(criteria, suppliers)
        first = suppliers[0]["criteria_results"]
        self.assertEqual(first[0]["benchmark"], 10)
        self.assertEqual(first[0]["criterion_gap"], -2)
        self.assertEqual(first[0]["relative_performance_pct"], 80)
        self.assertIsNone(first[1]["relative_performance_pct"])
        self.assertEqual(suppliers[0]["ppi"], 80)

    def test_ppi_ranking_does_not_apply_eligibility_precedence(self):
        result = rank_suppliers([
            {"name": "Lower PPI", "eligibility": "Eligible", "ppi": 1},
            {"name": "Higher PPI", "eligibility": "Ineligible", "ppi": 100},
        ])
        self.assertEqual(result[0]["name"], "Higher PPI")
        self.assertEqual(result[0]["rank"], 1)

    def test_ppi_precedes_absolute_score_then_uses_required_tie_breaks(self):
        result = rank_suppliers([
            {
                "name": "Lower PPI",
                "eligibility": "Eligible",
                "final_score": 100,
                "ppi": 90,
                "submission_date": "2026-01-01",
                "experience": 5,
            },
            {
                "name": "Later",
                "eligibility": "Eligible",
                "final_score": 1,
                "ppi": 100,
                "submission_date": "2026-02-01",
                "experience": 5,
            },
            {
                "name": "Earlier",
                "eligibility": "Eligible",
                "final_score": 1,
                "ppi": 100,
                "submission_date": "2026-01-01",
                "experience": 3,
            },
            {
                "name": "More experience",
                "eligibility": "Eligible",
                "final_score": 1,
                "ppi": 100,
                "submission_date": "2026-01-01",
                "experience": 5,
            },
        ])
        self.assertEqual(
            [supplier["name"] for supplier in result],
            ["More experience", "Earlier", "Later", "Lower PPI"],
        )

    def test_comparable_price_ratio_and_invalid_values(self):
        self.assertEqual(comparable_price_score(80_000, 100_000, 5), 4.0)
        self.assertIsNone(comparable_price_score(0, 100_000, 5))
        self.assertIsNone(comparable_price_score(80_000, -1, 5))
        self.assertIsNone(comparable_price_score(None, 100_000, 5))


if __name__ == "__main__":
    unittest.main()
