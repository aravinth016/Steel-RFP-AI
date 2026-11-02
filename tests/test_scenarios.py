import copy
import unittest

from src.scenarios import analyze_sensitivity, evaluate_scenario


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.criteria = [
            {"id": "technical", "name": "Technical", "weight": 50, "max_score": 10, "active": True},
            {"id": "commercial", "name": "Commercial", "weight": 50, "max_score": 10, "active": True},
        ]
        self.suppliers = [
            {
                "supplier_id": "a",
                "name": "Supplier A",
                "rank": 1,
                "criteria_results": [
                    {"criterion_id": "technical", "final_score": 10},
                    {"criterion_id": "commercial", "final_score": 2},
                ],
            },
            {
                "supplier_id": "b",
                "name": "Supplier B",
                "rank": 2,
                "criteria_results": [
                    {"criterion_id": "technical", "final_score": 2},
                    {"criterion_id": "commercial", "final_score": 10},
                ],
            },
        ]

    def test_scenario_recalculates_scores_and_rank_without_mutating_saved_data(self):
        original = copy.deepcopy(self.suppliers)

        scenario = evaluate_scenario(
            self.suppliers,
            self.criteria,
            {"technical": 90, "commercial": 10},
        )

        self.assertEqual(scenario["winner"], "Supplier A")
        self.assertEqual(scenario["ranked_suppliers"][0]["absolute_score"], 92.0)
        self.assertEqual(scenario["ranked_suppliers"][0]["baseline_rank"], 1)
        self.assertEqual(self.suppliers, original)
        self.assertEqual(scenario["weights"], {"technical": 90.0, "commercial": 10.0})
        self.assertFalse(scenario["official"])

    def test_sensitivity_analysis_is_deterministic(self):
        first = analyze_sensitivity(self.suppliers, self.criteria, simulations=25, seed=17)
        second = analyze_sensitivity(self.suppliers, self.criteria, simulations=25, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(first["simulations"], 25)
        self.assertFalse(first["official"])

    def test_rejects_unknown_or_invalid_weights(self):
        with self.assertRaisesRegex(ValueError, "Unknown criterion"):
            evaluate_scenario(self.suppliers, self.criteria, {"unknown": 10})
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            evaluate_scenario(self.suppliers, self.criteria, {"technical": float("nan")})


if __name__ == "__main__":
    unittest.main()
