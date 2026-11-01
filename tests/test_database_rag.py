import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.database import (
    connect,
    get_app_setting,
    initialize,
    load_criteria,
    load_rag_corpus,
    load_run,
    save_app_setting,
    save_criteria,
    save_rag_corpus,
    save_run,
)
from src.scoring import STARTER_CRITERIA


class RagCorpusPersistenceTests(unittest.TestCase):
    def test_migrates_exact_builtin_steel_criteria_without_touching_run_history(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "evaluations.sqlite3"
            with patch("src.database.DB_PATH", database_path):
                initialize()
                with connect() as db:
                    db.execute("DELETE FROM evaluation_criteria")
                    old_template = [
                        (
                            "C01", "Technical and grade compliance",
                            "Offered steel grade, requested grade, dimensions or product form, stated specification, deviations, and compliance evidence.",
                            30, "9-10: stated RFP requirements fully addressed with clear evidence; 7-8: meets requirements with minor documented gaps; 4-6: partially meets or needs clarification; 1-3: material deviation shown; 0: no usable evidence. Missing facts must be marked NOT_PROVIDED, not assumed non-compliant.",
                        ),
                        (
                            "C02", "Quality and test documentation",
                            "Mill Test Certificate availability, heat/batch traceability, test results, quality certificates, inspection details, and document validity.",
                            15, "9-10: applicable quality/test documents are complete, traceable and clearly cited; 7-8: strong evidence with minor gaps; 4-6: partial evidence or verification needed; 1-3: major gaps shown; 0: no usable evidence. Document presence does not prove authenticity.",
                        ),
                        (
                            "C03", "Commercial value",
                            "Comparable price per MT, total quoted amount, freight, taxes, unloading, other charges, exclusions, and payment terms.",
                            20, "9-10: transparent, complete and comparable evaluated cost under the RFP method; 7-8: clear quote with minor gaps; 4-6: partially itemized or comparability needs review; 1-3: material commercial uncertainty; 0: no usable quote evidence. Do not infer missing charges.",
                        ),
                        (
                            "C04", "Quantity and delivery capability",
                            "Quantity offered against required quantity, availability, delivery dates, delivery phases, and delivery to the required site.",
                            15, "9-10: quantity and delivery commitments clearly meet the RFP with traceable evidence; 7-8: credible commitments with minor gaps; 4-6: partial availability or date clarification needed; 1-3: material shortfall or uncertainty shown; 0: no usable evidence. Apply mandatory consequences only when explicitly configured from the RFP.",
                        ),
                        (
                            "C05", "Supplier experience and references",
                            "Relevant steel supply experience, similar project references, capacity claims, and supporting evidence.",
                            10, "9-10: multiple directly relevant, well-supported references and capacity evidence; 7-8: credible relevant experience with minor gaps; 4-6: limited or partly verifiable experience; 1-3: weak evidence; 0: no usable evidence.",
                        ),
                        (
                            "C06", "Warranty and replacement support",
                            "Warranty terms, rejection/replacement process, response commitments, and support obligations.",
                            5, "9-10: clear, comprehensive commitments aligned with RFP; 7-8: adequate terms with minor gaps; 4-6: partial or ambiguous terms; 1-3: significant limitations; 0: no usable evidence. Do not infer terms.",
                        ),
                        (
                            "C07", "Documentation and quotation completeness",
                            "Required forms, signed quotation, validity period, commercial breakdown, declarations, and traceable supporting documents.",
                            5, "9-10: requested documents and quote details complete and traceable; 7-8: minor omissions; 4-6: material gaps needing clarification; 1-3: major incompleteness; 0: no usable documents. Missing documents are NOT_PROVIDED, not proof of test failure.",
                        ),
                    ]
                    db.executemany(
                        """
                        INSERT INTO evaluation_criteria(
                            criterion_id, name, description, weight, max_score, rubric,
                            is_active, display_order
                        ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                        """,
                        [
                            (cid, name, description, weight, 10, rubric, index)
                            for index, (cid, name, description, weight, rubric) in enumerate(old_template)
                        ],
                    )
                    db.execute(
                        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                        ("OLD-RUN", "2026-01-01", "RFP", "Complete", 0, "{}"),
                    )

                initialize()
                self.assertEqual(
                    [(item["id"], item["name"], item["weight"]) for item in load_criteria()],
                    [(item["id"], item["name"], item["weight"]) for item in STARTER_CRITERIA],
                )
                with connect() as db:
                    self.assertIsNotNone(
                        db.execute("SELECT run_id FROM runs WHERE run_id = 'OLD-RUN'").fetchone()
                    )

    def test_initialize_preserves_customized_criteria(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "evaluations.sqlite3"
            with patch("src.database.DB_PATH", database_path):
                initialize()
                criteria = load_criteria()
                criteria[0]["description"] = "Organization-approved customized description."
                save_criteria(criteria)

                initialize()

                self.assertEqual(load_criteria(), criteria)

    def test_criteria_are_seeded_and_persisted_in_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "evaluations.sqlite3"
            with patch("src.database.DB_PATH", database_path):
                initialize()
                criteria = load_criteria()
                self.assertTrue(criteria)
                self.assertEqual(sum(item["weight"] for item in criteria if item["active"]), 100)

                criteria[0]["weight"] -= 1
                criteria[1]["weight"] += 1
                save_criteria(criteria)
                self.assertEqual(load_criteria(), criteria)

    def test_persists_provider_and_model_preferences(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "evaluations.sqlite3"
            with patch("src.database.DB_PATH", database_path):
                initialize()
                self.assertIsNone(get_app_setting("llm_provider"))
                save_app_setting("llm_provider", "OpenRouter")
                save_app_setting("model_OpenRouter", "openai/gpt-4.1-mini")
                self.assertEqual(get_app_setting("llm_provider"), "OpenRouter")
                self.assertEqual(
                    get_app_setting("model_OpenRouter"),
                    "openai/gpt-4.1-mini",
                )

    def test_persists_extracted_pages_separately_from_run_export(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "evaluations.sqlite3"
            with patch("src.database.DB_PATH", database_path):
                initialize()
                run = {
                    "run_id": "RUN-TEST",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "mode": "Proposal / Quotation Evaluation",
                    "status": "Evaluating",
                    "is_demo": False,
                }
                save_run(run)
                save_rag_corpus(
                    "RUN-TEST",
                    [
                        {
                            "id": "source-a",
                            "filename": "proposal.pdf",
                            "scope": "supplier",
                            "supplier_id": "supplier-a",
                            "pages": [
                                {"page": 2, "text": "Pump capacity is 20 litres per second."}
                            ],
                        }
                    ],
                )

                loaded = load_rag_corpus("RUN-TEST")
                self.assertEqual(loaded[0]["supplier_id"], "supplier-a")
                self.assertEqual(loaded[0]["pages"][0]["page"], 2)
                self.assertEqual(
                    loaded[0]["pages"][0]["text"],
                    "Pump capacity is 20 litres per second.",
                )
                self.assertNotIn("pages", load_run("RUN-TEST"))

    def test_persists_run_and_supplier_result_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "evaluations.sqlite3"
            with patch("src.database.DB_PATH", database_path):
                initialize()
                save_run({
                    "run_id": "RFP-TEST",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "mode": "RFP Proposal Evaluation",
                    "status": "Evaluating",
                    "is_demo": False,
                    "suppliers": [{
                        "supplier_id": "supplier-a",
                        "name": "Supplier A",
                        "submission_date": "2026-01-01",
                        "experience": 4.5,
                        "absolute_score": 87.5,
                        "ppi": 92.0,
                        "rank": 1,
                    }],
                })
                with connect() as db:
                    run_row = db.execute(
                        "SELECT status FROM rfp_runs WHERE rfp_run_id = ?",
                        ("RFP-TEST",),
                    ).fetchone()
                    supplier_row = db.execute(
                        "SELECT supplier_name, absolute_score, ppi, final_rank "
                        "FROM supplier_results WHERE rfp_run_id = ?",
                        ("RFP-TEST",),
                    ).fetchone()
                self.assertEqual(run_row["status"], "Evaluating")
                self.assertEqual(
                    tuple(supplier_row),
                    ("Supplier A", 87.5, 92.0, 1),
                )


if __name__ == "__main__":
    unittest.main()
