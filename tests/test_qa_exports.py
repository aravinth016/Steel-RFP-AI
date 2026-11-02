import csv
import io
import unittest

from pypdf import PdfReader

from src.qa_exports import build_qa_csv, build_qa_pdf


class QaExportTests(unittest.TestCase):
    def setUp(self):
        self.history = [
            {
                "question": 'What is the "required" grade?',
                "answer": "The grade is Fe 500 [S1].",
                "sources": [{"filename": "spec.pdf", "page": 2}],
            }
        ]

    def test_csv_export_includes_questions_answers_and_sources(self):
        exported = build_qa_csv(self.history).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(exported)))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Question"], self.history[0]["question"])
        self.assertEqual(rows[0]["Answer"], self.history[0]["answer"])
        self.assertEqual(rows[0]["Sources"], "spec.pdf (page 2)")

    def test_pdf_export_is_readable_and_includes_context_and_citations(self):
        exported = build_qa_pdf(
            self.history,
            run_id="RFP-123",
            scope="Project requirements only",
        )
        reader = PdfReader(io.BytesIO(exported))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

        self.assertEqual(len(reader.pages), 1)
        self.assertIn("RFP-123", text)
        self.assertIn("Project requirements only", text)
        self.assertIn('What is the "required" grade?', text)
        self.assertIn("Fe 500 [S1].", text)
        self.assertIn("spec.pdf", text)
        self.assertIn("(page 2)", text)
