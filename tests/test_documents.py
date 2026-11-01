import io
import unittest
from pathlib import Path

from pypdf import PdfWriter

from src.documents import extract_pdf


def blank_pdf(encrypted=False):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    if encrypted:
        writer.encrypt("test-password")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class DocumentExtractionTests(unittest.TestCase):
    def test_synthetic_sample_pack_contains_readable_template_based_rfp_and_proposals(self):
        sample_dir = Path(__file__).resolve().parents[1] / "sample_data"
        files = sorted(sample_dir.glob("*.pdf"))
        self.assertEqual(len(files), 5)
        expected_page_counts = {
            "synthetic_rfp.pdf": 2,
            "apex_steels_quote.pdf": 3,
            "bright_metals_quote.pdf": 3,
            "nexa_steelworks_quote.pdf": 3,
            "orbit_alloys_quote.pdf": 3,
        }
        for path in files:
            extracted = extract_pdf(path.name, path.read_bytes())
            self.assertEqual(len(extracted.pages), expected_page_counts[path.name])
            self.assertIn(
                "SYNTHETIC DEMO DATA",
                "\n".join(page["text"] for page in extracted.pages),
            )

        template_supplier_content = {
            "apex_steels_quote.pdf": (
                "Apex Steels",
                "33,040,000",
                "30 October 2026",
                "Metro Viaduct M-17",
                "SYN-HEAT-101",
            ),
            "bright_metals_quote.pdf": (
                "Bright Metals",
                "28,575,000",
                "No Mill Test Certificate",
                "20 December 2026",
                "Freight and taxes are excluded",
                "no warranty or replacement terms",
            ),
            "nexa_steelworks_quote.pdf": (
                "Nexa Steelworks",
                "30,562,000",
                "15 Nov 2026",
                "River Crossing R-11",
                "Payment terms and other charges are not stated.",
            ),
            "orbit_alloys_quote.pdf": (
                "Orbit Alloys",
                "27,250,000",
                "Fe 500, not Fe 500D",
                "28 November 2026",
                "Metro Link",
                "written replacement process",
            ),
        }
        for filename, expected_content in template_supplier_content.items():
            path = sample_dir / filename
            extracted = extract_pdf(path.name, path.read_bytes())
            text = "\n".join(page["text"] for page in extracted.pages)
            self.assertEqual(len(extracted.pages), 3)
            self.assertIn("not a scorecard", text)
            for criterion_id in ("C01", "C02", "C03", "C04", "C05"):
                self.assertIn(criterion_id, text)
            for expected in expected_content:
                self.assertIn(expected, text)

    def test_rejects_non_pdf_extension(self):
        with self.assertRaisesRegex(ValueError, "only PDF"):
            extract_pdf("proposal.txt", b"not a pdf")

    def test_rejects_corrupt_pdf(self):
        with self.assertRaises(ValueError):
            extract_pdf("corrupt.pdf", b"not a pdf")

    def test_rejects_empty_or_scanned_text(self):
        with self.assertRaisesRegex(ValueError, "no readable text"):
            extract_pdf("blank.pdf", blank_pdf())

    def test_rejects_encrypted_pdf(self):
        with self.assertRaisesRegex(ValueError, "encrypted"):
            extract_pdf("encrypted.pdf", blank_pdf(encrypted=True))

    def test_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            extract_pdf("empty.pdf", b"")


if __name__ == "__main__":
    unittest.main()
