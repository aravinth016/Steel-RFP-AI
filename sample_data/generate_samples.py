from __future__ import annotations

import textwrap
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


OUTPUT_DIR = Path(__file__).resolve().parent
SAMPLE_ARCHIVE = OUTPUT_DIR / "synthetic_steel_rfp_pack.zip"

SAMPLE_DOCUMENTS = {
    "synthetic_rfp.pdf": [
        [
            "SYNTHETIC DEMO DATA - RFP: TMT Reinforcement Steel",
            "Issued by: Example Infrastructure Project (fictional)",
            "All organizations, prices, dates, grades, and requirements are fictional.",
            "",
            "1. Requirement",
            "Supply 500 MT of TMT reinforcement bars, grade Fe 500D, to the project site",
            "in Pune, Maharashtra. Required completion date: 30 November 2026.",
            "State the product standard and identify any deviations.",
            "",
            "2. Quality documentation",
            "Submit a Mill Test Certificate (MTC) for each supplied heat or batch, showing",
            "traceable heat references and the test results required by the applicable standard.",
            "Certificate presence does not establish independent authenticity.",
            "",
            "3. Delivery",
            "State quantity availability, first delivery date, completion date, delivery phases,",
            "and any conditions affecting delivery to the project site.",
            "",
            "SYNTHETIC DEMO DATA - page 1 of 2",
        ],
        [
            "SYNTHETIC DEMO DATA - RFP continuation",
            "4. Commercial quotation",
            "Provide currency, basic rate per MT, freight, taxes, loading/unloading, other charges,",
            "total quoted amount, delivery basis, quote validity, payment terms, and exclusions.",
            "Do not omit or assume any charge. Identify whether the total is delivered or ex-works.",
            "",
            "5. Supplier information",
            "Provide relevant steel-supply experience, project references, warranty/replacement",
            "support, and all required quotation documents.",
            "",
            "Mandatory conditions for this illustrative example: offer Fe 500D grade; offer at",
            "least 500 MT; submit an MTC. Human reviewers must verify every gate and all evidence.",
            "Lowest-total-price scoring is illustrative only and requires a human comparability check.",
            "",
            "SYNTHETIC DEMO DATA - page 2 of 2",
        ],
    ],
    "apex_steels_quote.pdf": [
        [
            "SYNTHETIC DEMO DATA - Apex Steels quotation",
            "All organizations, prices, dates, and claims are fictional.",
            "",
            "Technical offer: TMT reinforcement bars, grade Fe 500D, standard IS 1786.",
            "Offered quantity: 500 MT. No technical deviations are declared.",
            "Delivery location: Project site, Pune, Maharashtra.",
            "Completion date: 25 November 2026.",
            "",
            "Quality: Mill Test Certificate supplied per heat. Heat reference SYN-HEAT-101.",
            "This synthetic certificate is not independently verified.",
            "",
            "SYNTHETIC DEMO DATA - page 1 of 2",
        ],
        [
            "SYNTHETIC DEMO DATA - Apex Steels quotation continuation",
            "Commercial offer: INR basic rate 55,000 per MT; freight INR 1,000 per MT.",
            "Taxes: INR 5,040,000. Total quoted amount: INR 33,040,000.",
            "Delivery basis: Delivered to project site. Quote valid until 30 October 2026.",
            "Payment: 30 days from accepted delivery. No other charges or exclusions stated.",
            "",
            "",
            "SYNTHETIC DEMO DATA - page 2 of 2",
        ],
        [
            "SYNTHETIC DEMO DATA - Apex Steels evidence schedule - not a scorecard",
            "Evidence is fictional and supplied for evaluator review; no criterion scores are assigned.",
            "",
            "C01 Technical Capability: Fe 500D, IS 1786, 500 MT; no deviation declared.",
            "C04 Security & Compliance: illustrative MTC ref SYN-HEAT-101, 12 mm, 60 MT.",
            "Sample fields stated: yield 545 MPa, tensile 625 MPa, elongation 18.5%.",
            "These sample fields are not an authentic certificate or independently verified.",
            "C03 Commercial Value: INR 55,000/MT basic; INR 1,000/MT freight; GST INR 5,040,000.",
            "Total INR 33,040,000; delivered; quote valid to 30 Oct 2026; net 30 after acceptance.",
            "Loading/unloading included; no other charges or exclusions stated.",
            "C02 Implementation Plan: delivered to Pune; total 500 MT by 25 Nov 2026.",
            "Lot plan: 150 MT by 10 Nov; 200 MT by 17 Nov; 150 MT by 25 Nov 2026.",
            "C05 Support & Experience: fictional Metro Viaduct M-17, 320 MT (2024); Utility Corridor",
            "N-08, 210 MT (2025); East Bypass E-04, 180 MT (2025). Not independently verified.",
            "Support: notify within 7 days; joint inspection; confirmed nonconforming",
            "material proposed for replacement within 14 days, subject to inspection.",
            "Compliance documents: this quotation and evidence schedule; no signed-form claim is made.",
            "Final batch MTCs are proposed at dispatch; submitted sample is illustrative only.",
            "SYNTHETIC DEMO DATA - page 3 of 3",
        ],
    ],
    "bright_metals_quote.pdf": [
        [
            "SYNTHETIC DEMO DATA - Bright Metals quotation",
            "All organizations, prices, dates, and claims are fictional.",
            "",
            "Technical offer: TMT reinforcement bars, grade Fe 500D, standard IS 1786.",
            "Offered quantity: 500 MT. No technical deviations are declared.",
            "Delivery basis: Ex-works at supplier facility. Completion date: 20 December 2026.",
            "Delivery to project site is not included.",
            "",
            "Quality documents: No Mill Test Certificate is attached to this quotation.",
            "",
            "SYNTHETIC DEMO DATA - page 1 of 2",
        ],
        [
            "SYNTHETIC DEMO DATA - Bright Metals quotation continuation",
            "Commercial offer: INR basic rate 48,000 per MT.",
            "Total quoted amount: INR 28,575,000, ex-works.",
            "Freight and taxes are excluded and are not priced in this offer.",
            "Quote valid until 15 October 2026. Payment: 50 percent advance.",
            "",
            "",
            "SYNTHETIC DEMO DATA - page 2 of 2",
        ],
        [
            "SYNTHETIC DEMO DATA - Bright Metals evidence schedule - not a scorecard",
            "Evidence is fictional and supplied for evaluator review; no criterion scores are assigned.",
            "",
            "C01 Technical Capability: Fe 500D, IS 1786, 500 MT; no deviation declared.",
            "C04 Security & Compliance: no MTC or heat/batch traceability document is attached.",
            "C03 Commercial Value: INR 48,000/MT basic; quoted total INR 28,575,000 ex-works.",
            "Freight and taxes excluded and unpriced; no delivered total is stated.",
            "Quote valid through 15 Oct 2026; 50 percent advance; other charges not stated.",
            "C02 Implementation Plan: 500 MT completion stated as 20 Dec 2026 at supplier facility.",
            "Ex-works only; delivery to Project Cedar site and delivery phases are not offered.",
            "C05 Support & Experience: fictional West Connector package, 120 MT (2024).",
            "Reference and quantity are supplier claims and are not independently verified.",
            "Support: no warranty or replacement terms are provided.",
            "Compliance documents: quotation and this evidence schedule only.",
            "No MTC, signed-form status, delivery-to-site commitment, or full cost breakdown",
            "is evidenced in this submission. Missing items remain for buyer review.",
            "SYNTHETIC DEMO DATA - page 3 of 3",
        ],
    ],
    "nexa_steelworks_quote.pdf": [
        [
            "SYNTHETIC DEMO DATA - Nexa Steelworks quotation",
            "All organizations, prices, dates, and claims are fictional.",
            "",
            "Technical offer: TMT reinforcement bars, grade Fe 500D, standard IS 1786.",
            "Offered quantity: 500 MT. No technical deviations are declared.",
            "Delivery location: Project site, Pune, Maharashtra.",
            "Completion date: 15 November 2026.",
            "",
            "Quality: Mill Test Certificate supplied per heat. Heat reference SYN-HEAT-103.",
            "This synthetic certificate is not independently verified.",
            "",
            "SYNTHETIC DEMO DATA - page 1 of 2",
        ],
        [
            "SYNTHETIC DEMO DATA - Nexa Steelworks quotation continuation",
            "Commercial offer: INR basic rate 51,000 per MT; freight INR 800 per MT.",
            "Taxes: INR 4,662,000. Total quoted amount: INR 30,562,000.",
            "Delivery basis: Delivered to project site. Quote validity is not stated.",
            "Payment terms and other charges are not stated.",
            "",
            "",
            "SYNTHETIC DEMO DATA - page 2 of 2",
        ],
        [
            "SYNTHETIC DEMO DATA - Nexa Steelworks evidence schedule - not a scorecard",
            "Evidence is fictional and supplied for evaluator review; no criterion scores are assigned.",
            "",
            "C01 Technical Capability: Fe 500D, IS 1786, 500 MT; no deviation declared.",
            "C04 Security & Compliance: illustrative MTC ref SYN-HEAT-103, 16 mm, 80 MT.",
            "Sample fields stated: yield 535 MPa, tensile 615 MPa, elongation 18.0%.",
            "These sample fields are not an authentic certificate or independently verified.",
            "C03 Commercial Value: INR 51,000/MT basic; INR 800/MT freight; GST INR 4,662,000.",
            "Total INR 30,562,000, delivered; quote validity, payment terms, and other charges",
            "are not stated. Do not assume missing commercial terms.",
            "C02 Implementation Plan: delivered to Pune; total 500 MT by 15 Nov 2026.",
            "Lot plan: 100 MT by 05 Nov; 200 MT by 10 Nov; 200 MT by 15 Nov 2026.",
            "C05 Support & Experience: fictional River Crossing R-11, 360 MT (2025); East Ring Road",
            "E-03, 280 MT (2024). References and quantities are not independently verified.",
            "Support: supplier proposes joint inspection; response and replacement",
            "timelines are not stated. Supporting inspection report is not attached.",
            "Compliance documents: quotation and this evidence schedule; no signed-form claim is made.",
            "MTC reference is illustrative only; commercial validity/payment information is incomplete.",
            "SYNTHETIC DEMO DATA - page 3 of 3",
        ],
    ],
    "orbit_alloys_quote.pdf": [
        [
            "SYNTHETIC DEMO DATA - Orbit Alloys quotation",
            "All organizations, prices, dates, and claims are fictional.",
            "",
            "Technical offer: TMT reinforcement bars, grade Fe 500, standard IS 1786.",
            "Offered quantity: 500 MT. Deviation: offered grade is Fe 500, not Fe 500D.",
            "Delivery basis: Ex-works. Completion date: 28 November 2026.",
            "",
            "Quality: Mill Test Certificate supplied per heat. Heat reference SYN-HEAT-104.",
            "This synthetic certificate is not independently verified.",
            "",
            "SYNTHETIC DEMO DATA - page 1 of 2",
        ],
        [
            "SYNTHETIC DEMO DATA - Orbit Alloys quotation continuation",
            "Commercial offer: INR basic rate 50,500 per MT; freight INR 2,000 per MT.",
            "Total quoted amount: INR 27,250,000, ex-works. Taxes are excluded.",
            "Quote validity and payment terms are not stated.",
            "",
            "",
            "SYNTHETIC DEMO DATA - page 2 of 2",
        ],
        [
            "SYNTHETIC DEMO DATA - Orbit Alloys evidence schedule - not a scorecard",
            "Evidence is fictional and supplied for evaluator review; no criterion scores are assigned.",
            "",
            "C01 Technical Capability: offered grade Fe 500, not the RFP's Fe 500D; 500 MT offered.",
            "This is a material grade deviation; do not treat it as compliant.",
            "C04 Security & Compliance: illustrative MTC ref SYN-HEAT-104; supplier states MTC per heat.",
            "No test-value schedule or authentic certificate is included; verification is required.",
            "C03 Commercial Value: INR 50,500/MT basic; INR 2,000/MT freight; total INR 27,250,000",
            "ex-works; taxes excluded. Quote validity, payment terms, and other charges unstated.",
            "C02 Implementation Plan: 500 MT completion stated as 28 Nov 2026, ex-works.",
            "Delivery location is supplier facility; no site-delivery commitment or lot phases stated.",
            "C05 Support & Experience: fictional five-reference claim; two examples are Metro Link",
            "Package M-22, 400 MT (2024), and South Yard S-06, 250 MT (2025). Not verified.",
            "Support: written replacement process subject to inspection; no response",
            "or replacement deadline is stated.",
            "Compliance documents: quotation and this evidence schedule; quote validity and payment",
            "terms are not stated. No signed-form status is evidenced.",
            "SYNTHETIC DEMO DATA - page 3 of 3",
        ],
    ],
}


def _pdf_string(text: str) -> bytes:
    safe = text.encode("ascii", errors="replace").decode("ascii")
    escaped = safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"({escaped})".encode("ascii")


def _page_stream(lines: list[str]) -> bytes:
    wrapped = [
        segment
        for line in lines
        for segment in (textwrap.wrap(line, width=88) or [""])
    ]
    commands = [b"BT", b"/F1 10 Tf", b"50 748 Td", b"14 TL"]
    for line in wrapped:
        commands.extend([_pdf_string(line) + b" Tj", b"T*"])
    commands.append(b"ET")
    return b"\n".join(commands)


def create_pdf(path: Path, pages: list[list[str]]) -> None:
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_references = []
    for page_number, lines in enumerate(pages):
        page_id = 4 + page_number * 2
        content_id = page_id + 1
        page_references.append(f"{page_id} 0 R")
        stream = _page_stream(lines)
        objects.extend([
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii"),
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream",
        ])
    objects[1] = (
        f"<< /Type /Pages /Kids [{' '.join(page_references)}] /Count {len(pages)} >>"
    ).encode("ascii")

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(pdf)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, pages in SAMPLE_DOCUMENTS.items():
        create_pdf(OUTPUT_DIR / name, pages)
    with ZipFile(SAMPLE_ARCHIVE, "w", compression=ZIP_DEFLATED) as archive:
        for name in SAMPLE_DOCUMENTS:
            archive.write(OUTPUT_DIR / name, arcname=name)
    for old_name in (
        "apex_systems_proposal.pdf",
        "brightpath_tech_proposal.pdf",
        "nexaworks_proposal.pdf",
        "orbit_digital_proposal.pdf",
    ):
        (OUTPUT_DIR / old_name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
