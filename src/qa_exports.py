from __future__ import annotations

import csv
import io
import textwrap
from datetime import datetime
from typing import Any


def build_qa_csv(history: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=["Question", "Answer", "Sources"],
    )
    writer.writeheader()
    for exchange in history:
        sources = "; ".join(
            f"{source['filename']} (page {source['page']})"
            for source in exchange.get("sources", [])
        )
        writer.writerow(
            {
                "Question": exchange["question"],
                "Answer": exchange["answer"],
                "Sources": sources,
            }
        )
    return output.getvalue().encode("utf-8-sig")


def build_qa_pdf(
    history: list[dict[str, Any]],
    *,
    run_id: str,
    scope: str,
) -> bytes:
    exchanges = []
    for index, exchange in enumerate(history, start=1):
        lines = [f"Q{index}: {exchange['question']}", f"A{index}: {exchange['answer']}"]
        sources = exchange.get("sources", [])
        if sources:
            lines.append("Sources:")
            for source in sources:
                lines.append(
                    f"- {source['filename']} (page {source['page']})"
                )
        exchanges.append("\n".join(lines))
    return build_report_pdf(
        "Forge Steel RFP AI - Q&A Chat",
        [
            f"Evaluation: {run_id}",
            f"Chat scope: {scope}",
            f"Exported: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        ],
        [("Conversation", exchanges)],
    )


def build_report_pdf(
    title: str,
    metadata: list[str],
    sections: list[tuple[str, list[str]]],
) -> bytes:
    lines = [title, *metadata, ""]
    for heading, entries in sections:
        lines.extend([heading, ""])
        for entry in entries:
            lines.extend(
                textwrap.wrap(entry, width=88, subsequent_indent="  ")
                or [""]
            )
            lines.append("")
    pages = [lines[index : index + 48] for index in range(0, len(lines), 48)] or [[]]
    return _make_pdf(pages)


def _make_pdf(pages: list[list[str]]) -> bytes:
    page_count = len(pages)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids ["
        + b" ".join(f"{4 + 2 * index} 0 R".encode("ascii") for index in range(page_count))
        + f"] /Count {page_count} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]

    for index, page_lines in enumerate(pages):
        page_id = 4 + 2 * index
        content_id = page_id + 1
        commands = ["BT", "/F1 10 Tf", "50 750 Td", "14 TL"]
        for line in page_lines:
            safe_line = line.encode("cp1252", errors="replace")
            escaped = safe_line.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
            commands.append(b"(" + escaped + b") Tj")
            commands.append("T*")
        commands.append("ET")
        stream = b"\n".join(
            item if isinstance(item, bytes) else item.encode("ascii")
            for item in commands
        )
        objects.extend(
            [
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
                ).encode("ascii"),
                b"<< /Length "
                + str(len(stream)).encode("ascii")
                + b" >>\nstream\n"
                + stream
                + b"\nendstream",
            ]
        )

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)
