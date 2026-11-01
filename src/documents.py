"""PDF validation and page-aware text extraction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from pypdf import PdfReader


MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_PAGES = 100


@dataclass
class ExtractedDocument:
    filename: str
    sha256: str
    pages: list[dict[str, int | str]]
    warning: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(str(page["text"]) for page in self.pages)


def extract_pdf(filename: str, data: bytes) -> ExtractedDocument:
    if not filename.lower().endswith(".pdf"):
        raise ValueError(f"{filename}: only PDF files are supported.")
    if not data:
        raise ValueError(f"{filename}: the uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{filename}: exceeds the 15 MB file limit.")
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("the PDF is encrypted; upload an unlocked copy")
        if len(reader.pages) > MAX_PAGES:
            raise ValueError(f"the PDF exceeds the {MAX_PAGES}-page limit")
        pages = [
            {"page": number, "text": (page.extract_text() or "").strip()}
            for number, page in enumerate(reader.pages, start=1)
        ]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not read PDF ({type(exc).__name__})") from exc
    if not pages or not any(page["text"] for page in pages):
        raise ValueError(
            f"{filename}: no readable text was found. This may be a scanned PDF; "
            "OCR is not enabled, so please provide a text-searchable PDF."
        )
    return ExtractedDocument(
        filename=filename,
        sha256=sha256(data).hexdigest(),
        pages=pages,
        warning=None,
    )
