from io import BytesIO

from pypdf import PdfReader


SUPPORTED_TEXT_TYPES = {
    "text/plain",
    "text/markdown",
    "application/x-tex",
}


def extract_cv_text(filename: str, content_type: str | None, data: bytes) -> str:
    lower_name = filename.lower()

    if lower_name.endswith(".pdf") or content_type == "application/pdf":
        return extract_pdf_text(data)

    if lower_name.endswith((".txt", ".md")) or content_type in SUPPORTED_TEXT_TYPES:
        return data.decode("utf-8", errors="replace")

    raise ValueError("Unsupported CV file type. Please upload a PDF, TXT, or MD file.")


def extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(f"--- Page {index} ---\n{text}")

    extracted = "\n\n".join(pages).strip()
    if not extracted:
        raise ValueError("No readable text was found in this PDF.")
    return extracted
