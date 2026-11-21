import os
def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        import fitz
        doc = fitz.open(path)
        return "\n".join(page.get_text("text") for page in doc)
    if ext == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Unsupported document type.")
