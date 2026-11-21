from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from xml.sax.saxutils import escape

def create_pdf(result, path):
    styles = getSampleStyleSheet()
    story = [Paragraph("SemantiCheck AI – Plagiarism Analysis Report", styles["Title"]), Spacer(1, 6*mm)]
    story += [
        Paragraph(f"Document A: {escape(result.get('doc_a',''))}", styles["BodyText"]),
        Paragraph(f"Document B: {escape(result.get('doc_b',''))}", styles["BodyText"]),
        Paragraph(f"Assessment: {escape(result.get('risk',''))}", styles["Heading2"]),
        Paragraph(f"Overall similarity: {result.get('overall_similarity',0)}%", styles["BodyText"]),
        Paragraph(f"Exact matches: {result.get('exact_count',0)}", styles["BodyText"]),
        Paragraph(f"Suspicious passages: {result.get('suspicious_count',0)}", styles["BodyText"]),
        Spacer(1, 5*mm), Paragraph("Matching passages", styles["Heading2"])
    ]
    for m in result.get("matches", []):
        story.append(Paragraph(f"<b>Sentence {m['sentence_number']} – {escape(m['category'])} ({m['score']}%)</b>", styles["BodyText"]))
        story.append(Paragraph("Document A: " + escape(m["sentence_a"]), styles["BodyText"]))
        story.append(Paragraph("Document B: " + escape(m["sentence_b"]), styles["BodyText"]))
        story.append(Spacer(1, 3*mm))
    story += [Spacer(1, 4*mm), Paragraph("Gemini explanation", styles["Heading2"]),
              Paragraph(escape(result.get("gemini_explanation","Not available")), styles["BodyText"]),
              Spacer(1, 4*mm), Paragraph("Disclaimer: Similarity scores are indicators for review, not conclusive proof of plagiarism.", styles["Italic"])]
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm)
    doc.build(story)
