"""PDF investigation report (ReportLab)."""
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

INK, RED, AMBER = colors.HexColor("#17202A"), colors.HexColor("#B3261E"), colors.HexColor("#B7791F")

def _table(head, rows, widths, level_col=None):
    t = Table([head] + rows, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8),
          ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C5CBD3")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F7")])]
    if level_col is not None:
        for i, r in enumerate(rows, 1):
            if r[level_col] == "High": st += [("TEXTCOLOR", (level_col, i), (level_col, i), RED), ("FONTNAME", (level_col, i), (level_col, i), "Helvetica-Bold")]
            elif r[level_col] == "Medium": st.append(("TEXTCOLOR", (level_col, i), (level_col, i), AMBER))
    t.setStyle(TableStyle(st)); return t

def build_pdf(run, students, centers, pairs):
    buf = io.BytesIO(); s = getSampleStyleSheet()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm, title="Examination Irregularity Report")
    el = [Paragraph("Examination Irregularity Investigation Report", s["Title"]),
          Paragraph(f"Dataset: {run['filename']} &nbsp;|&nbsp; Analysed: {run['created_at']} &nbsp;|&nbsp; Run #{run['id']}", s["Normal"]), Spacer(1, 6),
          Paragraph(f"{run['n_students']} students, {run['n_centers']} centres and {run['n_questions']} questions were analysed. "
                    f"<b>{run['n_high_students']}</b> students and <b>{run['n_high_centers']}</b> centres are rated High risk. "
                    "Risk scores (0-100) come from an Isolation Forest over answer-similarity and score features. A high score is a lead for investigation, not proof of malpractice.", s["Normal"]),
          Spacer(1, 10), Paragraph("High-risk centres", s["Heading2"])]
    hc = centers[centers.risk_level != "Low"].head(15)
    el.append(_table(["Centre", "Students", "Avg score", "Flagged %", "Risk", "Level"],
                     [[r.center_id, r.n_students, f"{r.avg_score:.1f}", f"{r.pct_flagged:.0f}", f"{r.risk_score:.0f}", r.risk_level] for r in hc.itertuples()] or [["None", "", "", "", "", ""]],
                     [30*mm, 25*mm, 30*mm, 30*mm, 25*mm, 25*mm], 5))
    el += [Spacer(1, 10), Paragraph("Top 25 high-risk students", s["Heading2"])]
    top = students.head(25)
    el.append(_table(["Student", "Centre", "Score", "Identical wrong", "Risk", "Level"],
                     [[r.student_id, r.center_id, f"{r.score:.0f}%", r.max_same_wrong, f"{r.risk_score:.0f}", r.risk_level] for r in top.itertuples()],
                     [40*mm, 25*mm, 25*mm, 35*mm, 20*mm, 20*mm], 5))
    el += [Spacer(1, 10), Paragraph("Most similar answer-sheet pairs", s["Heading2"])]
    el.append(_table(["Student A", "Student B", "Centre", "Similarity", "Identical wrong"],
                     [[r.student_a, r.student_b, r.center_id, f"{r.similarity:.0f}%", r.same_wrong] for r in pairs.head(20).itertuples()] or [["None", "", "", "", ""]],
                     [40*mm, 40*mm, 25*mm, 30*mm, 30*mm]))
    doc.build(el); buf.seek(0); return buf
