"""
Automated Forensic PDF Investigation Report Generator using ReportLab.
Generates an institutional audit report with executive summaries, risk scores,
psychometric collusion analysis, center-level leak detections, and legal disclaimers.
"""

import io
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable

INK = colors.HexColor("#0F172A")
NAVY = colors.HexColor("#1E3A8A")
RED = colors.HexColor("#DC2626")
AMBER = colors.HexColor("#D97706")
GREEN = colors.HexColor("#16A34A")
LIGHT_GRAY = colors.HexColor("#F8FAFC")
BORDER_GRAY = colors.HexColor("#CBD5E1")


def _build_table(head, rows, widths, level_col=None):
    """Formats a clean, bordered table with alternating background colors."""
    t = Table([head] + rows, colWidths=widths, repeatRows=1)
    st = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]
    if level_col is not None:
        for i, r in enumerate(rows, 1):
            if level_col < len(r):
                lvl = str(r[level_col]).strip()
                if lvl == "High":
                    st += [
                        ("TEXTCOLOR", (level_col, i), (level_col, i), RED),
                        ("FONTNAME", (level_col, i), (level_col, i), "Helvetica-Bold"),
                    ]
                elif lvl == "Medium":
                    st.append(("TEXTCOLOR", (level_col, i), (level_col, i), AMBER))
                elif lvl == "Low":
                    st.append(("TEXTCOLOR", (level_col, i), (level_col, i), GREEN))

    t.setStyle(TableStyle(st))
    return t


def build_pdf(run, students_df, centers_df, pairs_df):
    """
    Compiles a comprehensive institutional PDF audit report for an examination analysis run.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=f"Forensic_Report_Run_{run['id']}",
    )

    styles = getSampleStyleSheet()

    inst_style = ParagraphStyle(
        "InstHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=NAVY,
        alignment=1,
    )
    inst_sub = ParagraphStyle(
        "InstSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569"),
        alignment=1,
    )
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=INK,
        alignment=0,
        spaceBefore=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#64748B"),
    )

    elements = []

    # Institutional Header
    elements.append(Paragraph("ANNASAHEB DANGE COLLEGE OF ENGINEERING & TECHNOLOGY, ASHTA", inst_style))
    elements.append(Paragraph("Department of Computer Science & Engineering | Examination Integrity & Forensic Audit Division", inst_sub))
    elements.append(Spacer(1, 2 * mm))
    elements.append(HRFlowable(width="100%", thickness=1, color=BORDER_GRAY, spaceAfter=3 * mm))

    # Title Banner
    elements.append(Paragraph("AI-Based Forensic Examination Irregularity Audit Report", title_style))
    elements.append(
        Paragraph(
            f"<b>Confidential Dossier</b> | Analysis Run #{run['id']} &nbsp;|&nbsp; "
            f"Dataset: <i>{run['filename']}</i> &nbsp;|&nbsp; Generated: {run['created_at']}",
            subtitle_style,
        )
    )
    elements.append(Spacer(1, 3 * mm))

    # Executive Summary Card
    exec_summary_text = (
        f"This post-examination forensic audit analyzed <b>{run['n_students']:,}</b> candidates across "
        f"<b>{run['n_centers']}</b> examination centers evaluating <b>{run['n_questions']}</b> test items. "
        f"The Hierarchical Isolation Forest anomaly detection pipeline identified <b>{run['n_high_students']}</b> examinees "
        f"and <b>{run['n_high_centers']}</b> examination centers in the <b>High-Risk</b> tier (&ge;70/100). "
        "Flags are established via multi-variate statistical psychometrics combining response similarity, "
        "identical incorrect responses (IIR with exact binomial rejection $p$-values), item difficulty aberrance, "
        "and center baseline standard deviations ($Z$-score)."
    )
    elements.append(Paragraph(exec_summary_text, body_style))
    elements.append(Spacer(1, 3 * mm))

    # Section 1: Anomalous Examination Centers
    elements.append(Paragraph("1. High-Risk Examination Centers (Potential Question Leakage / Institutional Irregularities)", h2_style))
    hc = centers_df[centers_df["risk_level"] != "Low"].head(10)
    if not hc.empty:
        c_rows = []
        for r in hc.itertuples():
            c_rows.append([
                str(r.center_id),
                str(r.n_students),
                f"{r.avg_score:.1f}%",
                f"{r.pct_flagged:.0f}%",
                f"{r.risk_score:.0f}",
                str(r.risk_level),
            ])
    else:
        c_rows = [["None", "-", "-", "-", "-", "Normal"]]

    c_table = _build_table(
        ["Center ID", "Candidates", "Avg Score", "Collusion %", "Risk Score", "Risk Tier"],
        c_rows,
        [32 * mm, 28 * mm, 30 * mm, 30 * mm, 30 * mm, 32 * mm],
        level_col=5,
    )
    elements.append(c_table)
    elements.append(Spacer(1, 3 * mm))

    # Section 2: Top Suspicious Examinees
    elements.append(Paragraph("2. Top Priority Examinees Recommended for Review", h2_style))
    top_students = students_df.head(15)
    st_rows = []
    for r in top_students.itertuples():
        st_rows.append([
            str(r.student_id),
            str(r.center_id),
            f"{r.score:.0f}%",
            f"{r.max_similarity:.0f}%",
            str(r.max_same_wrong),
            f"{r.risk_score:.0f}",
            str(r.risk_level),
        ])

    st_table = _build_table(
        ["Candidate ID", "Center ID", "Score", "Max Similarity", "Identical Wrong (IIR)", "Risk Score", "Risk Tier"],
        st_rows,
        [30 * mm, 24 * mm, 20 * mm, 28 * mm, 32 * mm, 24 * mm, 24 * mm],
        level_col=6,
    )
    elements.append(st_table)
    elements.append(Spacer(1, 3 * mm))

    # Section 3: Most Anomalous Collusion Pairs
    elements.append(Paragraph("3. Most Aberrant Examinee Collusion Pairs (Psychometric Concordance)", h2_style))
    top_pairs = pairs_df.head(12) if not pairs_df.empty else None
    if top_pairs is not None and not top_pairs.empty:
        p_rows = []
        for r in top_pairs.itertuples():
            p_val = getattr(r, "p_value", 1.0)
            p_str = f"p < 1e-5" if p_val < 1e-5 else f"p={p_val:.3f}"
            p_rows.append([
                str(r.student_a),
                str(r.student_b),
                str(r.center_id),
                f"{r.similarity:.0f}%",
                str(r.same_wrong),
                p_str,
            ])
    else:
        p_rows = [["None", "None", "-", "-", "-", "-"]]

    p_table = _build_table(
        ["Candidate A", "Candidate B", "Center", "Similarity", "Shared Wrong (IIR)", "p-Value (H0)"],
        p_rows,
        [32 * mm, 32 * mm, 24 * mm, 28 * mm, 34 * mm, 32 * mm],
    )
    elements.append(p_table)
    elements.append(Spacer(1, 4 * mm))

    # Verification Signatures Block
    sign_table = Table(
        [
            ["Report Generated By:", "Verified By:", "Approved By:"],
            ["\n\n__________________________", "\n\n__________________________", "\n\n__________________________"],
            ["AI Forensics System Operator", "Forensic Audit Committee", "Controller of Examinations"],
        ],
        colWidths=[60 * mm, 60 * mm, 62 * mm],
    )
    sign_table.setStyle(
        TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#334155")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    elements.append(sign_table)
    elements.append(Spacer(1, 3 * mm))

    # Disclaimer Footer
    disclaimer_text = (
        "<b>Investigative Protocol Notice:</b> This audit report is generated automatically by an unsupervised "
        "machine learning forensics framework (Dual-Stage Isolation Forest). High risk scores and flags indicate "
        "statistically improbable response patterns and aberrant answer concordance that warrant physical "
        "seating verification and academic review, not definitive judicial proof of examination malpractice."
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(elements)
    buf.seek(0)
    return buf
