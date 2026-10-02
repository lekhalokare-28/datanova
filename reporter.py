import os
import re
import json

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

from llm_client import call_llm


def sanitize_llm_text(text: str) -> str:
    """Normalize unicode dashes, quotes, and symbols to standard ASCII."""
    replacements = {
        '\u2011': '-', '\u2010': '-', '\u2013': '-', '\u2014': '-', '\u2212': '-',
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
    }
    for bad_char, good_char in replacements.items():
        text = text.replace(bad_char, good_char)
    return text


def convert_markdown_bold(text: str) -> str:
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)


def clean_markdown_for_ppt(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'^\s*[-*•]\s*', '', text)
    return text.strip()


def build_finance_prompt(state: dict, report_style: str) -> str:
    """Builds a finance/credit-risk prompt incorporating real pipeline metrics."""
    target_col = state.get("target_col", "default")
    leakage_cols = state.get("leakage_cols", [])
    imbalance = state.get("imbalance_ratio", "N/A")
    selected_features = state.get("selected_features", [])
    pso_runtime = state.get("pso_runtime", "N/A")
    best_model_name = state.get("best_model_name", "N/A")
    metrics = state.get("metrics", {})
    shap_summary = state.get("shap_summary", [])
    comparison_table = state.get("comparison_table", [])

    prompt = f"""
You are an expert Credit Risk Quantitative Analyst and Data Scientist preparing an executive loan default risk report.
Report Style: {report_style.upper()}

Dataset & Preprocessing Facts:
- Target Variable: {target_col} (binary credit default classification)
- Data Leakage Excluded: {leakage_cols if leakage_cols else 'None detected'}
- Class Imbalance: {imbalance}

Feature Optimization (Particle Swarm Optimization - PSO):
- Features Selected by PSO: {len(selected_features)} features ({', '.join(selected_features[:8]) if selected_features else 'All'})
- PSO Search Runtime: {pso_runtime}s

Model Evaluation & Benchmarking:
- Champion Model: {best_model_name}
- Test Set Evaluation: {json.dumps(metrics)}
- Baseline Comparison Summary: {json.dumps(comparison_table)}

Explainability (SHAP Credit Risk Factors):
- Top Risk Drivers: {json.dumps(shap_summary)}

Instructions:
Write an authoritative credit-risk analysis report structured strictly under these headings:
Section: Executive / Dataset Overview
Section: Feature Selection & Model Evaluation
Section: Explainability & Credit Risk Takeaways

Rules:
1. You MUST reference the exact metrics (AUC, F1, exact imbalance, runtime, feature counts) provided above.
2. Explain the business importance of data leakage prevention and why AUC/F1 are prioritized over accuracy due to class imbalance.
3. Keep the content clear, quantitative, and professional.
"""
    return prompt


def parse_presentation_sections(narrative: str) -> dict:
    """Parses LLM narrative text into individual slides by section header."""
    sections = {
        "Executive / Dataset Overview": [],
        "Feature Selection & Model Evaluation": [],
        "Explainability & Credit Risk Takeaways": [],
    }
    current_section = "Executive / Dataset Overview"

    for raw_line in narrative.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        lower_line = line.lower()
        if "overview" in lower_line:
            current_section = "Executive / Dataset Overview"
            continue
        elif "feature" in lower_line or "model" in lower_line or "evaluation" in lower_line:
            current_section = "Feature Selection & Model Evaluation"
            continue
        elif "explainability" in lower_line or "takeaway" in lower_line or "risk" in lower_line:
            current_section = "Explainability & Credit Risk Takeaways"
            continue

        clean_bullet = clean_markdown_for_ppt(line)
        if clean_bullet and len(clean_bullet) > 3:
            sections[current_section].append(clean_bullet)

    return sections


def generate_presentation(state: dict, narrative: str, chart_paths: list) -> str:
    """Builds a structured 16:9 PowerPoint presentation with metrics and tables."""
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    primary_color = RGBColor(16, 37, 66)
    accent_color = RGBColor(0, 119, 182)
    text_color = RGBColor(40, 40, 40)

    slide1 = prs.slides.add_slide(blank_layout)
    title_box = slide1.shapes.add_textbox(Inches(1.2), Inches(2.0), Inches(11), Inches(3.0))
    tf1 = title_box.text_frame
    tf1.word_wrap = True

    p1 = tf1.paragraphs[0]
    p1.text = "DataNova: Credit Risk & Loan Default Intelligence"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = primary_color

    p2 = tf1.add_paragraph()
    p2.text = f"Autonomous Bio-Inspired (PSO) Credit Modeling Pipeline"
    p2.font.size = Pt(20)
    p2.font.color.rgb = accent_color
    p2.space_before = Pt(14)

    best_model = state.get("best_model_name", "N/A")
    auc_val = state.get("metrics", {}).get("AUC", state.get("metrics", {}).get("roc_auc", "N/A"))
    p3 = tf1.add_paragraph()
    p3.text = f"Champion Model: {best_model} | Test ROC-AUC: {auc_val}"
    p3.font.size = Pt(16)
    p3.font.color.rgb = text_color
    p3.space_before = Pt(10)

    def add_content_slide(title_text, bullet_list):
        slide = prs.slides.add_slide(blank_layout)

        header_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.33), Inches(0.8))
        hp = header_box.text_frame.paragraphs[0]
        hp.text = title_text
        hp.font.size = Pt(26)
        hp.font.bold = True
        hp.font.color.rgb = primary_color

        content_box = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.33), Inches(5.0))
        ctf = content_box.text_frame
        ctf.word_wrap = True

        for idx, bullet in enumerate(bullet_list[:5]):
            bp = ctf.paragraphs[0] if idx == 0 else ctf.add_paragraph()
            bp.text = f"•  {bullet}"
            bp.font.size = Pt(17)
            bp.font.color.rgb = text_color
            bp.space_after = Pt(12)

    parsed_sections = parse_presentation_sections(narrative)
    for title, bullets in parsed_sections.items():
        if bullets:
            add_content_slide(title, bullets)

    comparison_table = state.get("comparison_table", [])
    if comparison_table and isinstance(comparison_table, list) and len(comparison_table) > 0:
        table_slide = prs.slides.add_slide(blank_layout)
        header_box = table_slide.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.33), Inches(0.8))
        hp = header_box.text_frame.paragraphs[0]
        hp.text = "Benchmark: PSO vs. Baseline Methods"
        hp.font.size = Pt(26)
        hp.font.bold = True
        hp.font.color.rgb = primary_color

        rows = len(comparison_table) + 1
        cols = len(comparison_table[0].keys())
        table_shape = table_slide.shapes.add_table(rows, cols, Inches(1.0), Inches(1.8), Inches(11.33), Inches(4.5))
        tbl = table_shape.table

        headers = list(comparison_table[0].keys())
        for c_idx, head in enumerate(headers):
            cell = tbl.cell(0, c_idx)
            cell.text = head.replace("_", " ").upper()
            cell.text_frame.paragraphs[0].font.bold = True
            cell.text_frame.paragraphs[0].font.size = Pt(14)

        for r_idx, row in enumerate(comparison_table):
            for c_idx, head in enumerate(headers):
                cell = tbl.cell(r_idx + 1, c_idx)
                cell.text = str(row.get(head, ""))
                cell.text_frame.paragraphs[0].font.size = Pt(13)

    for idx, chart_path in enumerate(chart_paths):
        if os.path.exists(chart_path):
            chart_slide = prs.slides.add_slide(blank_layout)
            header_box = chart_slide.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.33), Inches(0.8))
            hp = header_box.text_frame.paragraphs[0]
            chart_title = os.path.basename(chart_path).replace("_", " ").replace(".png", "").title()
            hp.text = f"Visual Analysis: {chart_title}"
            hp.font.size = Pt(24)
            hp.font.bold = True
            hp.font.color.rgb = primary_color

            chart_slide.shapes.add_picture(
                chart_path,
                Inches(1.66),
                Inches(1.5),
                width=Inches(10.0),
            )

    ppt_path = "reports/eda_presentation_report.pptx"
    prs.save(ppt_path)
    return ppt_path


def render_narrative_to_story(narrative: str, styles) -> list:
    """PDF Flowable builder for narrative sections."""
    story = []
    known_headers = [
        "Executive / Dataset Overview", "Executive/Dataset Overview",
        "Feature Selection & Model Evaluation", "Feature Selection and Model Evaluation",
        "Explainability & Credit Risk Takeaways", "Explainability and Credit Risk Takeaways"
    ]

    body_style = ParagraphStyle(
        "NarrativeBody", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6
    )
    heading_style = ParagraphStyle(
        "NarrativeHeading", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6,
        textColor=colors.HexColor("#102542")
    )

    for section in narrative.split("\n\n"):
        clean = section.strip().replace("\n", " ")
        if not clean:
            continue

        header_match = next((h for h in known_headers if clean.startswith(h)), None)
        if header_match and len(clean) < len(header_match) + 15:
            story.append(Paragraph(clean, heading_style))
            continue

        story.append(Paragraph(convert_markdown_bold(clean), body_style))
        story.append(Spacer(1, 6))

    return story


def build_metrics_summary_table(state: dict) -> Table:
    """Generates an executive KPI table for the PDF."""
    metrics = state.get("metrics", {})
    selected_features = state.get("selected_features", [])

    data = [
        ["Target Column", str(state.get("target_col", "default")), "Class Imbalance", str(state.get("imbalance_ratio", "N/A"))],
        ["Best Model", str(state.get("best_model_name", "N/A")), "Features Selected (PSO)", f"{len(selected_features)} features"],
        ["Test ROC-AUC", str(metrics.get("AUC", metrics.get("roc_auc", "N/A"))), "Test F1-Score", str(metrics.get("F1", metrics.get("f1_score", "N/A")))],
        ["Precision", str(metrics.get("Precision", metrics.get("precision", "N/A"))), "Recall", str(metrics.get("Recall", metrics.get("recall", "N/A")))],
        ["PSO Runtime", f"{state.get('pso_runtime', 'N/A')}s", "Data Leakage Dropped", f"{len(state.get('leakage_cols', []))} columns"]
    ]

    t = Table(data, colWidths=[120, 140, 140, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8F9FA")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#212529")),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DDE2E5")),
    ]))
    return t


def reporter_node(state):
    """
    Agent Reporter Node:
    Takes finance credit state (PSO, models, metrics, leakage, charts) and
    produces the requested report format (.pdf or .pptx).
    """
    os.makedirs("reports", exist_ok=True)

    report_style = state.get("report_style", "executive")
    
    chart_paths = []
    if "chart_paths" in state:
        chart_paths.extend(state["chart_paths"])
    if "eda_charts" in state:
        chart_paths.extend([c for c in state["eda_charts"] if c not in chart_paths])
    if "model_charts" in state:
        chart_paths.extend([c for c in state["model_charts"] if c not in chart_paths])

    prompt = build_finance_prompt(state, report_style)
    narrative = call_llm(prompt)
    narrative = sanitize_llm_text(narrative)
    state["insights"] = narrative

    if report_style == "presentation":
        pptx_path = generate_presentation(state, narrative, chart_paths)
        state["pdf_path"] = pptx_path
        return state

    pdf_path = f"reports/credit_risk_{report_style}_report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("DataNova: Credit Risk & Loan Default Report", styles["Title"]))
    story.append(Paragraph(
        f"Domain: Finance (Credit Risk Prediction) | Format: {report_style.capitalize()}",
        styles["Heading3"]
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Key Risk & Model Performance Metrics", styles["Heading3"]))
    story.append(Spacer(1, 4))
    story.append(build_metrics_summary_table(state))
    story.append(Spacer(1, 14))

    story.extend(render_narrative_to_story(narrative, styles))

    if chart_paths:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Visual Exhibits (EDA, Optimization & Explainability)", styles["Heading2"]))
        story.append(Spacer(1, 8))
        for img_path in chart_paths:
            if os.path.exists(img_path):
                story.append(Image(img_path, width=480, height=260))
                story.append(Spacer(1, 10))

    doc.build(story)
    state["pdf_path"] = pdf_path
    return state
