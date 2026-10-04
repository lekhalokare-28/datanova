import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import plotly.express as px

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

from llm_client import call_llm


def finance_gate(df: pd.DataFrame) -> dict:
    """Task 1.1: Checks column names/types for credit/loan dataset characteristics."""
    credit_keywords = [
        "income", "loan", "credit", "debt", "dti", "default", "status",
        "delinq", "revol", "interest", "installment", "borrower", "annual_inc"
    ]
    cols = [str(c).lower() for c in df.columns]
    matched = [c for c in cols if any(k in c for k in credit_keywords)]
    score = len(matched) / max(len(cols), 1)

    is_finance = score >= 0.15 or len(matched) >= 3
    return {
        "is_finance": is_finance,
        "matched_columns": matched,
        "confidence": "High" if score > 0.4 else ("Medium" if is_finance else "Low"),
        "reason": f"Matched {len(matched)} credit-risk related columns: {matched[:5]}"
    }


def smart_profiler(df: pd.DataFrame) -> dict:
    """Task 1.2: Proposes target, flags data leakage, and computes class imbalance."""
    cols_lower = {col: str(col).lower() for col in df.columns}

    target_names = ["default", "loan_status", "target", "bad_loan", "delinquent", "is_default"]
    proposed_target = None
    for name in target_names:
        for orig_col, lower_col in cols_lower.items():
            if name in lower_col:
                proposed_target = orig_col
                break
        if proposed_target:
            break

    leakage_keywords = ["recoveries", "last_pymnt", "total_rec", "collection", "out_prncp"]
    leakage_cols = []
    for orig_col, lower_col in cols_lower.items():
        if any(leak in lower_col for leak in leakage_keywords):
            leakage_cols.append(orig_col)

    imbalance_ratio = "Unknown"
    if proposed_target and proposed_target in df.columns:
        counts = df[proposed_target].value_counts(normalize=True)
        if len(counts) == 2:
            min_class = counts.min() * 100
            imbalance_ratio = f"{min_class:.1f}% default rate (1:{int(counts.max() / counts.min())} ratio)"

    missing = df.isnull().sum()
    missing_dict = missing[missing > 0].to_dict()

    return {
        "proposed_target": proposed_target,
        "leakage_cols": leakage_cols,
        "imbalance_ratio": imbalance_ratio,
        "missing_values": missing_dict,
        "total_rows": len(df),
        "total_columns": len(df.columns)
    }


def human_checkpoint_apply(df: pd.DataFrame, target_col: str, cols_to_drop: list) -> pd.DataFrame:
    """Task 1.3: Drops confirmed leakage/unwanted columns and validates target existence."""
    clean_df = df.copy()
    valid_drops = [c for c in cols_to_drop if c in clean_df.columns]
    if valid_drops:
        clean_df = clean_df.drop(columns=valid_drops)
    return clean_df


def preprocess_and_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """Task 1.4: Preprocessing module (fit scalers/imputers on train only to prevent leakage)."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(num_imputer.fit_transform(X_train[num_cols]))
    X_test_num = scaler.transform(num_imputer.transform(X_test[num_cols]))

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_train_cat = encoder.fit_transform(cat_imputer.fit_transform(X_train[cat_cols]))
        X_test_cat = encoder.transform(cat_imputer.transform(X_test[cat_cols]))

        X_train_final = np.hstack([X_train_num, X_train_cat])
        X_test_final = np.hstack([X_test_num, X_test_cat])
        feature_cols = num_cols + encoder.get_feature_names_out(cat_cols).tolist()
    else:
        X_train_final = X_train_num
        X_test_final = X_test_num
        feature_cols = num_cols

    return X_train_final, X_test_final, y_train, y_test, feature_cols


def generate_eda_charts(df: pd.DataFrame, target_col: str, output_dir: str = "charts") -> list:
    """Task 1.5: Pass-1 EDA charts (class balance, distributions, correlations)."""
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []

    if target_col in df.columns:
        fig = px.histogram(df, x=target_col, title=f"Class Balance: {target_col}", color=target_col)
        path = os.path.join(output_dir, "eda_target_balance.png")
        fig.write_image(path)
        chart_paths.append(path)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    if num_cols:
        primary_num = num_cols[0]
        fig = px.histogram(df, x=primary_num, marginal="box", title=f"Distribution of {primary_num}")
        path = os.path.join(output_dir, f"eda_{primary_num}_dist.png")
        fig.write_image(path)
        chart_paths.append(path)

    if len(num_cols) >= 2:
        corr_matrix = df[num_cols[:8]].corr()
        fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix (Top Financial Features)")
        path = os.path.join(output_dir, "eda_correlation_matrix.png")
        fig.write_image(path)
        chart_paths.append(path)

    return chart_paths


def sanitize_llm_text(text: str) -> str:
    replacements = {
        '\u2011': '-', '\u2010': '-', '\u2013': '-', '\u2014': '-', '\u2212': '-',
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
    }
    for bad_char, good_char in replacements.items():
        text = text.replace(bad_char, good_char)
    return text


def clean_markdown_for_ppt(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'^\s*[-*•]\s*', '', text)
    return text.strip()


def build_finance_prompt(state: dict, report_style: str) -> str:
    target_col = state.get("target_col", "default")
    leakage_cols = state.get("leakage_cols", [])
    imbalance = state.get("imbalance_ratio", "N/A")
    selected_features = state.get("selected_features", [])
    pso_runtime = state.get("pso_runtime", "N/A")
    best_model_name = state.get("best_model_name", "N/A")
    metrics = state.get("metrics", {})
    shap_summary = state.get("shap_summary", [])
    comparison_table = state.get("comparison_table", [])

    return f"""
You are an expert Credit Risk Quantitative Analyst preparing an executive loan default risk report.
Report Style: {report_style.upper()}

Dataset & Preprocessing Facts:
- Target Variable: {target_col} (binary default classification)
- Data Leakage Excluded: {leakage_cols if leakage_cols else 'None detected'}
- Class Imbalance: {imbalance}

Feature Optimization (PSO):
- Features Selected: {len(selected_features)} features ({', '.join(selected_features[:8]) if selected_features else 'All'})
- PSO Search Runtime: {pso_runtime}s

Model Evaluation & Benchmarking:
- Champion Model: {best_model_name}
- Test Set Metrics: {json.dumps(metrics)}
- Baseline Comparison: {json.dumps(comparison_table)}

Explainability (SHAP Risk Factors):
- Top Drivers: {json.dumps(shap_summary)}

Write an authoritative credit-risk report structured strictly under these headings:
Section: Executive / Dataset Overview
Section: Feature Selection & Model Evaluation
Section: Explainability & Credit Risk Takeaways

Rules:
1. Reference exact numbers (AUC, F1, imbalance, runtime, feature counts).
2. Detail data leakage prevention and why AUC/F1 are prioritized over accuracy.
"""


def parse_presentation_sections(narrative: str) -> dict:
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
        lower = line.lower()
        if "overview" in lower:
            current_section = "Executive / Dataset Overview"
            continue
        elif "feature" in lower or "model" in lower or "evaluation" in lower:
            current_section = "Feature Selection & Model Evaluation"
            continue
        elif "explainability" in lower or "takeaway" in lower or "risk" in lower:
            current_section = "Explainability & Credit Risk Takeaways"
            continue

        clean_bullet = clean_markdown_for_ppt(line)
        if clean_bullet and len(clean_bullet) > 3:
            sections[current_section].append(clean_bullet)
    return sections


def generate_presentation(state: dict, narrative: str, chart_paths: list) -> str:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
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
    p1.font.size, p1.font.bold = Pt(36), True
    p1.font.color.rgb = primary_color

    p2 = tf1.add_paragraph()
    p2.text = "Autonomous Bio-Inspired (PSO) Credit Modeling Pipeline"
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
        hp.font.size, hp.font.bold = Pt(26), True
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
        hp.font.size, hp.font.bold = Pt(26), True
        hp.font.color.rgb = primary_color

        rows, cols = len(comparison_table) + 1, len(comparison_table[0].keys())
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
            hp.text = f"Visual Exhibit: {chart_title}"
            hp.font.size, hp.font.bold = Pt(24), True
            hp.font.color.rgb = primary_color

            chart_slide.shapes.add_picture(chart_path, Inches(1.66), Inches(1.5), width=Inches(10.0))

    ppt_path = "reports/eda_presentation_report.pptx"
    prs.save(ppt_path)
    return ppt_path


def build_metrics_summary_table(state: dict) -> Table:
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
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DDE2E5")),
    ]))
    return t


def render_narrative_to_story(narrative: str, styles) -> list:
    story = []
    known_headers = [
        "Executive / Dataset Overview", "Feature Selection & Model Evaluation",
        "Explainability & Credit Risk Takeaways"
    ]
    body_style = ParagraphStyle("NarrativeBody", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6)
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
        clean_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', clean)
        story.append(Paragraph(clean_html, body_style))
        story.append(Spacer(1, 6))
    return story


def reporter_node(state: dict) -> dict:
    """Consolidates inputs across Member 1 and Member 3 to generate outputs."""
    os.makedirs("reports", exist_ok=True)
    report_style = state.get("report_style", "executive")

    chart_paths = []
    for k in ["chart_paths", "eda_charts", "model_charts"]:
        if k in state:
            chart_paths.extend([c for c in state[k] if c not in chart_paths])

    prompt = build_finance_prompt(state, report_style)
    narrative = sanitize_llm_text(call_llm(prompt))
    state["insights"] = narrative

    if report_style == "presentation":
        state["pdf_path"] = generate_presentation(state, narrative, chart_paths)
        return state

    pdf_path = f"reports/credit_risk_{report_style}_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("DataNova: Credit Risk & Loan Default Report", styles["Title"]),
        Paragraph(f"Domain: Finance (Credit Risk Prediction) | Format: {report_style.capitalize()}", styles["Heading3"]),
        Spacer(1, 10),
        Paragraph("Key Risk & Model Performance Metrics", styles["Heading3"]),
        Spacer(1, 4),
        build_metrics_summary_table(state),
        Spacer(1, 14)
    ]
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
