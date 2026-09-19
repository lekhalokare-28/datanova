import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from llm_client import call_llm

def build_prompt(domain, profile_report, report_style):
    style_guidelines = {
        "executive": (
            "Write a concise, 1-2 page high-level executive summary for decision-makers. "
            "Focus on high-level business impact, overall data health, and key takeaways without heavy technical jargon."
        ),
        "technical": (
            "Write a detailed technical deep-dive for data analysts. "
            "Include exact missing value counts, outlier detections (IQR), data type issues, and analytical recommendations."
        ),
        "presentation": (
            "Write punchy, bullet-style takeaways suitable for a slide-deck-style layout."
        )
    }
    
    guideline = style_guidelines.get(report_style, style_guidelines["executive"])
    
    prompt = f"""
You are an expert Data Scientist preparing an automated EDA report.
Report Style: {report_style.upper()}
Guideline: {guideline}

Domain Information:
{json.dumps(domain, indent=2, default=str)}

Data Quality Findings:
{json.dumps(profile_report, indent=2, default=str)}

Write a clean narrative report structured into clear sections:
1. Executive/Dataset Overview
2. Data Quality & Profiling Insights
3. Strategic / Operational Takeaways
Do not use Markdown symbols like ** or ## in your text output; write plain paragraphs and section names.
"""
    return prompt

def reporter_node(state):
    """
    Agent 3 Node: Consumes domain, profile_report, chart_paths, and report_style.
    Produces state['insights'] and compiles state['pdf_path'].
    """
    
    os.makedirs("reports", exist_ok=True)
    
    
    report_style = state.get("report_style", "executive")
    domain = state.get("domain", {})
    profile_report = state.get("profile_report", {})
    chart_paths = state.get("chart_paths", [])
    
    
    prompt = build_prompt(domain, profile_report, report_style)
    narrative = call_llm(prompt)
    
    
    pdf_path = f"reports/eda_{report_style}_report.pdf"
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
    
    
    domain_name = domain.get("domain", "Generic") if isinstance(domain, dict) else str(domain)
    story.append(Paragraph(f"DataNova: Automated EDA Report", styles["Title"]))
    story.append(Paragraph(f"Format: {report_style.capitalize()} | Domain Detected: {domain_name.capitalize()}", styles["Heading3"]))
    story.append(Spacer(1, 14))
    
    
    for section in narrative.split("\n\n"):
        clean_text = section.strip().replace("\n", " ")
        if clean_text:
            story.append(Paragraph(clean_text, styles["Normal"]))
            story.append(Spacer(1, 10))
            
    
    if chart_paths:
        story.append(Spacer(1, 14))
        story.append(Paragraph("Visual Analysis", styles["Heading2"]))
        story.append(Spacer(1, 10))
        for img_path in chart_paths:
            if os.path.exists(img_path):
                story.append(Image(img_path, width=450, height=250))
                story.append(Spacer(1, 12))
                
    
    doc.build(story)
    
    state["insights"] = narrative
    state["pdf_path"] = pdf_path
    return state
