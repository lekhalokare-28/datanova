import os
import re
import json
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from llm_client import call_llm


def sanitize_llm_text(text: str) -> str:
    """LLMs sometimes write compound words using a non-breaking hyphen
    (U+2011) instead of a plain ASCII hyphen, which some PDF viewers
    render as a broken/wrong glyph. Normalize dash and smart-quote
    characters to plain ASCII equivalents before building the PDF."""
    replacements = {
        '\u2011': '-', '\u2010': '-', '\u2013': '-', '\u2014': '-', '\u2212': '-',
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
    }
    for bad_char, good_char in replacements.items():
        text = text.replace(bad_char, good_char)
    return text


def convert_markdown_bold(text: str) -> str:
    """Converts **bold** (Markdown, which the LLM sometimes uses despite
    being told not to) into <b>bold</b>, ReportLab's actual supported tag."""
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)


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


def render_narrative_to_story(narrative: str, styles) -> list:
    """Turns the LLM's narrative into properly structured PDF elements:
    - Recognized section headers get a distinct heading style
    - Numbered takeaway/action lists become a real bulleted list
    - **bold** markdown gets converted to ReportLab's <b> tag
    """
    story = []
    known_headers = [
        "Executive/Dataset Overview", "Dataset Overview",
        "Data Quality & Profiling Insights", "Data Quality and Profiling Insights",
        "Strategic / Operational Takeaways", "Strategic/Operational Takeaways",
    ]

    body_style = ParagraphStyle("NarrativeBody", parent=styles["Normal"],
        fontSize=10.5, leading=15, spaceAfter=8)
    heading_style = ParagraphStyle("NarrativeHeading", parent=styles["Heading2"],
        fontSize=13, spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#16213e"))
    list_item_style = ParagraphStyle("ListItem", parent=styles["Normal"],
        fontSize=10.5, leading=15, spaceAfter=6)

    list_item_pattern = re.compile(
        r'(\d+)\.\s+\*\*(.+?)\*\*\s*[-\u2013\u2014]?\s*(.*?)(?=\s*\d+\.\s+\*\*|$)',
        re.DOTALL
    )

    for section in narrative.split("\n\n"):
        clean = section.strip().replace("\n", " ")
        if not clean:
            continue

        header_match = next((h for h in known_headers if clean.startswith(h)), None)
        if header_match and len(clean) < len(header_match) + 10:
            story.append(Paragraph(clean, heading_style))
            continue

        list_matches = list_item_pattern.findall(clean)
        if len(list_matches) >= 2:
            first_marker_pos = clean.find(list_matches[0][0] + ".")
            preamble = clean[:first_marker_pos].strip()
            if preamble:
                story.append(Paragraph(convert_markdown_bold(preamble), body_style))

            items = []
            for num, title, desc in list_matches:
                item_html = f"<b>{title.strip()}</b> - {desc.strip()}"
                items.append(ListItem(Paragraph(item_html, list_item_style)))

            story.append(ListFlowable(items, bulletType="1", start="1"))
            story.append(Spacer(1, 8))
            continue

        story.append(Paragraph(convert_markdown_bold(clean), body_style))
        story.append(Spacer(1, 8))

    return story


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
    narrative = sanitize_llm_text(narrative)

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

    story.extend(render_narrative_to_story(narrative, styles))

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