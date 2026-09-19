import os
import re
import json
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from llm_client import call_llm




def sanitize_llm_text(text: str) -> str:
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
    """Strips Markdown bolding and excessive symbols for PowerPoint slides."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'^\s*[-*•]\s*', '', text)
    return text.strip()


def build_pptx_report(narrative, domain_name, report_style, chart_paths):
    prs = Presentation()

    # Slide 1: Title
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = "DataNova: Automated EDA Report"
    title_slide.placeholders[1].text = f"Domain Detected: {domain_name.capitalize()}"

    # One slide per narrative section
    for section in narrative.split("\n\n"):
        clean = section.strip()
        if not clean:
            continue

        slide = prs.slides.add_slide(prs.slide_layouts[1])
        lines = clean.split(" - ")

        slide.shapes.title.text = lines[0][:60]

        body = slide.placeholders[1].text_frame

        for line in lines[1:]:
            p = body.add_paragraph()
            p.text = line.strip()
            p.level = 0

    # Chart slides
    for chart_path in chart_paths:
        if os.path.exists(chart_path):
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            slide.shapes.add_picture(
                chart_path,
                Inches(1),
                Inches(1),
                width=Inches(8)
            )

    pptx_path = "reports/eda_presentation_report.pptx"
    prs.save(pptx_path)

    return pptx_path

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
            "Write punchy, bullet-style takeaways suitable for a slide deck. "
            "Keep each point concise (1-2 sentences maximum). "
            "Explicitly divide the content with the exact section titles below."
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

Write a clean narrative report structured strictly into these 3 sections:
Section: Executive / Dataset Overview
Section: Data Quality & Profiling Insights
Section: Strategic / Operational Takeaways

Under each section, provide 3 to 5 concise bullet points.
"""
    return prompt


def parse_presentation_sections(narrative: str) -> dict:
    """Parses LLM narrative text into individual slides by section header."""
    sections = {
        "Executive / Dataset Overview": [],
        "Data Quality & Profiling Insights": [],
        "Strategic / Operational Takeaways": []
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
        elif "quality" in lower_line or "profiling" in lower_line:
            current_section = "Data Quality & Profiling Insights"
            continue
        elif "strategic" in lower_line or "takeaway" in lower_line or "operational" in lower_line:
            current_section = "Strategic / Operational Takeaways"
            continue

        clean_bullet = clean_markdown_for_ppt(line)
        if clean_bullet and len(clean_bullet) > 3:
            sections[current_section].append(clean_bullet)

    return sections


def generate_presentation(domain: dict, narrative: str, chart_paths: list) -> str:
    """Builds a formatted 16:9 PowerPoint presentation."""
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    primary_color = RGBColor(22, 33, 62)     
    accent_color = RGBColor(0, 102, 204)    
    text_color = RGBColor(40, 40, 40)    
    domain_name = domain.get("domain", "Generic").capitalize() if isinstance(domain, dict) else str(domain).capitalize()

   
    slide1 = prs.slides.add_slide(blank_layout)
    title_box = slide1.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(11), Inches(2.5))
    tf1 = title_box.text_frame
    tf1.word_wrap = True

    p1 = tf1.paragraphs[0]
    p1.text = "DataNova: Automated EDA Report"
    p1.font.size = Pt(40)
    p1.font.bold = True
    p1.font.color.rgb = primary_color

    p2 = tf1.add_paragraph()
    p2.text = f"Detected Domain: {domain_name} | Automated Intelligence Engine"
    p2.font.size = Pt(20)
    p2.font.color.rgb = accent_color
    p2.space_before = Pt(16)

    
    def add_content_slide(title_text, bullet_list):
        slide = prs.slides.add_slide(blank_layout)

        
        header_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.33), Inches(0.8))
        htf = header_box.text_frame
        hp = htf.paragraphs[0]
        hp.text = title_text
        hp.font.size = Pt(28)
        hp.font.bold = True
        hp.font.color.rgb = primary_color

       
        content_box = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.33), Inches(5.0))
        ctf = content_box.text_frame
        ctf.word_wrap = True

        for idx, bullet in enumerate(bullet_list):
            bp = ctf.paragraphs[0] if idx == 0 else ctf.add_paragraph()
            bp.text = f"•  {bullet}"
            bp.font.size = Pt(18)
            bp.font.color.rgb = text_color
            bp.space_after = Pt(14)

    
    parsed_sections = parse_presentation_sections(narrative)
    for title, bullets in parsed_sections.items():
        if bullets:
            add_content_slide(title, bullets)

    
    for idx, chart_path in enumerate(chart_paths):
        if os.path.exists(chart_path):
            chart_slide = prs.slides.add_slide(blank_layout)
            header_box = chart_slide.shapes.add_textbox(Inches(1.0), Inches(0.6), Inches(11.33), Inches(0.8))
            hp = header_box.text_frame.paragraphs[0]
            hp.text = f"Visual Analysis (Exhibit {idx + 1})"
            hp.font.size = Pt(26)
            hp.font.bold = True
            hp.font.color.rgb = primary_color

           
            chart_slide.shapes.add_picture(
                chart_path,
                Inches(1.66),
                Inches(1.6),
                width=Inches(10.0)
            )

    ppt_path = "reports/eda_presentation_report.pptx"
    prs.save(ppt_path)
    return ppt_path


def render_narrative_to_story(narrative: str, styles) -> list:
    """PDF Story renderer for Executive and Technical styles."""
    story = []
    known_headers = [
        "Executive/Dataset Overview", "Dataset Overview",
        "Data Quality & Profiling Insights", "Data Quality and Profiling Insights",
        "Strategic / Operational Takeaways", "Strategic/Operational Takeaways",
        "Executive / Dataset Overview:", "Data Quality & Profiling Insights:",
        "Strategic / Operational Takeaways:"
    ]

    body_style = ParagraphStyle("NarrativeBody", parent=styles["Normal"],
        fontSize=10.5, leading=15, spaceAfter=8)
    heading_style = ParagraphStyle("NarrativeHeading", parent=styles["Heading2"],
        fontSize=13, spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#16213e"))

    for section in narrative.split("\n\n"):
        clean = section.strip().replace("\n", " ")
        if not clean:
            continue

        header_match = next((h for h in known_headers if clean.startswith(h)), None)
        if header_match and len(clean) < len(header_match) + 10:
            story.append(Paragraph(clean, heading_style))
            continue

        story.append(Paragraph(convert_markdown_bold(clean), body_style))
        story.append(Spacer(1, 8))

    return story


def reporter_node(state):
    """
    Agent 3 Node:
    - If report_style == 'presentation': generates a clean .pptx deck.
    - If report_style == 'executive' or 'technical': generates a ReportLab PDF document.
    """
    os.makedirs("reports", exist_ok=True)

    report_style = state.get("report_style", "executive")
    domain = state.get("domain", {})
    profile_report = state.get("profile_report", {})
    chart_paths = state.get("chart_paths", [])

    prompt = build_prompt(domain, profile_report, report_style)
    narrative = call_llm(prompt)
    narrative = sanitize_llm_text(narrative)

    state["insights"] = narrative

    
    if report_style == "presentation":
        pptx_path = generate_presentation(domain, narrative, chart_paths)
        state["pdf_path"] = pptx_path
        return state

    
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

    domain_name = domain.get("domain", "Generic").capitalize() if isinstance(domain, dict) else str(domain).capitalize()
    story.append(Paragraph("DataNova: Automated EDA Report", styles["Title"]))
    story.append(Paragraph(f"Format: {report_style.capitalize()} | Domain Detected: {domain_name}", styles["Heading3"]))
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
    state["pdf_path"] = pdf_path
    return state
    if report_style == "presentation":
        output_path = build_pptx_report(narrative, domain_name, report_style, chart_paths)
    else:
        doc.build(story)
        output_path = pdf_path

    state["insights"] = narrative
    state["pdf_path"] = output_path
    return state
