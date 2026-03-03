"""
=============================================================================
PATTENS - R&D Tax Incentive Claims
AI Pipeline: Form Data → GPT-4o → Dashboard PDF
=============================================================================

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────────┐
│  Django View receives POST with form data (7 sections)                  │
│       ↓                                                                  │
│  Step 1: GPT-4o generates structured JSON for all charts                │
│          (uses built-in Australian R&D tax law knowledge)               │
│       ↓                                                                  │
│  Step 2: PDF Builder (ReportLab + matplotlib → professional dashboard)  │
│       ↓                                                                  │
│  Returns: PDF file path                                                  │
└─────────────────────────────────────────────────────────────────────────┘

INSTALL:
    pip install openai reportlab matplotlib numpy

USAGE IN DJANGO VIEW:
    from .rd_ai_pipeline import generate_rd_dashboard_pdf

    def export_pdf(request):
        pdf_path = generate_rd_dashboard_pdf(
            form_data=request.POST.dict(),
            openai_api_key=settings.OPENAI_API_KEY
        )
        return FileResponse(open(pdf_path, 'rb'), content_type='application/pdf')
"""

import os
import json
import io
import tempfile
import uuid
import textwrap
from datetime import datetime

# Load .env file
from dotenv import load_dotenv
load_dotenv()

import openai
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT


# =============================================================================
# COLOUR PALETTE
# =============================================================================
NAVY       = "#0D2B4E"
BLUE       = "#1A5276"
ACCENT     = "#2E86C1"
TEAL       = "#1ABC9C"
ORANGE     = "#E67E22"
LIGHT_GREY = "#F4F6F8"
MID_GREY   = "#BDC3C7"
WHITE      = "#FFFFFF"

RL_NAVY  = colors.HexColor(NAVY)
RL_BLUE  = colors.HexColor(BLUE)
RL_ACCENT= colors.HexColor(ACCENT)
RL_TEAL  = colors.HexColor(TEAL)
RL_ORANGE= colors.HexColor(ORANGE)
RL_LGREY = colors.HexColor(LIGHT_GREY)


# =============================================================================
# STEP 1 — GPT-4o DATA GENERATION
# =============================================================================

ATO_SYSTEM_PROMPT = """You are a senior Australian R&D Tax Incentive consultant with 
comprehensive knowledge of:
- Industry Research and Development Act 1986 (IR&D Act)
- ATO and AusIndustry joint administration guidelines
- Core R&D Activities vs Supporting R&D Activities definitions
- The three eligibility tests: new knowledge, genuine uncertainty, systematic progression
- Section 355-25 ITAA 1997 (Core R&D Activities)
- Section 355-30 ITAA 1997 (Supporting R&D Activities)
- Feedstock rules, clawback provisions, and at-risk rules
- 43.5% refundable offset for companies with aggregated turnover < $20M
- 38.5% non-refundable offset for larger companies
- ATO's 4 compliance risk zones and common audit triggers
- Documentation requirements: lab notebooks, financial records, contemporaneous evidence

You generate accurate, compliance-focused analysis of R&D claims for Australian companies.
Always return ONLY valid JSON with no markdown or explanation."""


def generate_dashboard_data(form_data: dict, openai_api_key: str) -> dict:
    """
    Sends all 7 form sections to GPT-4o.
    Returns fully structured JSON ready for PDF rendering.

    FORM DATA EXPECTED KEYS:
        project_title, brief_summary, financial_year,
        project_start_date, project_end_date, industry, staff_members,
        q1_activities, q1_hypothesis, q1_uncertainty,
        q1_systematic, q1_outcomes, q1_new_knowledge,
        q2_* ... q4_*  (same pattern),
        total_rd_expenditure, staff_costs, contractor_costs,
        materials_consumables, equipment_depreciation, other_eligible_costs
    """
    client = openai.OpenAI(api_key=openai_api_key)

    # Map q1-q4 labels to activity titles if provided
    activity_labels = []
    for i in range(1, 5):
        title = form_data.get(f"q{i}_activity_title", f"Core Activity {i}")
        if title:
            activity_labels.append(f"Activity {i}: {title}")
    activity_label_str = "\n".join(activity_labels) if activity_labels else "4 core R&D activities"

    user_prompt = f"""
FORM DATA SUBMITTED BY APPLICANT:
{json.dumps(form_data, indent=2)}

TODAY'S DATE: {datetime.now().strftime('%d %B %Y')}

CORE ACTIVITIES IN THIS CLAIM:
{activity_label_str}

Using your expert knowledge of Australian R&D Tax Incentive legislation,
analyse the above form data and generate a complete dashboard JSON.

NOTE: The form uses q1/q2/q3/q4 keys to represent the 4 Core R&D Activities listed above
(not calendar quarters). Map them accordingly in your output labels.

Return this EXACT JSON structure (fill every field with real values derived from the form):

{{
  "project_info": {{
    "title": "full project title from form",
    "company": "company name from form, else 'Applicant Company Pty Ltd'",
    "financial_year": "e.g. FY2024-25",
    "industry": "industry from form",
    "anzsic": "ANZSIC code and description if provided",
    "field_of_research": "ANZSRC field if provided",
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "duration_years": 11,
    "staff_members": ["Name 1", "Name 2"],
    "ip_owner": "company name",
    "generated_date": "{datetime.now().strftime('%d %B %Y')}"
  }},

  "kpi_cards": [
    {{"label": "Project Duration",     "value": "XX Years",  "icon": "clock"}},
    {{"label": "Total R&D Budget",     "value": "$X.XM",     "icon": "dollar"}},
    {{"label": "Core R&D Activities",  "value": "4",         "icon": "flask"}},
    {{"label": "Supporting Activities","value": "X",         "icon": "calendar"}},
    {{"label": "Est. Tax Offset",      "value": "$XXX,XXX",  "icon": "tax"}},
    {{"label": "ATO Compliance Score", "value": "X/10",      "icon": "check"}}
  ],

  "ato_compliance": {{
    "overall_score": 9,
    "eligible": true,
    "risk_level": "Low",
    "summary": "2-3 sentence overall eligibility assessment referencing s355-25 ITAA 1997",
    "meets_new_knowledge": true,
    "meets_uncertainty": true,
    "meets_systematic": true,
    "strengths": [
      "Specific strength citing actual content from the form",
      "Another strength with ATO reference"
    ],
    "issues": [
      "Any specific compliance gap — leave empty array [] if none"
    ],
    "suggestions": [
      "Specific improvement to strengthen claim or documentation"
    ],
    "audit_risk_flags": [
      "Any areas that might attract ATO scrutiny — leave empty array [] if none"
    ]
  }},

  "quarterly_summary": [
    {{
      "quarter": "use the activity title from q1_activity_title (shortened to ~40 chars)",
      "activities_summary": "2-sentence summary from q1_activities field",
      "hypothesis": "concise version of q1_hypothesis",
      "uncertainty": "core uncertainty from q1_uncertainty",
      "systematic_method": "how experiments were conducted from q1_systematic",
      "key_outcome": "main result from q1_outcomes",
      "new_knowledge": "key new knowledge from q1_new_knowledge",
      "eligibility_score": 9,
      "core_or_supporting": "Core",
      "compliance_notes": "specific ATO compliance observation for this activity"
    }},
    {{
      "quarter": "use q2_activity_title shortened",
      "activities_summary": "from q2_activities",
      "hypothesis": "from q2_hypothesis",
      "uncertainty": "from q2_uncertainty",
      "systematic_method": "from q2_systematic",
      "key_outcome": "from q2_outcomes",
      "new_knowledge": "from q2_new_knowledge",
      "eligibility_score": 9,
      "core_or_supporting": "Core",
      "compliance_notes": "ATO note"
    }},
    {{
      "quarter": "q3_activity_title shortened",
      "activities_summary": "q3 summary",
      "hypothesis": "q3 hypothesis",
      "uncertainty": "q3 uncertainty",
      "systematic_method": "q3 systematic",
      "key_outcome": "q3 outcomes",
      "new_knowledge": "q3 new knowledge",
      "eligibility_score": 8,
      "core_or_supporting": "Core",
      "compliance_notes": "ATO note"
    }},
    {{
      "quarter": "q4_activity_title shortened",
      "activities_summary": "q4 summary",
      "hypothesis": "q4 hypothesis",
      "uncertainty": "q4 uncertainty",
      "systematic_method": "q4 systematic",
      "key_outcome": "q4 outcomes",
      "new_knowledge": "q4 new knowledge",
      "eligibility_score": 8,
      "core_or_supporting": "Core",
      "compliance_notes": "ATO note"
    }}
  ],

  "expenditure": {{
    "total": 1800000,
    "staff_costs": 900000,
    "contractor_costs": 270000,
    "materials_consumables": 360000,
    "equipment_depreciation": 180000,
    "other_eligible_costs": 90000,
    "estimated_tax_offset": 783000,
    "core_percentage": 35,
    "supporting_percentage": 65,
    "annual_breakdown": [
      {{"year": "FY2016", "staff": 60000,  "materials": 15000, "equipment": 5000,  "external": 5000}},
      {{"year": "FY2017", "staff": 70000,  "materials": 20000, "equipment": 5000,  "external": 5000}},
      {{"year": "FY2018", "staff": 80000,  "materials": 25000, "equipment": 10000, "external": 5000}},
      {{"year": "FY2019", "staff": 90000,  "materials": 30000, "equipment": 10000, "external": 10000}},
      {{"year": "FY2020", "staff": 100000, "materials": 35000, "equipment": 15000, "external": 15000}},
      {{"year": "FY2021", "staff": 110000, "materials": 35000, "equipment": 15000, "external": 20000}},
      {{"year": "FY2022", "staff": 120000, "materials": 40000, "equipment": 20000, "external": 20000}},
      {{"year": "FY2023", "staff": 130000, "materials": 45000, "equipment": 20000, "external": 25000}},
      {{"year": "FY2024", "staff": 140000, "materials": 55000, "equipment": 25000, "external": 30000}},
      {{"year": "FY2025", "staff": 150000, "materials": 60000, "equipment": 30000, "external": 35000}}
    ]
  }},

  "technical_progress": [
    {{
      "challenge": "MRD Shield Material (Silicone)",
      "progress_pct": 80,
      "resolved_items": ["✓ TPE limitations fully characterised", "✓ Silicone superior performance confirmed", "✓ Patient self-customisation validated"],
      "pending_items": ["⚠ Multi-site long-term clinical validation (FY26)"]
    }},
    {{
      "challenge": "Polyolefin Fin System",
      "progress_pct": 75,
      "resolved_items": ["✓ Cross-pin lock mechanism validated", "✓ 36hr bruxism stress test passed"],
      "pending_items": ["⚠ Side-wall fracture line reinforcement under investigation"]
    }},
    {{
      "challenge": "EMG RMMA Module",
      "progress_pct": 55,
      "resolved_items": ["✓ ANR Corp M40 selected as optimal hardware", "✓ Temporalis exclusion reduces false positives"],
      "pending_items": ["⚠ Python algorithm insufficient — AI platform pivot required", "⚠ Philips Respironics discontinuation — hardware pathway closed"]
    }},
    {{
      "challenge": "OTS MAS Device",
      "progress_pct": 65,
      "resolved_items": ["✓ Injection moulded MVP produced", "✓ Boil-and-bite aligner compatibility confirmed"],
      "pending_items": ["⚠ EVA lining detachment issue — mechanical retention redesign in progress"]
    }}
  ],

  "innovations": [
    {{
      "category": "Material Science",
      "finding": "Tongue shield integration with MRD provides unexpected dual therapeutic role supporting myofunctional therapy (MFT) — first documented evidence",
      "impact": "High"
    }},
    {{
      "category": "Material Science",
      "finding": "Thermoformable silicone demonstrated self-customisation capability over diverse oral anatomies without lab fabrication",
      "impact": "High"
    }},
    {{
      "category": "Device Engineering",
      "finding": "Cross-pin lock mechanism enables secure polyolefin fin retention under 10x worst-case bruxism force cycles (36-hour jig test)",
      "impact": "High"
    }},
    {{
      "category": "Device Engineering",
      "finding": "Colour-coded fins (left/right differentiation) reduce patient orientation errors and improve self-fitting compliance",
      "impact": "Medium"
    }},
    {{
      "category": "Clinical / Diagnostic",
      "finding": "Masseter-only EMG recording eliminates temporalis false positives caused by scallop-shaped muscle fibres — novel finding for sleep diagnostics",
      "impact": "High"
    }},
    {{
      "category": "Clinical / OTS MAS",
      "finding": "Boil-and-bite MAS approach provides sufficient adaptability for concurrent sequential orthodontic aligner therapy — first documented validation",
      "impact": "High"
    }},
    {{
      "category": "Clinical / OTS MAS",
      "finding": "Adjustment fins influence broader mode of device fitment beyond mandibular protrusion, altering tooth-device interactions in previously undocumented ways",
      "impact": "Medium"
    }}
  ],

  "recommendations": [
    {{
      "priority": "High",
      "action": "Complete multi-site structured clinical trials for silicone shield in FY26 to generate statistically significant patient outcomes data for ATO documentation and regulatory submission"
    }},
    {{
      "priority": "High",
      "action": "Document fin side-wall reinforcement experiments in detail as a new systematic progression — hypothesis, test method, results — to satisfy s355-25 for next registration period"
    }},
    {{
      "priority": "High",
      "action": "Formalise AI-based EMG platform pivot as a new core R&D activity with explicit hypothesis and uncertainty documentation; current Python algorithm failure is strong evidence of genuine uncertainty"
    }},
    {{
      "priority": "Medium",
      "action": "Strengthen EVA-shell mechanical retention experiments with quantified adhesion testing data (N/mm² bond strength before/after cycling) to satisfy ATO experimental evaluation requirements"
    }},
    {{
      "priority": "Medium",
      "action": "Ensure all FY25 staff time sheets clearly differentiate R&D vs BAU activities across all 4 core activities to support cost apportionment in the event of ATO review"
    }},
    {{
      "priority": "Low",
      "action": "Consider provisional patent filing for tongue shield MFT dual-therapy finding and boil-and-bite aligner compatibility — both are novel, commercially valuable, and currently undocumented in literature"
    }}
  ]
}}

CRITICAL RULES:
- estimated_tax_offset = total * 0.435 (43.5% refundable offset for companies with turnover < $20M)
- core_percentage and supporting_percentage should reflect the effort % from the timeline table in the form
- Use ACTUAL content from the form fields — do not fabricate generic statements
- activity labels in quarterly_summary should use the real activity titles from q1_activity_title etc.
- annual_breakdown: derive realistic year-by-year split from project start (FY2016) to FY2025 summing to total
- Be specific in compliance notes — cite actual ATO requirements (s355-25, IR&D Act, genuine uncertainty test etc.)
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ATO_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.15,
        response_format={"type": "json_object"},
        max_tokens=4000
    )

    return json.loads(response.choices[0].message.content)


# =============================================================================
# STEP 2 — CHART GENERATORS (matplotlib → ReportLab Image)
# =============================================================================

def _fig_to_image(fig, width_mm: float, height_mm: float) -> Image:
    """Convert matplotlib figure to a ReportLab Image via in-memory PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_mm * mm, height=height_mm * mm)


def make_kpi_cards(kpi_data: list) -> Image:
    """6 KPI summary cards."""
    fig, axes = plt.subplots(1, 6, figsize=(18, 2.8))
    fig.patch.set_facecolor(LIGHT_GREY)
    icon_map = {
        "clock": "⏱", "dollar": "$", "people": "👥",
        "calendar": "📅", "tax": "📋", "check": "✓"
    }
    for ax, kpi in zip(axes, kpi_data):
        ax.set_facecolor(WHITE)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(ACCENT)
            spine.set_linewidth(1.5)
        ax.text(0.5, 0.82, icon_map.get(kpi.get("icon", ""), "●"),
                ha="center", va="center", fontsize=18, color=ACCENT)
        ax.text(0.5, 0.50, kpi["value"],
                ha="center", va="center", fontsize=13,
                fontweight="bold", color=NAVY)
        label = kpi["label"]
        ax.text(0.5, 0.18, label, ha="center", va="center",
                fontsize=7.5, color=BLUE, wrap=True)
    fig.tight_layout(pad=0.8)
    return _fig_to_image(fig, width_mm=170, height_mm=50)


def make_expenditure_pie(exp: dict) -> Image:
    """Expenditure breakdown pie chart."""
    labels = ["Staff", "Contractors", "Materials", "Equipment", "Other"]
    values = [
        exp.get("staff_costs", 0),
        exp.get("contractor_costs", 0),
        exp.get("materials_consumables", 0),
        exp.get("equipment_depreciation", 0),
        exp.get("other_eligible_costs", 0),
    ]
    chart_colors = [NAVY, BLUE, ACCENT, TEAL, ORANGE]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor(WHITE)
    wedges, _, autotexts = ax.pie(
        values, colors=chart_colors, autopct="%1.0f%%",
        startangle=140, pctdistance=0.72,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontsize(8); t.set_color("white"); t.set_fontweight("bold")
    ax.legend(wedges, labels, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.12), fontsize=7.5, frameon=False)
    ax.set_title("Expenditure Breakdown", fontsize=10,
                 fontweight="bold", color=NAVY, pad=8)
    fig.tight_layout()
    return _fig_to_image(fig, width_mm=82, height_mm=78)


def make_quarterly_scores(quarterly: list) -> Image:
    """Bar chart of quarterly ATO eligibility scores."""
    quarters = [q["quarter"] for q in quarterly]
    scores   = [q.get("eligibility_score", 0) for q in quarterly]
    # Wrap long activity names so they don't overlap on the x-axis
    wrapped_labels = [textwrap.fill(q, width=18) for q in quarters]
    x_pos = range(len(quarters))
    fig, ax  = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LIGHT_GREY)
    bar_colors = [TEAL if s >= 7 else (ORANGE if s >= 5 else "#E74C3C") for s in scores]
    bars = ax.bar(x_pos, scores, color=bar_colors, width=0.45,
                  edgecolor="white", linewidth=1.5)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(wrapped_labels, fontsize=6.5, ha="center")
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("Score / 10", fontsize=8, color=NAVY)
    ax.set_title("Quarterly ATO Eligibility Scores", fontsize=10,
                 fontweight="bold", color=NAVY)
    ax.axhline(7, color=ORANGE, linestyle="--", linewidth=1.2, label="Min. recommended (7)")
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15, f"{score}/10",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=NAVY)
    fig.tight_layout()
    return _fig_to_image(fig, width_mm=100, height_mm=80)


def make_progress_bars(challenges: list) -> Image:
    """Horizontal progress bars for technical challenges."""
    n = max(len(challenges), 1)
    fig, ax = plt.subplots(figsize=(10, max(2.5, n * 1.1)))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    for i, ch in enumerate(challenges):
        pct = ch.get("progress_pct", 0) / 100
        y   = n - 1 - i
        ax.barh(y, 1.0, height=0.5, color=LIGHT_GREY,
                edgecolor=MID_GREY, linewidth=0.5)
        bc = TEAL if pct > 0.7 else (ACCENT if pct > 0.4 else ORANGE)
        ax.barh(y, pct, height=0.5, color=bc)
        ax.text(-0.02, y, ch["challenge"], ha="right", va="center",
                fontsize=8, color=NAVY, fontweight="bold")
        ax.text(pct + 0.01, y, f"{ch['progress_pct']}%",
                ha="left", va="center", fontsize=8, color=NAVY)
    ax.set_xlim(-0.38, 1.15)
    ax.set_ylim(-0.8, n - 0.2)
    ax.axis("off")
    ax.set_title("Technical Challenge Resolution Progress",
                 fontsize=10, fontweight="bold", color=NAVY)
    fig.tight_layout()
    return _fig_to_image(fig, width_mm=165, height_mm=max(45, n * 18))


def make_annual_expenditure(annual: list) -> Image:
    """Stacked bar chart for annual expenditure."""
    if not annual:
        return None
    years    = [d["year"]                  for d in annual]
    staff    = [d.get("staff",     0)      for d in annual]
    mats     = [d.get("materials", 0)      for d in annual]
    equip    = [d.get("equipment", 0)      for d in annual]
    ext      = [d.get("external",  0)      for d in annual]
    x = np.arange(len(years))
    w = 0.5

    fig, ax = plt.subplots(figsize=(max(7, len(years) * 1.1), 4))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LIGHT_GREY)
    ax.bar(x, staff, w, label="Staff",     color=NAVY)
    ax.bar(x, mats,  w, label="Materials", color=ACCENT,
           bottom=np.array(staff))
    ax.bar(x, equip, w, label="Equipment", color=TEAL,
           bottom=np.array(staff) + np.array(mats))
    ax.bar(x, ext,   w, label="External",  color=ORANGE,
           bottom=np.array(staff) + np.array(mats) + np.array(equip))
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
    ax.set_title("Annual R&D Expenditure Breakdown", fontsize=10,
                 fontweight="bold", color=NAVY)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _fig_to_image(fig, width_mm=165, height_mm=68)


# =============================================================================
# STEP 3 — PDF BUILDER
# =============================================================================

def build_pdf(data: dict, output_path: str) -> str:
    """Render the complete dashboard PDF from structured data dict."""

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=14*mm, leftMargin=14*mm,
        topMargin=18*mm,   bottomMargin=18*mm
    )
    styles = getSampleStyleSheet()

    def S(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=styles[parent], **kw)

    H1   = S("H1",  "Heading1", fontSize=17, textColor=RL_NAVY,
              fontName="Helvetica-Bold", spaceAfter=3, spaceBefore=6)
    H2   = S("H2",  "Heading2", fontSize=12, textColor=RL_BLUE,
              fontName="Helvetica-Bold", spaceAfter=3, spaceBefore=10)
    H3   = S("H3",  "Normal",   fontSize=9,  textColor=RL_ACCENT,
              fontName="Helvetica-Bold", spaceAfter=2, spaceBefore=5)
    BODY = S("BODY","Normal",   fontSize=8.5,
              textColor=colors.HexColor("#2C3E50"), spaceAfter=3, leading=13)
    SMALL= S("SM",  "Normal",   fontSize=7.5,
              textColor=colors.HexColor("#2C3E50"), spaceAfter=2, leading=11)
    FOOT = S("FT",  "Normal",   fontSize=7,
              textColor=colors.HexColor(MID_GREY), alignment=TA_CENTER)

    info = data.get("project_info", {})
    story = []

    # ── HEADER BANNER ────────────────────────────────────────────────────────
    hdr = Table([[
        Paragraph("<b>PATTENS</b><br/><font size='7'>Tax Incentive Claims</font>",
                  S("bn", fontSize=13, textColor=colors.white,
                    fontName="Helvetica-Bold")),
        Paragraph("<b>R&amp;D Tax Incentive — Dashboard Report</b>",
                  S("rt", fontSize=14, textColor=colors.white,
                    fontName="Helvetica-Bold", alignment=TA_RIGHT))
    ]], colWidths=[88*mm, 88*mm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), RL_NAVY),
        ("TOPPADDING",    (0,0),(-1,-1), 11),
        ("BOTTOMPADDING", (0,0),(-1,-1), 11),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story += [hdr, Spacer(1, 5*mm)]

    # ── PROJECT TITLE + META ─────────────────────────────────────────────────
    story.append(Paragraph(info.get("title", "R&D Project"), H1))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=RL_ACCENT, spaceAfter=4))

    meta = Table([
        [Paragraph(f"<b>Financial Year:</b> {info.get('financial_year','—')}", BODY),
         Paragraph(f"<b>Industry:</b> {info.get('industry','—')}", BODY),
         Paragraph(f"<b>Generated:</b> {info.get('generated_date','—')}", BODY)],
        [Paragraph(f"<b>Period:</b> {info.get('start_date','—')} → {info.get('end_date','—')}", BODY),
         Paragraph(f"<b>Duration:</b> {info.get('duration_years','—')} yrs", BODY),
         Paragraph(f"<b>Team:</b> {', '.join(info.get('staff_members', []))}", BODY)],
    ], colWidths=[59*mm, 59*mm, 59*mm])
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), RL_LGREY),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor(MID_GREY)),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ]))
    story += [meta, Spacer(1, 5*mm)]

    # ── KPI CARDS ────────────────────────────────────────────────────────────
    story.append(Paragraph("Key Performance Indicators", H2))
    story.append(make_kpi_cards(data.get("kpi_cards", [])))
    story.append(Spacer(1, 5*mm))

    # ── EXPENDITURE PIE + QUARTERLY SCORES (side by side) ────────────────────
    story.append(Paragraph("Financial Overview & Quarterly Compliance Scores", H2))
    exp   = data.get("expenditure", {})
    total = exp.get("total", 0)
    offset= exp.get("estimated_tax_offset", round(total * 0.435))

    side_by_side = Table(
        [[make_expenditure_pie(exp), make_quarterly_scores(data.get("quarterly_summary", []))]],
        colWidths=[80*mm, 102*mm]
    )
    side_by_side.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0),(-1,-1), 2),
        ("RIGHTPADDING", (0,0),(-1,-1), 2),
    ]))
    story.append(side_by_side)
    story.append(Spacer(1, 3*mm))

    # Financial summary table — all cells wrapped in Paragraph for proper text flow
    def _fp(txt, bold=False):
        t = f"<b>{txt}</b>" if bold else str(txt)
        return Paragraph(t, S(f"fp{txt[:8]}", fontSize=8,
                               textColor=colors.HexColor("#2C3E50"), leading=12))

    fin_rows = [
        [Paragraph("<b>Expenditure Category</b>", S("fh1", fontSize=8, textColor=colors.white, fontName="Helvetica-Bold")),
         Paragraph("<b>Amount (AUD)</b>",          S("fh2", fontSize=8, textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_RIGHT)),
         Paragraph("<b>% of Total</b>",            S("fh3", fontSize=8, textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_RIGHT))],
        [_fp("Staff Costs"),           _fp(f"${exp.get('staff_costs',0):,.0f}"),           _fp(f"{exp.get('staff_costs',0)/max(total,1)*100:.0f}%")],
        [_fp("Contractor Costs"),      _fp(f"${exp.get('contractor_costs',0):,.0f}"),      _fp(f"{exp.get('contractor_costs',0)/max(total,1)*100:.0f}%")],
        [_fp("Materials & Consumables"),_fp(f"${exp.get('materials_consumables',0):,.0f}"),_fp(f"{exp.get('materials_consumables',0)/max(total,1)*100:.0f}%")],
        [_fp("Equipment & Depreciation"),_fp(f"${exp.get('equipment_depreciation',0):,.0f}"),_fp(f"{exp.get('equipment_depreciation',0)/max(total,1)*100:.0f}%")],
        [_fp("Other Eligible Costs"),  _fp(f"${exp.get('other_eligible_costs',0):,.0f}"),  _fp(f"{exp.get('other_eligible_costs',0)/max(total,1)*100:.0f}%")],
        [_fp("TOTAL R&D EXPENDITURE", bold=True), _fp(f"${total:,.0f}", bold=True),        _fp("100%", bold=True)],
        [_fp("Estimated Tax Offset (43.5%)", bold=True), _fp(f"${offset:,.0f}", bold=True),_fp("—")],
    ]
    ft = Table(fin_rows, colWidths=[100*mm, 52*mm, 30*mm])
    ft.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  RL_NAVY),
        ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 8),
        ("BACKGROUND",     (0,-2),(-1,-2), colors.HexColor("#D6EAF8")),
        ("FONTNAME",       (0,-2),(-1,-2), "Helvetica-Bold"),
        ("BACKGROUND",     (0,-1),(-1,-1), colors.HexColor("#D5F5E3")),
        ("FONTNAME",       (0,-1),(-1,-1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,1), (-1,-3), [colors.white, RL_LGREY]),
        ("GRID",           (0,0), (-1,-1), 0.4, colors.HexColor(MID_GREY)),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
        ("ALIGN",          (1,0), (-1,-1), "RIGHT"),
    ]))
    story += [ft, Spacer(1, 5*mm)]

    # ── QUARTERLY ACTIVITIES ──────────────────────────────────────────────────
    story.append(Paragraph("Core R&D Activities — Quarterly Detail", H2))
    for q in data.get("quarterly_summary", []):
        qtr   = q.get("quarter", "Q?")
        score = q.get("eligibility_score", 0)
        c_or_s= q.get("core_or_supporting", "Core")
        sc    = TEAL if score >= 7 else (ORANGE if score >= 5 else "#E74C3C")

        # Quarter header row — total width = 182mm (A4 180mm usable)
        qhdr = Table([[
            Paragraph(f"<b>{qtr} — {c_or_s} R&D Activity</b>",
                      S(f"qh{qtr}", fontSize=9.5, textColor=colors.white,
                        fontName="Helvetica-Bold", leading=13)),
            Paragraph(f"Eligibility Score: <b>{score}/10</b>",
                      S(f"qs{qtr}", fontSize=9, textColor=colors.white,
                        alignment=TA_RIGHT))
        ]], colWidths=[130*mm, 52*mm])
        qhdr.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), RL_BLUE),
            ("TOPPADDING",    (0,0),(-1,-1), 6),
            ("BOTTOMPADDING", (0,0),(-1,-1), 6),
            ("LEFTPADDING",   (0,0),(0,0),   8),
            ("RIGHTPADDING",  (-1,0),(-1,0), 8),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        story.append(qhdr)

        q_body = [
            ["R&D Activities",     Paragraph(q.get("activities_summary", "—"), SMALL)],
            ["Hypothesis Tested",  Paragraph(q.get("hypothesis",         "—"), SMALL)],
            ["Uncertainty",        Paragraph(q.get("uncertainty",        "—"), SMALL)],
            ["Systematic Method",  Paragraph(q.get("systematic_method",  "—"), SMALL)],
            ["Key Outcome",        Paragraph(q.get("key_outcome",        "—"), SMALL)],
            ["New Knowledge",      Paragraph(q.get("new_knowledge",      "—"), SMALL)],
        ]
        if q.get("compliance_notes"):
            q_body.append(["ATO Notes", Paragraph(q["compliance_notes"], SMALL)])

        qt = Table(q_body, colWidths=[40*mm, 142*mm])
        qt.setStyle(TableStyle([
            ("FONTNAME",       (0,0),(0,-1), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1), 8),
            ("TEXTCOLOR",      (0,0),(0,-1), RL_BLUE),
            ("ROWBACKGROUNDS", (0,0),(-1,-1), [colors.white, RL_LGREY]),
            ("GRID",           (0,0),(-1,-1), 0.3, colors.HexColor(MID_GREY)),
            ("TOPPADDING",     (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
            ("LEFTPADDING",    (0,0),(-1,-1), 8),
            ("RIGHTPADDING",   (0,0),(-1,-1), 6),
            ("VALIGN",         (0,0),(-1,-1), "TOP"),
        ]))
        story += [qt, Spacer(1, 4*mm)]

    # ── TECHNICAL PROGRESS ────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Technical Challenge Resolution", H2))
    challenges = data.get("technical_progress", [])
    if challenges:
        story.append(make_progress_bars(challenges))
        story.append(Spacer(1, 3*mm))

        prog_rows = [
            [Paragraph("<b>Challenge</b>",     S("ph1", fontSize=7.5, textColor=colors.white, fontName="Helvetica-Bold")),
             Paragraph("<b>Progress</b>",      S("ph2", fontSize=7.5, textColor=colors.white, fontName="Helvetica-Bold")),
             Paragraph("<b>Resolved Items</b>",S("ph3", fontSize=7.5, textColor=colors.white, fontName="Helvetica-Bold")),
             Paragraph("<b>Pending Items</b>", S("ph4", fontSize=7.5, textColor=colors.white, fontName="Helvetica-Bold"))]
        ]
        for ch in challenges:
            prog_rows.append([
                Paragraph(f"<b>{ch['challenge']}</b>", SMALL),
                Paragraph(f"<b>{ch['progress_pct']}%</b>", SMALL),
                Paragraph("<br/>".join(ch.get("resolved_items", [])), SMALL),
                Paragraph("<br/>".join(ch.get("pending_items",  [])), SMALL),
            ])
        pt = Table(prog_rows, colWidths=[48*mm, 16*mm, 60*mm, 58*mm])
        pt.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0),  RL_NAVY),
            ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
            ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1), 7.5),
            ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, RL_LGREY]),
            ("GRID",           (0,0),(-1,-1), 0.3, colors.HexColor(MID_GREY)),
            ("TOPPADDING",     (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 4),
            ("LEFTPADDING",    (0,0),(-1,-1), 6),
            ("VALIGN",         (0,0),(-1,-1), "TOP"),
        ]))
        story += [pt, Spacer(1, 5*mm)]

    # ── ANNUAL EXPENDITURE CHART ──────────────────────────────────────────────
    annual = exp.get("annual_breakdown", [])
    if annual:
        story.append(Paragraph("Annual Expenditure Breakdown", H2))
        ann_img = make_annual_expenditure(annual)
        if ann_img:
            story += [ann_img, Spacer(1, 5*mm)]

    # ── KEY INNOVATIONS ───────────────────────────────────────────────────────
    story.append(Paragraph("Key Innovations & New Knowledge Generated", H2))
    innovations = data.get("innovations", [])
    if innovations:
        innov_rows = [
            [Paragraph("Category", S("ih1", fontSize=8, textColor=colors.white, fontName="Helvetica-Bold")),
             Paragraph("Innovation / Knowledge Generated", S("ih2", fontSize=8, textColor=colors.white, fontName="Helvetica-Bold")),
             Paragraph("Impact", S("ih3", fontSize=8, textColor=colors.white, fontName="Helvetica-Bold"))]
        ]
        for iv in innovations:
            innov_rows.append([
                Paragraph(f"<b>{iv.get('category','—')}</b>", SMALL),
                Paragraph(iv.get("finding", "—"), SMALL),
                Paragraph(f"<b>{iv.get('impact','—')}</b>", SMALL),
            ])
        it = Table(innov_rows, colWidths=[40*mm, 128*mm, 14*mm])
        it.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0),  RL_NAVY),
            ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
            ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, RL_LGREY]),
            ("GRID",           (0,0),(-1,-1), 0.3, colors.HexColor(MID_GREY)),
            ("TOPPADDING",     (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
            ("LEFTPADDING",    (0,0),(-1,-1), 8),
            ("VALIGN",         (0,0),(-1,-1), "TOP"),
        ]))
        story += [it, Spacer(1, 5*mm)]

    # ── ATO COMPLIANCE ASSESSMENT ─────────────────────────────────────────────
    story.append(Paragraph("ATO Compliance Assessment", H2))
    ato = data.get("ato_compliance", {})
    risk_bg = {"Low": "#D5F5E3", "Medium": "#FEF9E7", "High": "#FDEDEC"}
    eligible = ato.get("eligible", True)
    risk     = ato.get("risk_level", "Low")
    score_v  = ato.get("overall_score", "—")

    ct = Table([
        ["Eligible for R&D Tax Incentive", "ATO Audit Risk Level", "Overall Compliance Score"],
        ["✓ Yes" if eligible else "✗ No",  risk,                   f"{score_v}/10"]
    ], colWidths=[59*mm, 59*mm, 59*mm])
    ct.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  RL_NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("FONTNAME",      (0,1),(-1,1),  "Helvetica-Bold"),
        ("BACKGROUND",    (0,1),(0,1),   colors.HexColor("#D5F5E3" if eligible else "#FDEDEC")),
        ("BACKGROUND",    (1,1),(1,1),   colors.HexColor(risk_bg.get(risk, "#FEF9E7"))),
        ("BACKGROUND",    (2,1),(2,1),   colors.HexColor("#D6EAF8")),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor(MID_GREY)),
    ]))
    story += [ct, Spacer(1, 3*mm)]
    story.append(Paragraph(f"<b>Assessment:</b> {ato.get('summary','—')}", BODY))

    # Three tests checklist
    tests = [
        ("New Knowledge Test",      ato.get("meets_new_knowledge", False)),
        ("Genuine Uncertainty Test", ato.get("meets_uncertainty",   False)),
        ("Systematic Progression",  ato.get("meets_systematic",    False)),
    ]
    test_rows = [
        [Paragraph("<b>ATO Eligibility Test</b>", S("th1", fontSize=8, textColor=colors.white, fontName="Helvetica-Bold")),
         Paragraph("<b>Status</b>",               S("th2", fontSize=8, textColor=colors.white, fontName="Helvetica-Bold"))]
    ]
    for test_name, passed in tests:
        test_rows.append([
            Paragraph(test_name, S(f"tn{test_name[:6]}", fontSize=8,
                                   textColor=colors.HexColor("#2C3E50"), leading=12)),
            Paragraph("<b>✓ Met</b>" if passed else "<b>✗ Not Met</b>",
                      S(f"ts{test_name[:6]}", fontSize=8,
                        textColor=colors.HexColor(TEAL if passed else "#E74C3C"),
                        fontName="Helvetica-Bold", alignment=TA_CENTER))
        ])
    tt = Table(test_rows, colWidths=[148*mm, 34*mm])
    tt.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0),  RL_BLUE),
        ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
        ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, RL_LGREY]),
        ("GRID",           (0,0),(-1,-1), 0.3, colors.HexColor(MID_GREY)),
        ("TOPPADDING",     (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
        ("LEFTPADDING",    (0,0),(-1,-1), 8),
        ("ALIGN",          (1,0),(-1,-1), "CENTER"),
    ]))
    story += [Spacer(1, 3*mm), tt, Spacer(1, 3*mm)]

    for section, key, color in [
        ("Strengths",                    "strengths",         TEAL),
        ("Compliance Issues",            "issues",            ORANGE),
        ("Recommendations to Strengthen","suggestions",       ACCENT),
        ("Audit Risk Flags",             "audit_risk_flags",  "#E74C3C"),
    ]:
        items = ato.get(key, [])
        if items:
            story.append(Paragraph(f"<b>{section}:</b>", H3))
            for item in items:
                prefix = "✓" if key == "strengths" else ("⚠" if key == "issues" else "→")
                story.append(Paragraph(
                    f'<font color="{color}"><b>{prefix}</b></font> {item}', BODY))

    story.append(Spacer(1, 5*mm))

    # ── ACTION ITEMS ──────────────────────────────────────────────────────────
    story.append(Paragraph("Action Items & Next Steps", H2))
    recs = data.get("recommendations", [])
    if recs:
        rec_rows = [
            [Paragraph("<b>Priority</b>",           S("rh1", fontSize=8, textColor=colors.white, fontName="Helvetica-Bold")),
             Paragraph("<b>Recommended Action</b>", S("rh2", fontSize=8, textColor=colors.white, fontName="Helvetica-Bold"))]
        ]
        for rec in recs:
            prio = rec.get("priority", "Medium")
            pc   = "#E74C3C" if prio=="High" else (ORANGE if prio=="Medium" else TEAL)
            rec_rows.append([
                Paragraph(f'<font color="{pc}"><b>{prio}</b></font>',
                          S(f"rp{prio}", fontSize=8, leading=12, alignment=TA_CENTER)),
                Paragraph(rec.get("action", "—"),
                          S(f"ra{prio}", fontSize=8, textColor=colors.HexColor("#2C3E50"), leading=12)),
            ])
        rt = Table(rec_rows, colWidths=[22*mm, 160*mm])
        rt.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0),  RL_NAVY),
            ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
            ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, RL_LGREY]),
            ("GRID",           (0,0),(-1,-1), 0.3, colors.HexColor(MID_GREY)),
            ("TOPPADDING",     (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
            ("LEFTPADDING",    (0,0),(-1,-1), 8),
            ("VALIGN",         (0,0),(-1,-1), "TOP"),
        ]))
        story.append(rt)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 8*mm),
        HRFlowable(width="100%", thickness=1, color=RL_LGREY),
        Paragraph(
            f"Generated by Pattens Tax Incentive Claims Platform  |  "
            f"Confidential & For Client Use Only  |  {info.get('generated_date','')}",
            FOOT
        )
    ]

    doc.build(story)
    return output_path


# =============================================================================
# MAIN ENTRY POINT — call this from your Django view
# =============================================================================

def generate_rd_dashboard_pdf(
    form_data: dict,
    openai_api_key: str,
    output_path: str = None,
) -> str:
    """
    Full pipeline: form_data → GPT-4o → PDF dashboard.

    Args:
        form_data:       All 7 form sections as a flat dict
        openai_api_key:  Your OpenAI API key (from settings.OPENAI_API_KEY)
        output_path:     Where to save the PDF (default: /tmp/<uuid>.pdf)

    Returns:
        Absolute path to generated PDF (str)

    DJANGO VIEW EXAMPLE:
        from .rd_ai_pipeline import generate_rd_dashboard_pdf
        from django.http import FileResponse
        from django.conf import settings

        def export_pdf(request):
            form_data = request.POST.dict()
            pdf_path  = generate_rd_dashboard_pdf(
                form_data=form_data,
                openai_api_key=settings.OPENAI_API_KEY,
            )
            return FileResponse(
                open(pdf_path, 'rb'),
                content_type='application/pdf',
                as_attachment=True,
                filename='RD_Dashboard.pdf'
            )
    """
    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(), f"rd_dashboard_{uuid.uuid4().hex}.pdf"
        )

    print("[1/2] Generating dashboard data with GPT-4o...")
    dashboard_data = generate_dashboard_data(form_data, openai_api_key)

    print("[2/2] Building PDF...")
    build_pdf(dashboard_data, output_path)

    print(f"[✓] PDF ready: {output_path}")
    return output_path


# =============================================================================
# CLI TEST — python rd_ai_pipeline.py
# Real data from Company 2 Pty Ltd — FY25 R&D Plan (Mandibular Repositioning Device)
# =============================================================================
if __name__ == "__main__":
    SAMPLE = {
        # ── Section 1: Project Details ────────────────────────────────────────
        "project_title":       "Mandibular Repositioning Device [System for the identification, reporting, communication, and treatment of OSA]",
        "company_name":        "Company 2 Pty Ltd",
        "brief_summary":       (
            "Development of a myofunctional therapy (MRT) system for Obstructive Sleep Apnoea (OSA) "
            "comprising: a novel MRD customisable directly in a patient's mouth; a one-size-fits-all "
            "off-the-shelf mandibular advancement splint (MAS) compatible with orthodontic aligners; "
            "a portable EMG headset for home-based RMMA sleep monitoring; and an online platform "
            "integrating patient data collection, scheduling, and clinical reporting."
        ),
        "financial_year":      "FY2024-25",
        "project_start_date":  "2015-07-01",
        "project_end_date":    "2026-06-30",
        "industry":            "ANZSIC Division Q — Health Care and Social Assistance / ANZSIC Class 8531 Dental Services",
        "field_of_research":   "ANZSRC Division 42 — Health Sciences / Group 4299 Other Health Sciences",
        "staff_members":       "Christopher Kelly (Director / Lead Researcher)",
        "overseas_work":       "No",
        "rsp_involvement":     "No",

        # ── Section 2: Core Activity 1.1 — MRD Shield Development ────────────
        "q1_activity_title":   "Core R&D Activity 1.1 – Development of a mandibular repositioning device",
        "q1_activities":       (
            "Development of a novel MRD with redesigned intraoral guard incorporating elastic forces "
            "to replicate mastication muscle function, reducing hypopharyngeal airway obstruction. "
            "Iterative trials of Aquaplast, TPE, and thermoformable silicone shields. Modifications "
            "included blocking suction holes, removing tongue suction extensions, altering labial and "
            "tooth support areas. In FY25 patient testing continues with silicone shields guided by "
            "comfort, durability, and ease-of-use feedback."
        ),
        "q1_hypothesis":       (
            "A novel MRD incorporating an off-the-shelf thermoformable silicone intraoral shield with "
            "elastic properties can replicate mastication muscle support during sleep, reduce "
            "hypopharyngeal airway obstruction, and improve patient comfort vs existing TPE designs. "
            "Silicone can be clinically fitted and self-adjusted by patients, enabling universal "
            "off-the-shelf use with personalised outcomes. Combined tongue shield integration will "
            "provide dual benefit: airway maintenance and myofunctional retraining."
        ),
        "q1_uncertainty":      (
            "No existing commercial solution addresses tongue posture training within an MRD. TPE "
            "shields (e.g. Somnics iNAP) were unsuitable due to palate space restriction and poor "
            "durability. Silicone offers elasticity and hydrophobic properties but has unknown long-term "
            "dimensional stability under intraoral conditions (moisture, pH, enzymes, cyclic loading). "
            "Whether patients can achieve effective in-situ self-adjustment without compromising fit "
            "or seal could not be determined without experimentation."
        ),
        "q1_systematic":       (
            "Iterative experimental activities tested: silicone material ease of use; lip seal and vacuum "
            "potential without tissue damage; optimal thickness for comfort and function; patient "
            "acceptance in cohorts; medium-term adoption rates at 2-week follow-up sleep studies; "
            "fit feedback around custom and OTS MAS components; bacterial tolerance via cleaning "
            "product trials; plaque build-up observation; time-based efficacy of cleaning products. "
            "Results fed back iteratively to modify shield design for FY25 continued trials."
        ),
        "q1_outcomes":         (
            "Silicone proven superior to TPE: low water absorption, resistance to clouding, stability "
            "against elastic deformation breakdown. Redesigned shield substantially replicates "
            "mastication muscle support, contributing to reductions in sleep apnoea symptoms. "
            "Unexpected discovery: tongue shield integration supports myofunctional therapy (MFT) — "
            "an unanticipated clinical benefit. Patients can self-customise using company guides. "
            "Structured multi-site clinical validation trials continuing into FY26."
        ),
        "q1_new_knowledge":    (
            "Established that thermoformable silicone can achieve universal-size, off-the-shelf shield "
            "solution adapting to diverse oral anatomies. Confirmed material selection cannot rely on "
            "bulk mechanical properties alone — must be validated in situ under clinical conditions "
            "(prior TPE failure demonstrated this). Identified that tongue shield integration with MRD "
            "provides unexpected dual therapeutic role supporting MFT — previously undocumented "
            "in literature. Demonstrated patient self-customisation feasibility."
        ),

        # ── Section 3: Core Activity 1.2 — Polyolefin Fin Redesign ──────────
        "q2_activity_title":   "Core R&D Activity 1.2 – Redesign fins from nylon to injection moulding polyolefin fins",
        "q2_activities":       (
            "Redesign of MRD fins from nylon prototypes to injection-moulded polyolefin to address "
            "durability, retention, and wear resistance limitations. Nylon fins showed excessive "
            "friction-induced wear and unreliable retention across offset/undercut geometries. "
            "Polyolefin fins produced with draft angles, vent holes, and structural modifications for "
            "injection moulding process. Cross-pin lock mechanism developed. Colour differentiation "
            "(left/right) trialled. In FY25: 36-hour bruxism force simulation jig testing conducted; "
            "fin side-wall fracture line identified; reinforcement design work underway."
        ),
        "q2_hypothesis":       (
            "If MRD spine/support mechanism is redesigned to securely anchor injection-moulded "
            "polyolefin fins, these fins can replace silicone and nylon prototypes while maintaining "
            "intraoral stability, patient comfort, and clinical effectiveness. Polyolefin will deliver "
            "greater wear resistance and fatigue performance, reduce friction-induced wear, extend "
            "component lifespan to match MRD body, and reduce premature failures. Colour-coded "
            "fins will reduce patient orientation errors and improve compliance."
        ),
        "q2_uncertainty":      (
            "No published literature or patents address polyolefin fins for MRDs. Somnomed's MAS G2 "
            "clip-on fin attempt failed clinically (patients swallowing fins) and was abandoned — "
            "confirming no usable competitor precedent. Whether polyolefin could withstand repetitive "
            "occlusal/bruxism forces, resist salivary degradation, maintain locking interface across "
            "repeated insertion cycles, and align lifespan with MRD body was entirely unknown prior "
            "to experimentation."
        ),
        "q2_systematic":       (
            "CAD designs developed for optimal fin/spine combinations across range of undercut values, "
            "cross-sectional strengths, and offsets. Iterative experiments tested fin design, spine "
            "design, and fin slip potential. Cross-pin lock mechanism designed and field-tested in "
            "longitudinal observational studies. Injection moulding process refined with draft angles "
            "and vent holes. In vitro aspiration-safety testing over months. FY25: 36-hour jig "
            "testing at 10x worst-case bruxism cycles — fins retained, no locking failure."
        ),
        "q2_outcomes":         (
            "Redesigned spine/support mechanism provides robust structural support for polyolefin fins. "
            "Lock-and-release mechanism meets retention and replaceability goals. Stress testing "
            "confirms polyolefin superior to silicone in wear resistance and dimensional stability. "
            "36-hour bruxism simulation: fins remained secure with no locking mechanism failure. "
            "Identified unresolved issue: potential fracture line along side walls of fins under high "
            "load — design modification (side wall thickening) currently under investigation."
        ),
        "q2_new_knowledge":    (
            "Established that polyolefin provides more balanced fin solution vs nylon — extends lifespan "
            "aligning with MRD body while maintaining secure attachment. Confirmed that nylon-on-nylon "
            "contact causes accelerated wear, highlighting importance of material pairing optimisation. "
            "Generated new knowledge on interaction between fin geometry, spine geometry, and slip "
            "potential. Identified fin side-wall fracture risk under load — novel structural finding "
            "requiring further investigation in next R&D period."
        ),

        # ── Section 4: Core Activity 1.3 — EMG Module ────────────────────────
        "q3_activity_title":   "Core R&D Activity 1.3 – EMG module for RMMA testing for sleep apnoea testing",
        "q3_activities":       (
            "Development of portable EMG module to reliably detect rhythmic masticatory muscle activity "
            "(RMMA) during sleep for OSA phenotyping. ANR Corp Muscle Sense M40 wireless unit selected "
            "for masseter bipolar recording. Temporalis site excluded (false positives from scallop-shaped "
            "fibres). Python-based software filters developed for RMMA burst detection. In FY25: Philips "
            "Respironics hardware discontinued — pivot to AI-based EMG signal processing platforms. "
            "Python algorithm found unsustainable at scale."
        ),
        "q3_hypothesis":       (
            "A portable EMG testing device with compact sensor and Python-based EMG system can capture "
            "and analyse RMMA signals in home environments, enabling Type 2 and Type 3 diagnostic level "
            "sleep apnoea monitoring without full laboratory setup. Masseter-only recording with custom "
            "software filters can isolate RMMA bursts and replace cumbersome audiovisual validation "
            "currently required for mainstream clinical use."
        ),
        "q3_uncertainty":      (
            "No EMG modules currently configured for RMMA-specific data collection in sleep testing. "
            "Existing systems designed for daytime/clinical use — impractical for overnight deployment. "
            "RMMA research historically underfunded vs core apnoea parameters. Whether portable "
            "masseter-only EMG could replace audiovisual validation, maintain signal fidelity overnight "
            "in home environments, and be automated for clinical scalability was unknown. Python "
            "algorithm sustainability under repeated large-scale trials was unconfirmed."
        ),
        "q3_systematic":       (
            "EMG unit bench validation: controlled masseter contractions, simulated chewing, rhythmic "
            "contractions. Temporalis excluded after false positive analysis. In vivo overnight sleep "
            "studies with EEG/EOG/airflow/oximetry correlation. Iterative amplitude/frequency "
            "threshold calibration. Flexible printed circuit (FPC) prototypes developed with Sun "
            "Industries. Patient comfort/adhesion/usability trials. Python MVP for bruxism episode "
            "translation and per-hour event reporting. FY25: AI platform evaluation commenced."
        ),
        "q3_outcomes":         (
            "ANR Corp M40 confirmed as best hardware platform: compact, lightweight, bipolar masseter "
            "optimised. Temporalis exclusion significantly improved accuracy. Python MVP demonstrated "
            "proof-of-concept bruxism detection. FY25: Python algorithm unsustainable at scale; "
            "Philips Respironics discontinuation eliminated hardware pathway. AI-based signal "
            "classification identified as only viable pathway for automated, scalable RMMA detection "
            "suitable for mainstream Type 2/3 home sleep testing."
        ),
        "q3_new_knowledge":    (
            "Confirmed EMG modules designed for non-sleep applications can physically capture masseter "
            "activity, but Python-based algorithms insufficient for reliable RMMA signal conversion at "
            "scale. Established that masseter-only recording reduces false positives vs dual-muscle "
            "setups. Determined temporalis scallop-shaped fibres cause systematic false positives — "
            "novel finding. Clarified technical limitations of current hardware; confirmed AI-driven "
            "classification as necessary next step for clinically viable RMMA home testing."
        ),

        # ── Section 5: Core Activity 1.4 — OTS MAS ───────────────────────────
        "q4_activity_title":   "Core R&D Activity 1.4 – Development of an off the shelf mandibular advancement splint (MAS)",
        "q4_activities":       (
            "Development of one-size-fits-all OTS MAS compatible with sequential aligner orthodontic "
            "treatment without compromising OSA therapy. Injection-moulded prototype with EVA filler "
            "developed. FY25: 6-month+ patient trials for long-term usability data. Boil-and-bite "
            "application developed for multi-aligner system compatibility. EVA lining detachment "
            "issue identified — next prototype to incorporate mechanical retention enhancements to "
            "hard shell interface."
        ),
        "q4_hypothesis":       (
            "A one-size-fits-all OTS MAS incorporating a flexible EVA liner within a remouldable "
            "thermoplastic hard shell can deliver therapeutic effectiveness for OSA while remaining "
            "compatible with sequential orthodontic aligner therapy. EVA-shell system will distribute "
            "occlusal/bruxism forces and orthodontic forces, maintain mandibular advancement, "
            "accommodate changing dentition, and improve compliance vs current rigid OTS devices. "
            "User-adjustable fins will enable fit refinement without laboratory remanufacture."
        ),
        "q4_uncertainty":      (
            "No commercial MAS designed for dual OSA/orthodontic functionality. All surveyed devices "
            "(SnoreRx, SnoreMD, EMA, SnoreMedic, Ripsnore, SleepDoctor) designed solely for "
            "mandibular advancement. No peer-reviewed data on long-term durability of EVA-shell "
            "adhesion under bruxism/orthodontic conditions. No validation of user-adjustable "
            "mechanisms maintaining compliance during orthodontic tooth movement. No evidence "
            "whether boil-and-bite approach could accommodate sequential aligner therapy."
        ),
        "q4_systematic":       (
            "Iterative prototyping testing MAS design, material pairings, elasticity, and patient fit. "
            "Baseline established with commercial boil-and-bite and RPT designs. FY24: injection "
            "moulding test mould validated EVA-shell bond at scale — first MVP produced. Repeated "
            "insertion/removal stress testing. Laboratory EVA-shell adhesion testing under cyclic "
            "loads. FY25: 6-month+ structured patient trials. Boil-and-bite approach developed for "
            "aligner compatibility. Long-term durability and compliance data collected."
        ),
        "q4_outcomes":         (
            "Dual-material liner-shell design feasible and manufacturable. Device supports changes in "
            "operational fitment depending on adjustment fin in place — unanticipated finding. "
            "Boil-and-bite application confirmed compatible with multiple aligner systems. "
            "EVA lining detachment identified after prolonged use and repeated cleaning — current "
            "interface lacks long-term reliability. Next prototype to incorporate mechanical "
            "retention enhancements. Concept of one-size-fits-all adaptable device validated."
        ),
        "q4_new_knowledge":    (
            "Novel finding: adjustment fins influence not only mandibular protrusion but broader mode "
            "of device fitment, altering tooth-device interactions in previously undocumented ways. "
            "Confirmed boil-and-bite sufficient adaptability for sequential orthodontic aligners — "
            "first documented evidence. Established that EVA-shell adhesion insufficient for long-term "
            "reliability under clinical conditions — critical design constraint. Advanced understanding "
            "of material science, patient-device dynamics, and OSA/orthodontic therapy integration."
        ),

        # ── Section 6: Budget & Expenditure ──────────────────────────────────
        "total_rd_expenditure":  "1800000",
        "budget_forecast":       "1800000",
        "staff_costs":           "900000",
        "contractor_costs":      "270000",
        "materials_consumables": "360000",
        "equipment_depreciation":"180000",
        "other_eligible_costs":  "90000",

        # ── Section 7: Supporting Activities ─────────────────────────────────
        "supporting_activities": (
            "1.1.1 R&D Project administrative activities (HR, IT, travel, compliance) — 5% effort. "
            "1.1.2 Investigations: literature reviews, expert consultation, benchmarking, 3D printing, "
            "modelling, sampling — 5% effort. "
            "1.1.3 Procurement: materials (silicone, TPE, polyolefin, EVA, gypsum), 3D printing hardware, "
            "injection moulds, patient bookings, clinical/lab hours — 5% effort. "
            "1.1.4 Project management and control: progress monitoring, result evaluation, expert liaison, "
            "compliance management, risk management — 5% effort. "
            "1.2.1 Redesign of nylon fins to injection-moulded polyolefin: draft angles, vent holes, "
            "structural geometry modifications — 15% effort. "
            "1.3.1 Online tool: Bubble platform web app integrating JotForm, Acuity, Zapier for secure "
            "patient data capture and clinical workflow — 30% effort."
        ),

        # ── Recordkeeping & IP ────────────────────────────────────────────────
        "recordkeeping":         (
            "Financial/contractual records; project management records; design/development documentation; "
            "testing/evaluation records with structured test plans and QA reports; knowledge capture "
            "including meeting notes, user feedback, photographs, prototypes, design decision logs."
        ),
        "ip_ownership":          (
            "Company 2 PTY LTD owns all IP including technology, software code, algorithms, processes, "
            "devices. No IP assigned to third parties. Service agreements assign all IP to Company 2. "
            "Full control over R&D direction by Company Director. All costs funded by Company 2."
        ),
        "overseas_activities":   "No",
        "anzsic_division":       "Q – Health Care and Social Assistance",
        "anzsic_class":          "8531 – Dental Services",
        "anzsrc_division":       "42 – Health Sciences",
        "anzsrc_group":          "4299 – Other Health Sciences",
    }

    pdf = generate_rd_dashboard_pdf(
        form_data=SAMPLE,
        openai_api_key=os.environ.get("OPENAI_API_KEY", "YOUR_KEY_HERE"),
        output_path=os.path.expanduser("~/Desktop/rd_dashboard_company2.pdf")
    )
    print(f"\n✅ Done: {pdf}")