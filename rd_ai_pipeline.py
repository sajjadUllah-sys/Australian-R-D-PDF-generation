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
from dotenv import load_dotenv
load_dotenv()
import os
import json
import io
import tempfile
import uuid
from datetime import datetime

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

    user_prompt = f"""
FORM DATA SUBMITTED BY APPLICANT:
{json.dumps(form_data, indent=2)}

TODAY'S DATE: {datetime.now().strftime('%d %B %Y')}

Using your expert knowledge of Australian R&D Tax Incentive legislation,
analyse the above form data and generate a complete dashboard JSON.

Return this EXACT structure (fill every field with real, derived values):

{{
  "project_info": {{
    "title": "project title from form",
    "company": "company name if provided, else 'Applicant Company Pty Ltd'",
    "financial_year": "e.g. FY2024-25",
    "industry": "industry from form",
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "duration_years": 1.0,
    "staff_members": ["Name1", "Name2"],
    "generated_date": "{datetime.now().strftime('%d %B %Y')}"
  }},

  "kpi_cards": [
    {{"label": "Project Duration",    "value": "X Years",   "icon": "clock"}},
    {{"label": "Total R&D Spend",     "value": "$XXX,XXX",  "icon": "dollar"}},
    {{"label": "Staff Involved",      "value": "X Members", "icon": "people"}},
    {{"label": "Quarters Reported",   "value": "4",         "icon": "calendar"}},
    {{"label": "Tax Offset (43.5%)",  "value": "$XX,XXX",   "icon": "tax"}},
    {{"label": "ATO Compliance Score","value": "X/10",      "icon": "check"}}
  ],

  "ato_compliance": {{
    "overall_score": 8,
    "eligible": true,
    "risk_level": "Low",
    "summary": "2-3 sentence overall eligibility assessment citing specific IR&D Act sections",
    "meets_new_knowledge": true,
    "meets_uncertainty": true,
    "meets_systematic": true,
    "strengths": [
      "specific strength 1 with ATO reference",
      "specific strength 2"
    ],
    "issues": [
      "specific compliance gap 1 if any"
    ],
    "suggestions": [
      "specific improvement to strengthen the claim"
    ],
    "audit_risk_flags": [
      "any red flags that might attract ATO scrutiny"
    ]
  }},

  "quarterly_summary": [
    {{
      "quarter": "Q1",
      "activities_summary": "2-sentence summary derived from q1_activities",
      "hypothesis": "concise hypothesis from q1_hypothesis",
      "uncertainty": "what was genuinely unknown",
      "systematic_method": "how it was systematically conducted",
      "key_outcome": "main result",
      "new_knowledge": "what new knowledge was generated",
      "eligibility_score": 8,
      "core_or_supporting": "Core",
      "compliance_notes": "any specific ATO notes for this quarter"
    }},
    {{"quarter": "Q2", "...": "..."}},
    {{"quarter": "Q3", "...": "..."}},
    {{"quarter": "Q4", "...": "..."}}
  ],

  "expenditure": {{
    "total": 180000,
    "staff_costs": 120000,
    "contractor_costs": 20000,
    "materials_consumables": 25000,
    "equipment_depreciation": 10000,
    "other_eligible_costs": 5000,
    "estimated_tax_offset": 78300,
    "core_percentage": 65,
    "supporting_percentage": 35,
    "annual_breakdown": [
      {{"year": "FY2025", "staff": 80000, "materials": 15000, "equipment": 8000, "external": 5000}}
    ]
  }},

  "technical_progress": [
    {{
      "challenge": "Short challenge name",
      "progress_pct": 75,
      "resolved_items": ["✓ Resolved item 1", "✓ Resolved item 2"],
      "pending_items": ["⚠ Pending item 1"]
    }}
  ],

  "innovations": [
    {{
      "category": "Category (e.g. Material Science, Algorithm, Clinical)",
      "finding": "Specific innovation or knowledge generated",
      "impact": "High"
    }}
  ],

  "recommendations": [
    {{
      "priority": "High",
      "action": "Specific actionable recommendation to strengthen ATO claim or improve R&D"
    }}
  ]
}}

RULES:
- estimated_tax_offset = total_rd_expenditure * 0.435 (for < $20M turnover)
- Derive all quarterly data directly from the submitted form fields
- eligibility_score per quarter: score 1-10 based on how well each of the 3 ATO tests are met
- overall compliance score: weighted average + documentation quality assessment
- If a quarter's data is missing/sparse, score it lower and flag in recommendations
- technical_progress: infer from the activities and outcomes described across quarters
- innovations: extract genuine novel findings from new_knowledge fields
- Be specific — reference actual content from the form, not generic statements
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
    fig, ax  = plt.subplots(figsize=(6.5, 4))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LIGHT_GREY)
    bar_colors = [TEAL if s >= 7 else (ORANGE if s >= 5 else "#E74C3C") for s in scores]
    bars = ax.bar(quarters, scores, color=bar_colors, width=0.45,
                  edgecolor="white", linewidth=1.5)
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
    return _fig_to_image(fig, width_mm=90, height_mm=72)


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
        colWidths=[84*mm, 93*mm]
    )
    side_by_side.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0),(-1,-1), 2),
        ("RIGHTPADDING", (0,0),(-1,-1), 2),
    ]))
    story.append(side_by_side)
    story.append(Spacer(1, 3*mm))

    # Financial summary table
    fin_rows = [
        ["Expenditure Category",          "Amount (AUD)",            "% of Total"],
        ["Staff Costs",                    f"${exp.get('staff_costs',0):,.0f}",
         f"{exp.get('staff_costs',0)/max(total,1)*100:.0f}%"],
        ["Contractor Costs",               f"${exp.get('contractor_costs',0):,.0f}",
         f"{exp.get('contractor_costs',0)/max(total,1)*100:.0f}%"],
        ["Materials & Consumables",        f"${exp.get('materials_consumables',0):,.0f}",
         f"{exp.get('materials_consumables',0)/max(total,1)*100:.0f}%"],
        ["Equipment & Depreciation",       f"${exp.get('equipment_depreciation',0):,.0f}",
         f"{exp.get('equipment_depreciation',0)/max(total,1)*100:.0f}%"],
        ["Other Eligible Costs",           f"${exp.get('other_eligible_costs',0):,.0f}",
         f"{exp.get('other_eligible_costs',0)/max(total,1)*100:.0f}%"],
        ["TOTAL R&D EXPENDITURE",          f"${total:,.0f}",          "100%"],
        ["Estimated Tax Offset (43.5%)",   f"${offset:,.0f}",         "—"],
    ]
    ft = Table(fin_rows, colWidths=[95*mm, 52*mm, 30*mm])
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

        # Quarter header row
        qhdr = Table([[
            Paragraph(f"<b>{qtr} — {c_or_s} R&D Activity</b>",
                      S(f"qh{qtr}", fontSize=10, textColor=colors.white,
                        fontName="Helvetica-Bold")),
            Paragraph(f"Eligibility Score: <b>{score}/10</b>",
                      S(f"qs{qtr}", fontSize=9, textColor=colors.white,
                        alignment=TA_RIGHT))
        ]], colWidths=[120*mm, 57*mm])
        qhdr.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), RL_BLUE),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(0,0),   8),
            ("RIGHTPADDING",  (-1,0),(-1,0), 8),
        ]))
        story.append(qhdr)

        q_body = [
            ["R&D Activities",     q.get("activities_summary", "—")],
            ["Hypothesis Tested",  q.get("hypothesis",         "—")],
            ["Uncertainty",        q.get("uncertainty",        "—")],
            ["Systematic Method",  q.get("systematic_method",  "—")],
            ["Key Outcome",        q.get("key_outcome",        "—")],
            ["New Knowledge",      q.get("new_knowledge",      "—")],
        ]
        if q.get("compliance_notes"):
            q_body.append(["ATO Notes", q["compliance_notes"]])

        qt = Table(q_body, colWidths=[40*mm, 137*mm])
        qt.setStyle(TableStyle([
            ("FONTNAME",       (0,0),(0,-1), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1), 8),
            ("TEXTCOLOR",      (0,0),(0,-1), RL_BLUE),
            ("ROWBACKGROUNDS", (0,0),(-1,-1), [colors.white, RL_LGREY]),
            ("GRID",           (0,0),(-1,-1), 0.3, colors.HexColor(MID_GREY)),
            ("TOPPADDING",     (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
            ("LEFTPADDING",    (0,0),(-1,-1), 8),
            ("VALIGN",         (0,0),(-1,-1), "TOP"),
        ]))
        story += [qt, Spacer(1, 3*mm)]

    # ── TECHNICAL PROGRESS ────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Technical Challenge Resolution", H2))
    challenges = data.get("technical_progress", [])
    if challenges:
        story.append(make_progress_bars(challenges))
        story.append(Spacer(1, 3*mm))

        prog_rows = [["Challenge", "Progress", "Resolved Items", "Pending Items"]]
        for ch in challenges:
            prog_rows.append([
                Paragraph(f"<b>{ch['challenge']}</b>", SMALL),
                Paragraph(f"<b>{ch['progress_pct']}%</b>", SMALL),
                Paragraph("<br/>".join(ch.get("resolved_items", [])), SMALL),
                Paragraph("<br/>".join(ch.get("pending_items",  [])), SMALL),
            ])
        pt = Table(prog_rows, colWidths=[45*mm, 18*mm, 57*mm, 57*mm])
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
        innov_rows = [["Category", "Innovation / Knowledge Generated", "Impact"]]
        for iv in innovations:
            innov_rows.append([
                Paragraph(f"<b>{iv.get('category','—')}</b>", SMALL),
                Paragraph(iv.get("finding", "—"), SMALL),
                Paragraph(f"<b>{iv.get('impact','—')}</b>", SMALL),
            ])
        it = Table(innov_rows, colWidths=[42*mm, 118*mm, 17*mm])
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
    test_rows = [["ATO Eligibility Test", "Status"]]
    for test_name, passed in tests:
        test_rows.append([
            test_name,
            Paragraph("<b>✓ Met</b>" if passed else "<b>✗ Not Met</b>",
                      S(f"ts{test_name}", fontSize=8,
                        textColor=colors.HexColor(TEAL if passed else "#E74C3C"),
                        fontName="Helvetica-Bold"))
        ])
    tt = Table(test_rows, colWidths=[140*mm, 37*mm])
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
        rec_rows = [["Priority", "Recommended Action"]]
        for rec in recs:
            prio = rec.get("priority", "Medium")
            pc   = "#E74C3C" if prio=="High" else (ORANGE if prio=="Medium" else TEAL)
            rec_rows.append([
                Paragraph(f'<font color="{pc}"><b>{prio}</b></font>', SMALL),
                Paragraph(rec.get("action", "—"), SMALL),
            ])
        rt = Table(rec_rows, colWidths=[22*mm, 155*mm])
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
# =============================================================================
if __name__ == "__main__":
    SAMPLE = {
        "project_title":       "Mandibular Repositioning Device for Sleep Apnoea",
        "brief_summary":       "Development of a next-generation MRD using novel materials and EMG monitoring.",
        "financial_year":      "2025",
        "project_start_date":  "2015-01-01",
        "project_end_date":    "2026-12-31",
        "industry":            "Medical Devices",
        "staff_members":       "Dr. John Smith, Sarah Chen, Mark Thompson",
        "q1_activities":       "Systematic investigation of TPE material properties. 47 bench tests measuring tensile strength and overnight wear.",
        "q1_hypothesis":       "TPE Shore A 50 provides optimal comfort/rigidity balance for 8-hour wear cycles.",
        "q1_uncertainty":      "Unknown whether TPE maintains repositioning angle under variable occlusal forces without permanent deformation.",
        "q1_systematic":       "Controlled compression testing. 12 TPE formulations tested over 30-day simulated wear cycles.",
        "q1_outcomes":         "TPE showed 2.3mm deformation after 90 nights. Comfort 4.2/10. Deemed inadequate.",
        "q1_new_knowledge":    "TPE durometer range for MRD use fully characterised — first in literature.",
        "q2_activities":       "Nylon fin system development. Systematic investigation of nylon-on-nylon wear characteristics.",
        "q2_hypothesis":       "Injection-moulded nylon fins with 0.3mm tolerance enable precise repositioning without degradation.",
        "q2_uncertainty":      "Wear behaviour in biological moisture environment unknown — differs from industrial applications.",
        "q2_systematic":       "27 prototype configurations. Jaw simulator at 3Hz for 100,000 cycles.",
        "q2_outcomes":         "0.8mm wear after 12 months equivalent. Cross-pin lock improved retention 85%.",
        "q2_new_knowledge":    "Salivary enzymes accelerate nylon degradation 340% vs dry — first documented for MRD.",
        "q3_activities":       "Silicone material trials and EMG module prototype development.",
        "q3_hypothesis":       "Medical silicone outperforms nylon; masseter-only EMG achieves >90% bruxism detection.",
        "q3_uncertainty":      "Silicone dimensional stability unclear. EMG masseter/temporalis interference unknown in sleep.",
        "q3_systematic":       "56 patient participants. Sleep lab EMG over 12 weeks. Parallel material testing.",
        "q3_outcomes":         "Silicone 8.2/10 comfort. Masseter-only reduced false positives 25%→8%. Python algo 73% accuracy.",
        "q3_new_knowledge":    "Temporalis signals create systematic false positives — novel finding for dental sleep medicine.",
        "q4_activities":       "Polyolefin evaluation and LSTM-based EMG signal processing development.",
        "q4_hypothesis":       "Polyolefin provides superior fin durability; AI EMG exceeds 95% detection accuracy.",
        "q4_uncertainty":      "Optimal AI architecture for overnight physiological classification from limited data not established.",
        "q4_systematic":       "100 patient extended trial. LSTM trained on 1,847 overnight recordings.",
        "q4_outcomes":         "Polyolefin 62% wear reduction. LSTM 91% accuracy. Compliance 67% at 24 weeks.",
        "q4_new_knowledge":    "Transfer learning improves oral-device EMG detection 23% — novel LSTM application.",
        "total_rd_expenditure":"180000",
        "staff_costs":         "120000",
        "contractor_costs":    "20000",
        "materials_consumables":"25000",
        "equipment_depreciation":"10000",
        "other_eligible_costs": "5000",
    }

    pdf = generate_rd_dashboard_pdf(
        form_data=SAMPLE,
        openai_api_key=os.environ.get("OPENAI_API_KEY", "YOUR_KEY_HERE"),
        output_path=os.path.expanduser("~/Desktop/rd_dashboard_test.pdf")
    )
    print(f"\n✅ Done: {pdf}")
