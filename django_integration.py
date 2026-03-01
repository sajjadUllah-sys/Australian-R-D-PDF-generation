"""
=============================================================================
PATTENS — Django Integration (No RAG — GPT-4o built-in knowledge)
=============================================================================
3 things to wire up:
  1. Add 2 lines to settings.py
  2. Replace your views.py export function
  3. Add 1 URL to urls.py
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. settings.py  — add these two lines
# ─────────────────────────────────────────────────────────────────────────────
"""
import os

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")          # set in .env
RD_PDF_OUTPUT_DIR = os.environ.get("RD_PDF_OUTPUT_DIR", "/tmp/rd_pdfs")
"""


# ─────────────────────────────────────────────────────────────────────────────
# 2. views.py
# ─────────────────────────────────────────────────────────────────────────────
"""
import os, uuid, json
from django.conf import settings
from django.http import FileResponse, JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from .rd_ai_pipeline import generate_rd_dashboard_pdf


@method_decorator(csrf_exempt, name="dispatch")
class GenerateRDDashboardPDF(View):
    '''
    POST /api/projects/generate-pdf/
    
    Body (JSON or form-data) — all 7 form sections:
    {
        "project_title": "...",
        "brief_summary": "...",
        "financial_year": "2025",
        "project_start_date": "2024-07-01",
        "project_end_date": "2025-06-30",
        "industry": "Medical Devices",
        "staff_members": "John Smith, Jane Doe",

        "q1_activities": "...",   "q1_hypothesis": "...",
        "q1_uncertainty": "...",  "q1_systematic": "...",
        "q1_outcomes": "...",     "q1_new_knowledge": "...",

        "q2_activities": "...",   ...  (same for Q2, Q3, Q4)

        "total_rd_expenditure": 180000,
        "staff_costs": 120000,
        "contractor_costs": 20000,
        "materials_consumables": 25000,
        "equipment_depreciation": 10000,
        "other_eligible_costs": 5000
    }

    Returns: PDF file as attachment
    '''

    def post(self, request, *args, **kwargs):
        # Parse body
        try:
            if request.content_type == "application/json":
                form_data = json.loads(request.body)
            else:
                form_data = request.POST.dict()
        except Exception as e:
            return JsonResponse({"error": f"Invalid request body: {e}"}, status=400)

        # Validate minimum required fields
        required = ["project_title", "financial_year", "total_rd_expenditure"]
        missing  = [f for f in required if not form_data.get(f)]
        if missing:
            return JsonResponse({"error": f"Missing fields: {missing}"}, status=400)

        # Generate PDF
        try:
            os.makedirs(settings.RD_PDF_OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(
                settings.RD_PDF_OUTPUT_DIR,
                f"rd_dashboard_{uuid.uuid4().hex}.pdf"
            )

            pdf_path = generate_rd_dashboard_pdf(
                form_data=form_data,
                openai_api_key=settings.OPENAI_API_KEY,
                output_path=output_path,
            )

            safe_name = "".join(
                c for c in form_data.get("project_title","RD")
                if c.isalnum() or c in " _-"
            )[:40].strip()

            return FileResponse(
                open(pdf_path, "rb"),
                content_type="application/pdf",
                as_attachment=True,
                filename=f"RD_Dashboard_{safe_name}.pdf",
            )

        except Exception as e:
            import traceback
            return JsonResponse({
                "error": str(e),
                "trace": traceback.format_exc()
            }, status=500)
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3. urls.py  — add this path
# ─────────────────────────────────────────────────────────────────────────────
"""
from django.urls import path
from .views import GenerateRDDashboardPDF

urlpatterns = [
    path("api/projects/generate-pdf/", GenerateRDDashboardPDF.as_view(), name="generate_rd_pdf"),
]
"""


# ─────────────────────────────────────────────────────────────────────────────
# 4. requirements.txt  — only 4 packages needed
# ─────────────────────────────────────────────────────────────────────────────
"""
openai>=1.30.0
reportlab>=4.0.0
matplotlib>=3.8.0
numpy>=1.26.0
"""


# ─────────────────────────────────────────────────────────────────────────────
# 5. Frontend (React / Axios) call example
# ─────────────────────────────────────────────────────────────────────────────
"""
const handleExportPDF = async (formData) => {
    setLoading(true);
    try {
        const response = await axios.post(
            '/api/projects/generate-pdf/',
            formData,                        // your 7-section form object
            {
                responseType: 'blob',        // <-- critical for PDF download
                headers: { 'Content-Type': 'application/json' }
            }
        );
        const url      = URL.createObjectURL(new Blob([response.data]));
        const link     = document.createElement('a');
        link.href      = url;
        link.download  = 'RD_Dashboard.pdf';
        link.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        console.error('PDF generation failed:', err);
    } finally {
        setLoading(false);
    }
};
"""
