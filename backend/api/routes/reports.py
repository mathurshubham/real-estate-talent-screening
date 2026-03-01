from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import Response
import redis.asyncio as redis
from core.redis import get_redis
import json
import base64
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

router = APIRouter()

@router.post("/sessions/{session_id}/chart")
async def save_chart_image(session_id: str, image_data: str = Body(..., embed=True), r: redis.Redis = Depends(get_redis)):
    # image_data is expected to be a base64 string (data:image/png;base64,...)
    await r.set(f"chart:{session_id}", image_data, ex=3600) # Expire in 1 hour
    return {"status": "success"}

@router.get("/sessions/{session_id}/pdf")
async def generate_report_pdf(session_id: str, r: redis.Redis = Depends(get_redis)):
    state_raw = await r.get(f"session:{session_id}")
    if not state_raw:
        # Check candidate-submitted keys too if needed
        state_raw = await r.get(f"session:{session_id}")
        if not state_raw:
            raise HTTPException(status_code=404, detail="Session not found")
    
    state = json.loads(state_raw)
    chart_base64 = await r.get(f"chart:{session_id}")
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1, textColor=colors.HexColor("#1A365D"))
    elements.append(Paragraph(f"EstateAssess Screening Report", title_style))
    elements.append(Spacer(1, 20))

    # Candidate Info
    elements.append(Paragraph(f"<b>Candidate:</b> {state.get('candidate', {}).get('name')}", styles['Normal']))
    elements.append(Paragraph(f"<b>Role:</b> {state.get('candidate', {}).get('role')}", styles['Normal']))
    elements.append(Paragraph(f"<b>Status:</b> {state.get('status', 'N/A').upper()}", styles['Normal']))
    elements.append(Spacer(1, 20))

    # Add Chart if available
    if chart_base64:
        try:
            # Remove header if present
            if isinstance(chart_base64, bytes):
                chart_base64 = chart_base64.decode()
            if "," in chart_base64:
                header, encoded = chart_base64.split(",", 1)
            else:
                encoded = chart_base64
            img_data = base64.b64decode(encoded)
            img_buffer = io.BytesIO(img_data)
            img = Image(img_buffer, width=400, height=300)
            elements.append(img)
            elements.append(Spacer(1, 20))
        except Exception as e:
            print(f"Error adding chart to PDF: {e}")

    # Questions and Evaluations
    elements.append(Paragraph("Assessment Detail", styles['Heading2']))
    elements.append(Spacer(1, 10))

    questions = state.get("questions", [])
    scores = state.get("scores", {})
    ai_evals = state.get("ai_evaluations", {})
    candidate_answers = {a['question_id']: a for a in state.get("candidate_answers", [])}

    for idx, q in enumerate(questions):
        q_text = q.get('text', 'N/A')
        q_id = q.get('id')
        
        elements.append(Paragraph(f"Q{idx+1}: {q_text}", styles['Heading3']))
        
        # Panelist Score or Candidate Answer
        if q_id in candidate_answers:
            ans = candidate_answers[q_id]
            elements.append(Paragraph(f"<i>Candidate Answer:</i> {ans.get('transcript', 'No answer provided.')}", styles['Normal']))
            
            # AI Eval
            eval_data = ai_evals.get(q_id)
            if eval_data:
                elements.append(Paragraph(f"<b>AI Score: {eval_data.get('score')}/5</b>", styles['Normal']))
                elements.append(Paragraph(f"Justification: {eval_data.get('justification')}", styles['Normal']))
        else:
            score = scores.get(q_id, "N/A")
            elements.append(Paragraph(f"<b>Panelist Score: {score}/5</b>", styles['Normal']))
        
        elements.append(Spacer(1, 15))

    doc.build(elements)
    pdf_value = buffer.getvalue()
    buffer.close()

    return Response(
        content=pdf_value,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{session_id}.pdf"}
    )
