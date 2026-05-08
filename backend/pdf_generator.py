import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def _get_styles():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1D9E75'), # Brand Green
        spaceAfter=20
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#111827'),
        spaceAfter=10,
        spaceBefore=15
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#374151'),
        spaceAfter=12,
        leading=16
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#6B7280'),
        fontName='Helvetica-Oblique',
        spaceBefore=30
    )
    return styles, title_style, heading_style, body_style, disclaimer_style

def build_fraud_report(tx_id: str, score: float, explanation: str) -> io.BytesIO:
    """Builds a PDF report for a fraud case."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles, title_style, heading_style, body_style, _ = _get_styles()
    
    elements = []
    
    # Header
    elements.append(Paragraph("FinRisk AI Platform", styles['Normal']))
    elements.append(Paragraph("Fraud Case Report", title_style))
    
    # Meta Data Table
    data = [
        ['Report Generated', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ['Transaction ID', tx_id],
        ['Fraud Score', f"{score:.4f}"],
        ['Risk Level', 'HIGH RISK' if score > 0.8 else 'MEDIUM RISK' if score > 0.4 else 'LOW RISK']
    ]
    
    t = Table(data, colWidths=[150, 300])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#111827')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))
    
    # AI Explanation
    elements.append(Paragraph("AI Agent Analysis", heading_style))
    
    # Replace markdown bold ** with HTML <b> for reportlab
    fmt_explanation = explanation.replace('**', '<b>').replace('**', '</b>') # Need careful replace
    
    # Better bold replacement
    import re
    fmt_explanation = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', explanation)
    
    # Split by newlines and add paragraphs
    for p in fmt_explanation.split('\n'):
        if p.strip():
            elements.append(Paragraph(p.strip(), body_style))
            
    doc.build(elements)
    buffer.seek(0)
    return buffer

def build_investment_report(query: str, advice: str) -> io.BytesIO:
    """Builds a PDF report for investment advice."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles, title_style, heading_style, body_style, disclaimer_style = _get_styles()
    
    elements = []
    
    # Header
    elements.append(Paragraph("FinRisk AI Platform", styles['Normal']))
    elements.append(Paragraph("Investment Advice Report", title_style))
    
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    elements.append(Spacer(1, 10))
    
    # Query
    elements.append(Paragraph("Client Query", heading_style))
    elements.append(Paragraph(f"<i>\"{query}\"</i>", body_style))
    elements.append(Spacer(1, 10))
    
    # AI Advice
    elements.append(Paragraph("Agent Guidance", heading_style))
    
    import re
    fmt_advice = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', advice)
    
    # Try to clean out disclaimer from the main body if we want to style it differently
    disclaimer_text = ""
    if "DISCLAIMER:" in fmt_advice:
        parts = fmt_advice.split("DISCLAIMER:")
        fmt_advice = parts[0]
        disclaimer_text = "DISCLAIMER:" + parts[1]
    
    for p in fmt_advice.split('\n'):
        if p.strip():
            # Basic support for bullet points
            if p.strip().startswith('- '):
                elements.append(Paragraph(f"• {p.strip()[2:]}", body_style))
            else:
                elements.append(Paragraph(p.strip(), body_style))
                
    if disclaimer_text:
        elements.append(Paragraph(disclaimer_text, disclaimer_style))
            
    doc.build(elements)
    buffer.seek(0)
    return buffer
