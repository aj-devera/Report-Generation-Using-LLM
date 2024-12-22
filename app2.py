# app.py
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import os
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.units import inch
from reportlab.platypus.flowables import KeepTogether
from datetime import datetime
import textwrap
import tempfile

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_header_footer(canvas, doc):
    """Add header and footer to each page"""
    canvas.saveState()
    
    # Header
    canvas.setFillColor(colors.HexColor('#1e3d59'))
    canvas.rect(0, doc.pagesize[1] - 1.5*inch, doc.pagesize[0], 1.5*inch, fill=True)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 22)
    canvas.drawString(0.5*inch, doc.pagesize[1] - 1*inch, "Generated Report")
    canvas.setFont("Helvetica", 10)
    canvas.drawString(0.5*inch, doc.pagesize[1] - 1.25*inch, 
                     f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Footer
    canvas.setFillColor(colors.HexColor('#1e3d59'))
    canvas.rect(0, 0, doc.pagesize[0], 0.5*inch, fill=True)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.5*inch, 0.25*inch, f"Page {doc.page}")
    
    canvas.restoreState()

def generate_report(input_files):
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.7,
        model="gpt-3.5-turbo"
        # max_tokens=3000  # Increased token limit
    )
    
    # Process each PDF file
    combined_text = []
    for file in input_files:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        combined_text.extend(pages)
    
    # Create and run the summarization chain
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(combined_text)
    
    # Generate PDF report
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    doc = SimpleDocTemplate(
        temp_file.name,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=1.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        spaceBefore=12,
        spaceAfter=12,
        firstLineIndent=24
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1e3d59'),
        spaceBefore=24,
        spaceAfter=12
    ))
    
    # Build content
    story = []
    
    # Add sections
    sections = summary.split('\n\n')
    for section in sections:
        if section.strip():
            # Add section header if it looks like a header
            if len(section.split('\n')[0]) < 50:
                story.append(Paragraph(section.split('\n')[0], styles['SectionHeader']))
                section_content = '\n'.join(section.split('\n')[1:])
            else:
                section_content = section
                
            # Wrap long paragraphs
            wrapped_content = textwrap.fill(section_content, width=80)
            story.append(Paragraph(wrapped_content, styles['CustomBody']))
            story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story, onFirstPage=create_header_footer, onLaterPages=create_header_footer)
    return temp_file.name

# Flask routes remain the same as before
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return 'No file part', 400
    
    files = request.files.getlist('files[]')
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filepath)
    
    if not uploaded_files:
        return 'No valid files uploaded', 400
    
    try:
        report_path = generate_report(uploaded_files)
        return send_file(report_path, as_attachment=True, download_name='generated_report.pdf')
    except Exception as e:
        return str(e), 500
    finally:
        # Clean up uploaded files
        for filepath in uploaded_files:
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)