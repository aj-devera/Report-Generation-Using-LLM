# app.py
from flask import Flask, request, render_template, send_file, url_for, redirect
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
from langchain.prompts import PromptTemplate
from datetime import datetime
import textwrap
import tempfile
import uuid

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
ALLOWED_EXTENSIONS = {'pdf'}

# Create necessary directories
for folder in [UPLOAD_FOLDER, GENERATED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

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

def format_content_with_bullets(content):
    """Format content preserving bullet points and line breaks"""
    # Split content into lines
    lines = content.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('- '):
            # Format bullet point with bullet character and proper spacing
            formatted_lines.append('• ' + line[2:])
        elif line.startswith('• '):
            formatted_lines.append(line)
        elif line:
            formatted_lines.append(line)
    
    return formatted_lines

def generate_report(input_files):
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.0,
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
    sales_map_prompt_template = """
                      Write a summary of this chunk of text that focuses on the numerical figures of the report and its contributions.
                      {text}
                      """

    sales_map_prompt = PromptTemplate(template=sales_map_prompt_template, input_variables=["text"])

    sales_combine_prompt_template = """
                        Write a generated sales report of the following text delimited by triple backquotes.
                        Return your response in bullet points which focuses the numerical figures in the report such as the overall sales, generated income and revenue, etc.
                        Limit the response to up to 15 maximum bullet points.
                        ```{text}```
                        BULLET POINT SUMMARY:
                        """

    sales_combine_prompt = PromptTemplate(
        template=sales_combine_prompt_template, input_variables=["text"]
    )

    sales_map_reduce_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=sales_map_prompt,
        combine_prompt=sales_combine_prompt,
        return_intermediate_steps=True,
    )

    sales_map_reduce_outputs = sales_map_reduce_chain({"input_documents": pages})

    # Instead of using a temp file, we'll save it in the GENERATED_FOLDER
    report_id = str(uuid.uuid4())
    output_path = os.path.join(app.config['GENERATED_FOLDER'], f'{report_id}.pdf')
    
    doc = SimpleDocTemplate(
        output_path,
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
        name='BulletPoint',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        spaceBefore=6,
        spaceAfter=6,
        leftIndent=20,
        firstLineIndent=0
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1e3d59'),
        spaceBefore=24,
        spaceAfter=12
    ))

    styles.add(ParagraphStyle(
        name='BodyHeader',
        parent=styles['Heading2'],
        fontSize=15,
        spaceBefore=24,
        spaceAfter=12
    ))
    
    # Build content
    story = []
    
    # Add Sales Analysis header
    story.append(Paragraph('Sales Analysis', styles['BodyHeader']))

    # Add sections
    sections = sales_map_reduce_outputs['output_text'].split('\n\n')
    for section in sections:
        
        content_lines = format_content_with_bullets(section)
    
        # Add each line with appropriate styling
        for line in content_lines:
            if line.startswith('•'):
                # Use bullet point style for bullet points
                story.append(Paragraph(line, styles['BulletPoint']))
            else:
                # Use regular style for non-bullet text
                story.append(Paragraph(line, styles['CustomBody']))

        # if section.strip():
        #     # Add section header if it looks like a header
        #     if len(section.split('\n')[0]) < 50:
        #         story.append(Paragraph(section.split('\n')[0], styles['SectionHeader']))
        #         section_content = '\n'.join(section.split('\n')[1:])
        #     else:
        #         section_content = section
                
        #     # Wrap long paragraphs
        #     wrapped_content = textwrap.fill(section_content, width=80)
        #     story.append(Paragraph(wrapped_content, styles['CustomBody']))
        #     story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story, onFirstPage=create_header_footer, onLaterPages=create_header_footer)
    return report_id

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
        report_id = generate_report(uploaded_files)
        return redirect(url_for('view_report', report_id=report_id))
    except Exception as e:
        return str(e), 500
    finally:
        # Clean up uploaded files
        for filepath in uploaded_files:
            if os.path.exists(filepath):
                os.remove(filepath)

@app.route('/view/<report_id>')
def view_report(report_id):
    return render_template('view_report.html', report_id=report_id)

@app.route('/pdf/<report_id>')
def serve_pdf(report_id):
    pdf_path = os.path.join(app.config['GENERATED_FOLDER'], f'{report_id}.pdf')
    if os.path.exists(pdf_path):
        return send_file(pdf_path, mimetype='application/pdf')
    return 'PDF not found', 404

@app.route('/download/<report_id>')
def download_pdf(report_id):
    pdf_path = os.path.join(app.config['GENERATED_FOLDER'], f'{report_id}.pdf')
    if os.path.exists(pdf_path):
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='generated_report.pdf'
        )
    return 'PDF not found', 404

# Optional: Cleanup task (you might want to implement a proper cleanup strategy)
@app.route('/cleanup/<report_id>')
def cleanup(report_id):
    pdf_path = os.path.join(app.config['GENERATED_FOLDER'], f'{report_id}.pdf')
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    return 'Cleaned up'

if __name__ == '__main__':
    app.run(debug=True)