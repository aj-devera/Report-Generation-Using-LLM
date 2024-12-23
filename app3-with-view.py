from flask import Flask, request, render_template, send_file, url_for, redirect
from werkzeug.utils import secure_filename
from function.generate_report import generate_report
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

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return 'No file part', 400
    
    report_type = request.form.get('report-type')
    if not report_type:
        return 'Report type not specified', 400
    
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
        report_id = generate_report(uploaded_files, report_type)
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