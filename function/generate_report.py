from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus.flowables import KeepTogether
from datetime import datetime
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import textwrap
import tempfile
import uuid

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
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    pdf_loader = PyPDFLoader(input_files)
    pages = pdf_loader.load_and_split()
    
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
    output_path = os.path.join('generated', f'{report_id}.pdf')
    
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
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1e3d59'),
        spaceBefore=24,
        spaceAfter=12
    ))
    
    # Build content
    story = []
    # Add report title
    story.append(Paragraph('Sales Analysis', styles['CustomTitle']))
    
    # Add sections
    sections = sales_map_reduce_outputs.split('\n\n')
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
    return report_id