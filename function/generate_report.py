from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
import os
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

def format_content_with_bullets(content):
    """Format content preserving bullet points and line breaks"""
    if content.startswith('- '):
        text = '• ' + content[2:]
    else:
        text = content

    return text

def format_content_with_headings(content):
    if content.startswith('# ') or content.startswith('## ') or content.startswith('### '):
        heading_text = content[2:]
        style = "Subsection"
    elif content.startswith('#### '):
        heading_text = content[5:]
        style = "Subsubsection"
    elif content.startswith('- '):
        heading_text = '• ' + content[2:]
        style = "BulletPoint"
    else:
        heading_text = content
        style = "CustomBody"

    return heading_text, style

def get_prompt_template(report_type):
    if report_type == "sales":
        title = "Sales Analysis"
        map_prompt_template = """
                            Write a summary of this chunk of text that focuses on the numerical figures of the report and its contributions.
                            {text}
                            """
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        combine_prompt_template = """
                        Write a generated sales report of the following text delimited by triple backquotes.
                        Make two major sections. 
                        The first section should focus on the numerical figures such as overall sales, total revenue, number of deaths, etc.
                        The second section should focus on the descriptive highlights of the contributions on how these figures were met.
                        Do not use ** to indicate bold letters or headings. Use # as an indicator for major sections.
                        Use only one "\n" when doing a new line. Do not use two or more.
                        Limit the response to up to 15 maximum bullet points.
                        ```{text}```
                        BULLET POINT SUMMARY:
                        """

    elif report_type == "news":
        title = "News Summary"
        map_prompt_template = """
                            Write a summary of this chunk of text as if you are writing for a news article.
                            {text}
                            """
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        combine_prompt_template = """
                                Review the overall context first. Group it based on topics/focal points.
                                Write a comprehensive news summary covering the key events and developments for each topic/focal point.
                                Ensure that the important dates and timelines are included in the summary.
                                If there are numerical figures included such as sales, revenue, or something of value, highlight this section of the context.
                                Separate the summary by paragraphs based on topics/focal points.
                                The summary can be long as much as half of the full context.
                                Do not use ** to indicate bold letters or headings. Use # as an indicator for major sections and ## for minor sections.
                                Use only one "\n" when doing a new line. Do not use two or more.

                                The context for this news is delimited by the triple backquotes:
                                ```{text}```

                                News content:
                                """
        
    elif report_type == "content":
        title = "Content Summary"
        map_prompt_template = """
                            Write a summary of this chunk of text that highlights its content.
                            {text}
                            """
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        combine_prompt_template = """
                                Create a detailed summary of the following content, including:
                                1. Main themes and key points
                                2. Important findings
                                3. Significant details
                                4. Conclusions

                                The context for this content is delimited by the triple backquotes:
                                ```{text}```
                                
                                Content Summary:
                                """
    else:
        raise ValueError("Invalid report type")
        
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text"]
    )
    return title, map_prompt, combine_prompt

def generate_report(input_files, report_type):
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.0,
        model="gpt-4o-mini-2024-07-18"
        # max_tokens=3000  # Increased token limit
    )
    
    # Process each PDF file
    combined_text = []
    for file in input_files:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        combined_text.extend(pages)

    # Create the prompt template based on report type
    title, map_prompt, combine_prompt = get_prompt_template(report_type)

    map_reduce_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=True,
    )

    map_reduce_outputs = map_reduce_chain({"input_documents": combined_text})

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
        firstLineIndent=24,
        alignment=4
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
        spaceAfter=12,
        alignment=1
    ))

    styles.add(ParagraphStyle(
        name='Subsection',
        parent=styles['Heading1'],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=6,
        alignment=0
    ))

    styles.add(ParagraphStyle(
        name='Subsubsection',
        parent=styles['Heading1'],
        fontSize=11,
        spaceBefore=12,
        spaceAfter=6,
        alignment=0
    ))
    
    # Build content
    story = []
    
    # Add Sales Analysis header
    story.append(Paragraph(title, styles['SectionHeader']))

    # Add sections
    sections = map_reduce_outputs['output_text'].split('\n')
    print(sections)
    for line in sections:
        if line.startswith('•') or line.startswith('- '):
            # Use bullet point style for bullet points
            formatted_line = format_content_with_bullets(line)
            story.append(Paragraph(formatted_line, styles['BulletPoint']))
        elif line.startswith('#'):
            formatted_line, style = format_content_with_headings(line)
            story.append(Paragraph(formatted_line, styles[style]))             
        else:
            # Use regular style for non-bullet text
            story.append(Paragraph(line, styles['CustomBody']))
    
    # Build PDF
    doc.build(story, onFirstPage=create_header_footer, onLaterPages=create_header_footer)
    return report_id