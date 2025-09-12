#!/usr/bin/env python3
"""
Convert Markdown report to PDF
"""

import markdown
import pdfkit
import os
from pathlib import Path

def convert_md_to_pdf():
    """Convert the markdown report to PDF"""
    
    # Read the markdown file
    with open('FINAL_PROJECT_REPORT.md', 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html = markdown.markdown(md_content, extensions=['tables', 'codehilite'])
    
    # Add CSS styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #2c3e50;
                margin-top: 30px;
                margin-bottom: 15px;
            }}
            h1 {{
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 5px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                font-size: 14px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
                color: #2c3e50;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #3498db;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 20px 0;
                padding: 10px 20px;
                background-color: #f8f9fa;
            }}
            .abstract {{
                background-color: #e8f4f8;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
                border-left: 4px solid #3498db;
            }}
            .keywords {{
                font-style: italic;
                color: #7f8c8d;
            }}
            .page-break {{
                page-break-before: always;
            }}
            @media print {{
                body {{ margin: 0; }}
                .no-print {{ display: none; }}
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    # Save HTML file
    with open('FINAL_PROJECT_REPORT.html', 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    print("✅ HTML file created: FINAL_PROJECT_REPORT.html")
    
    # Try to convert to PDF
    try:
        # Configure PDF options
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        # Convert to PDF
        pdfkit.from_string(styled_html, 'FINAL_PROJECT_REPORT.pdf', options=options)
        print("✅ PDF file created: FINAL_PROJECT_REPORT.pdf")
        
    except Exception as e:
        print(f"❌ PDF conversion failed: {e}")
        print("📝 You can use the HTML file and convert it to PDF using:")
        print("   - Browser: Open HTML file → Print → Save as PDF")
        print("   - Online converter: Upload HTML file to any online converter")

if __name__ == "__main__":
    convert_md_to_pdf()
