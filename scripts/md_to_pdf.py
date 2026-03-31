#!/usr/bin/env python3
"""Convert Markdown files to beautifully styled PDFs."""

import sys
import markdown
from weasyprint import HTML
from pathlib import Path


CSS = """
@page {
    size: A4;
    margin: 2cm 2.5cm;
    @top-center {
        content: "NexiFuse Health: Robust Expert";
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 9px;
        color: #94a3b8;
        padding-bottom: 8px;
        border-bottom: 0.5px solid #e2e8f0;
    }
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 9px;
        color: #94a3b8;
        padding-top: 8px;
        border-top: 0.5px solid #e2e8f0;
    }
    @bottom-right {
        content: "Confidential";
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 8px;
        color: #cbd5e1;
    }
}

body {
    font-family: 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
    font-size: 11px;
    line-height: 1.65;
    color: #1e293b;
    max-width: 100%;
}

h1 {
    font-size: 26px;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 10px;
    margin-top: 30px;
    margin-bottom: 20px;
    page-break-after: avoid;
}

h2 {
    font-size: 20px;
    font-weight: 600;
    color: #1e40af;
    border-bottom: 1.5px solid #93c5fd;
    padding-bottom: 6px;
    margin-top: 28px;
    margin-bottom: 14px;
    page-break-after: avoid;
}

h3 {
    font-size: 15px;
    font-weight: 600;
    color: #1e3a5f;
    margin-top: 22px;
    margin-bottom: 10px;
    page-break-after: avoid;
}

h4 {
    font-size: 13px;
    font-weight: 600;
    color: #475569;
    margin-top: 16px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

p {
    margin-bottom: 10px;
}

strong {
    color: #0f172a;
    font-weight: 600;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0 18px 0;
    font-size: 10px;
    page-break-inside: auto;
}

thead {
    display: table-header-group;
}

tr {
    page-break-inside: avoid;
    page-break-after: auto;
}

th {
    background: linear-gradient(180deg, #1e40af, #1e3a8a);
    color: white;
    font-weight: 600;
    padding: 8px 10px;
    text-align: left;
    font-size: 10px;
    letter-spacing: 0.3px;
}

td {
    padding: 7px 10px;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
}

tr:nth-child(even) {
    background-color: #f8fafc;
}

tr:hover {
    background-color: #eff6ff;
}

code {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', Consolas, monospace;
    font-size: 9.5px;
    background-color: #f1f5f9;
    color: #dc2626;
    padding: 2px 5px;
    border-radius: 3px;
    border: 0.5px solid #e2e8f0;
}

pre {
    background-color: #0f172a;
    color: #e2e8f0;
    padding: 14px 16px;
    border-radius: 6px;
    font-size: 9px;
    line-height: 1.55;
    overflow-x: auto;
    margin: 12px 0;
    border-left: 4px solid #3b82f6;
    page-break-inside: avoid;
}

pre code {
    background: none;
    color: #e2e8f0;
    padding: 0;
    border: none;
    font-size: 9px;
}

blockquote {
    border-left: 4px solid #3b82f6;
    background-color: #eff6ff;
    padding: 10px 16px;
    margin: 12px 0;
    color: #1e40af;
    font-style: italic;
    border-radius: 0 4px 4px 0;
}

ul, ol {
    margin: 8px 0;
    padding-left: 24px;
}

li {
    margin-bottom: 4px;
}

li > ul, li > ol {
    margin-top: 4px;
    margin-bottom: 4px;
}

hr {
    border: none;
    height: 1.5px;
    background: linear-gradient(90deg, #3b82f6, #93c5fd, transparent);
    margin: 24px 0;
}

a {
    color: #2563eb;
    text-decoration: none;
}

/* Special styling for grade indicators */
td:last-child {
    font-weight: 500;
}

/* Checkbox styling */
li input[type="checkbox"] {
    margin-right: 6px;
}
"""


def md_to_pdf(md_path: str, pdf_path: str = None):
    """Convert a Markdown file to a styled PDF."""
    md_file = Path(md_path)
    if not md_file.exists():
        print(f"Error: {md_path} not found")
        sys.exit(1)

    if pdf_path is None:
        pdf_path = str(md_file.with_suffix('.pdf'))

    md_text = md_file.read_text(encoding='utf-8')

    # Convert Markdown to HTML
    extensions = ['tables', 'fenced_code', 'codehilite', 'toc', 'nl2br']
    html_body = markdown.markdown(md_text, extensions=extensions)

    # Wrap in full HTML document
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    # Generate PDF
    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"Generated: {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_pdf.py <file.md> [output.pdf]")
        sys.exit(1)

    md_path = sys.argv[1]
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else None
    md_to_pdf(md_path, pdf_path)
