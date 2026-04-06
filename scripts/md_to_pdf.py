#!/usr/bin/env python3
"""Convert Markdown to styled PDF with proper table rendering."""

import sys
import markdown
from weasyprint import HTML

CSS = """
@page {
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9px;
        color: #888;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
}

body {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 11px;
    line-height: 1.55;
    color: #1a1a1a;
    max-width: 100%;
}

h1 {
    font-size: 22px;
    font-weight: 700;
    color: #0d1b2a;
    border-bottom: 3px solid #1b4965;
    padding-bottom: 8px;
    margin-top: 30px;
    margin-bottom: 16px;
    page-break-after: avoid;
}

h2 {
    font-size: 16px;
    font-weight: 700;
    color: #1b4965;
    border-bottom: 2px solid #bee9e8;
    padding-bottom: 5px;
    margin-top: 26px;
    margin-bottom: 12px;
    page-break-after: avoid;
}

h3 {
    font-size: 13px;
    font-weight: 700;
    color: #2c5f7c;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

p { margin: 6px 0; }
strong { color: #0d1b2a; }

hr {
    border: none;
    border-top: 1px solid #d0d0d0;
    margin: 20px 0;
}

/* === TABLE STYLES === */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 16px 0;
    font-size: 10px;
    page-break-inside: auto;
    border: 1px solid #c0cdd8;
}

tr { page-break-inside: avoid; }
thead { display: table-header-group; }

th {
    background-color: #1b4965;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    padding: 7px 10px;
    font-size: 10px;
    letter-spacing: 0.3px;
    border: 1px solid #153d55;
}

td {
    padding: 6px 10px;
    border: 1px solid #d5dde4;
    vertical-align: top;
    line-height: 1.4;
}

tbody tr:nth-child(even) { background-color: #f0f5fa; }
tbody tr:nth-child(odd) { background-color: #ffffff; }
tbody tr:last-child { border-bottom: 2px solid #1b4965; }

td strong { color: #1b4965; font-weight: 700; }

code {
    background-color: #f0f4f8;
    padding: 1px 5px;
    border-radius: 3px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 9.5px;
    color: #c7254e;
}

pre {
    background-color: #f0f4f8;
    border: 1px solid #d8e2eb;
    border-radius: 4px;
    padding: 10px 14px;
    overflow-x: auto;
    font-size: 9px;
    line-height: 1.5;
    margin: 10px 0;
    page-break-inside: avoid;
}

pre code { background: none; padding: 0; color: #2d3748; }

ul, ol { margin: 6px 0; padding-left: 24px; }
li { margin: 3px 0; }

blockquote {
    border-left: 4px solid #1b4965;
    margin: 10px 0;
    padding: 8px 16px;
    background-color: #f5f9fc;
    color: #444;
}
"""


def convert(md_path, pdf_path):
    with open(md_path, "r") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "nl2br"],
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body>
</html>"""

    HTML(string=full_html).write_pdf(pdf_path)
    print(f"PDF saved: {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: md_to_pdf.py input.md output.pdf")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
