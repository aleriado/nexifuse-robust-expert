"""Documentation ingestion pipeline.

Ingests PDFs, HTML files, and OpenAPI/Swagger JSON specs from the docs/ directory.
Outputs structured text files organized by domain, ready for context injection
into the teacher model.

Optional dependencies:
  - pdfplumber (PDF extraction)
  - beautifulsoup4 + lxml (HTML extraction)
Falls back gracefully if not installed.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Try optional imports
try:
    import pdfplumber
    _HAS_PDF = True
except ImportError:
    pdfplumber = None  # type: ignore[assignment]
    _HAS_PDF = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment]
    _HAS_BS4 = False


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file using pdfplumber."""
    if not _HAS_PDF:
        logger.warning("pdfplumber not installed — skipping PDF: %s", pdf_path)
        return ""

    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
    except Exception as exc:
        logger.error("Failed to extract PDF %s: %s", pdf_path, exc)
        return ""

    return "\n\n".join(pages)


def extract_html_text(html_path: Path) -> str:
    """Extract text content from an HTML file, stripping tags."""
    try:
        raw = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.error("Failed to read HTML %s: %s", html_path, exc)
        return ""

    if _HAS_BS4:
        soup = BeautifulSoup(raw, "html.parser")
        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    else:
        # Basic fallback: strip HTML tags with regex
        logger.debug("beautifulsoup4 not installed — using regex fallback for %s", html_path)
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_openapi_summary(json_path: Path) -> str:
    """Extract endpoint summaries from an OpenAPI/Swagger JSON spec."""
    try:
        raw = json_path.read_text(encoding="utf-8")
        spec = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse OpenAPI spec %s: %s", json_path, exc)
        return ""

    lines: list[str] = []

    # Title and description
    info = spec.get("info", {})
    if info.get("title"):
        lines.append(f"API: {info['title']} (v{info.get('version', '?')})")
    if info.get("description"):
        lines.append(info["description"][:500])
    lines.append("")

    # Endpoints
    paths = spec.get("paths", {})
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, details in methods.items():
            if method.startswith("x-") or not isinstance(details, dict):
                continue
            summary = details.get("summary", details.get("operationId", ""))
            desc = details.get("description", "")[:200]
            params = details.get("parameters", [])
            param_names = [p.get("name", "") for p in params if isinstance(p, dict)]
            line = f"  {method.upper()} {path}"
            if summary:
                line += f" — {summary}"
            if param_names:
                line += f" (params: {', '.join(param_names)})"
            lines.append(line)
            if desc:
                lines.append(f"    {desc}")

    return "\n".join(lines)


def _is_openapi_spec(json_path: Path) -> bool:
    """Quick check if a JSON file looks like an OpenAPI/Swagger spec."""
    try:
        raw = json_path.read_text(encoding="utf-8")[:2000]
        data = json.loads(raw) if raw.strip().startswith("{") else {}
        return bool(
            data.get("openapi") or data.get("swagger") or
            (data.get("info") and data.get("paths"))
        )
    except (OSError, json.JSONDecodeError):
        return False


def ingest_docs(
    docs_dir: str | Path = "docs",
    output_dir: str | Path = "data/docs_processed",
) -> dict[str, int]:
    """Ingest all documentation from docs/ and write processed text files.

    Args:
        docs_dir: Root directory containing domain subdirectories with docs.
        output_dir: Where to write processed text files.

    Returns:
        Dict mapping domain to number of files processed.
    """
    docs_dir = Path(docs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not docs_dir.exists():
        logger.warning("Docs directory does not exist: %s", docs_dir)
        return {}

    stats: dict[str, int] = {}

    for domain_dir in sorted(docs_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain = domain_dir.name
        domain_out = output_dir / domain
        domain_out.mkdir(parents=True, exist_ok=True)
        count = 0

        for fpath in sorted(domain_dir.rglob("*")):
            if not fpath.is_file():
                continue

            suffix = fpath.suffix.lower()
            text = ""

            if suffix == ".pdf":
                text = extract_pdf_text(fpath)
            elif suffix in (".html", ".htm"):
                text = extract_html_text(fpath)
            elif suffix == ".json":
                if _is_openapi_spec(fpath):
                    text = extract_openapi_summary(fpath)
                else:
                    # Plain JSON — just include as-is for context
                    try:
                        text = fpath.read_text(encoding="utf-8")
                    except OSError:
                        continue
            elif suffix in (".txt", ".md", ".xml", ".yaml", ".yml"):
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

            if not text or len(text.strip()) < 20:
                continue

            # Write processed text
            out_name = fpath.relative_to(domain_dir).with_suffix(".txt")
            out_path = domain_out / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            count += 1

        stats[domain] = count
        logger.info("Ingested %d files from domain '%s'", count, domain)

    logger.info("Documentation ingestion complete: %s", stats)
    return stats
