"""GitHub corpus scraper for healthcare integration code.

Clones configured repos, extracts matching files, filters PHI/credentials,
and optionally calls a teacher model to synthesize instructions.
Outputs raw JSONL to data/raw/.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path

import requests

from nexifuse.config import PipelineConfig
from nexifuse.models import ScrapedExample

logger = logging.getLogger(__name__)

# Domain detection heuristics: map filename/content patterns to domains
_DOMAIN_HINTS: list[tuple[str, list[str]]] = [
    ("hl7v2", ["MSH|", "PID|", "OBX|", "ORC|", "ADT", "ORU", "SIU"]),
    ("fhir_r4", ["resourceType", "Bundle", "Patient", "Observation", "CapabilityStatement"]),
    ("mirth", ["<channel>", "<sourceConnector>", "<destinationConnectors>",
               "msg['", 'msg["', "channelMap", "globalMap", "DatabaseConnectionFactory"]),
    ("ehr_api", ["open.epic.com", "fhir.cerner.com", "athenahealth", "meditech", "veradigm"]),
]


def _detect_domain(content: str, file_path: str) -> str:
    """Guess the healthcare domain from file content and path."""
    text = content[:5000].lower()
    path_lower = file_path.lower()

    for domain, hints in _DOMAIN_HINTS:
        for hint in hints:
            if hint.lower() in text or hint.lower() in path_lower:
                return domain
    return "general"


def _clone_or_pull(repo: str, repos_dir: Path) -> Path:
    """Clone a GitHub repo if not present, otherwise pull latest."""
    repo_name = repo.replace("/", "_")
    repo_path = repos_dir / repo_name

    if repo_path.exists() and (repo_path / ".git").exists():
        logger.info("Pulling latest for %s", repo)
        subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(repo_path),
            capture_output=True,
            timeout=120,
        )
    else:
        url = f"https://github.com/{repo}.git"
        logger.info("Cloning %s → %s", url, repo_path)
        repo_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(repo_path)],
            capture_output=True,
            timeout=300,
        )

    return repo_path


def _matches_patterns(file_path: str, patterns: list[str]) -> bool:
    """Check if a file path matches any of the glob patterns."""
    name = Path(file_path).name
    return any(fnmatch(name, pat) for pat in patterns)


def _contains_phi(content: str, phi_patterns: list[str]) -> bool:
    """Return True if content matches any PHI/credential pattern."""
    for pattern in phi_patterns:
        try:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        except re.error:
            logger.warning("Invalid PHI regex pattern: %s", pattern)
    return False


def _synthesize_instruction(
    code: str,
    file_path: str,
    domain: str,
    teacher_endpoint: str,
    teacher_model: str,
) -> str:
    """Call the teacher model to generate an instruction for a code snippet."""
    prompt = (
        f"You are a healthcare integration training data curator.\n"
        f"Given this {domain} code from '{file_path}', write a concise natural language "
        f"instruction that a developer might give to produce this code. "
        f"Return ONLY the instruction, nothing else.\n\n"
        f"```\n{code[:8000]}\n```"
    )

    try:
        resp = requests.post(
            teacher_endpoint,
            json={"model": teacher_model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.warning("Teacher model call failed for %s: %s", file_path, exc)
        return ""


def _collect_files(repo_path: Path, file_patterns: list[str]) -> list[Path]:
    """Walk a repo directory and collect files matching patterns."""
    results = []
    for fpath in repo_path.rglob("*"):
        if fpath.is_file() and _matches_patterns(str(fpath), file_patterns):
            # Skip common non-content dirs
            parts = fpath.relative_to(repo_path).parts
            if any(p in (".git", "node_modules", "__pycache__", ".idea") for p in parts):
                continue
            results.append(fpath)
    return results


def scrape_repos(
    config: PipelineConfig,
    output_path: str | Path = "data/raw/scraped.jsonl",
    repos_dir: str | Path = "data/repos",
    use_teacher: bool = True,
    max_file_size: int = 100_000,
    max_files_per_repo: int = 500,
) -> list[ScrapedExample]:
    """Scrape configured GitHub repos and produce training examples.

    Args:
        config: Pipeline configuration.
        output_path: Where to write the output JSONL.
        repos_dir: Directory to clone repos into.
        use_teacher: Whether to call the teacher model for instruction synthesis.
        max_file_size: Skip files larger than this (bytes).

    Returns:
        List of ScrapedExample objects written to JSONL.
    """
    sc = config.scraper
    tc = config.data_factory
    repos_dir = Path(repos_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples: list[ScrapedExample] = []
    phi_skipped = 0
    size_skipped = 0
    read_errors = 0

    for repo in sc.repos:
        try:
            repo_path = _clone_or_pull(repo, repos_dir)
        except (subprocess.SubprocessError, OSError) as exc:
            logger.error("Failed to clone/pull %s: %s", repo, exc)
            continue

        files = _collect_files(repo_path, sc.file_patterns)
        if len(files) > max_files_per_repo:
            logger.info("Found %d matching files in %s; limiting to %d", len(files), repo, max_files_per_repo)
            files = files[:max_files_per_repo]
        else:
            logger.info("Found %d matching files in %s", len(files), repo)

        for idx, fpath in enumerate(files):
            # Size check
            try:
                if fpath.stat().st_size > max_file_size:
                    size_skipped += 1
                    continue
            except OSError:
                continue

            # Read content
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError) as exc:
                read_errors += 1
                logger.debug("Could not read %s: %s", fpath, exc)
                continue

            # PHI filter
            if _contains_phi(content, sc.phi_patterns):
                phi_skipped += 1
                logger.debug("PHI detected, skipping: %s", fpath)
                continue

            # Skip near-empty files
            if len(content.strip()) < 20:
                continue

            rel_path = str(fpath.relative_to(repo_path))
            domain = _detect_domain(content, rel_path)

            # Synthesize instruction via teacher model
            instruction = ""
            if use_teacher:
                instruction = _synthesize_instruction(
                    content, rel_path, domain,
                    tc.endpoint, tc.model_name,
                )

            if not instruction:
                instruction = f"Explain and reproduce this {domain} integration code from {rel_path}"

            example = ScrapedExample(
                instruction=instruction,
                input="",
                output=content,
                source_repo=repo,
                file_path=rel_path,
                domain=domain,
            )
            examples.append(example)

            if (idx + 1) % 25 == 0:
                logger.info("  [%s] processed %d/%d files", repo, idx + 1, len(files))

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")

    logger.info(
        "Scraping complete: %d examples written to %s "
        "(PHI skipped: %d, size skipped: %d, read errors: %d)",
        len(examples), output_path, phi_skipped, size_skipped, read_errors,
    )
    return examples
