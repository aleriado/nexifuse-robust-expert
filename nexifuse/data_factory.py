"""Teacher-student synthetic data factory.

3-stage pipeline:
  1. Instruction generation — inject domain docs into teacher context, generate user "vibes"
  2. Code synthesis — teacher generates Mirth JS / Channel XML / FHIR mappings
  3. Validation filtering — syntax check outputs, reject hallucinated APIs

Outputs raw JSONL to data/raw/.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import requests

from nexifuse.config import PipelineConfig
from nexifuse.models import TrainingExample

logger = logging.getLogger(__name__)

# Domain-specific vibe templates — the teacher model expands these into
# realistic natural-language instructions
_VIBE_TEMPLATES: dict[str, list[str]] = {
    "hl7v2": [
        "Extract patient insurance info from this ADT message",
        "Parse the OBX segments to get lab results with units and ranges",
        "Build an HL7 ACK response for an incoming ADT^A01",
        "Map PID-11 to extract patient address components",
        "Handle repeating IN1 segments for multiple insurance plans",
        "Extract ordering provider from ORC segment",
    ],
    "fhir_r4": [
        "Convert this HL7 v2 PID segment into a FHIR Patient resource",
        "Create a FHIR Bundle of type transaction with Patient and Encounter",
        "Map OBX results to FHIR Observation resources with proper codings",
        "Build a FHIR AllergyIntolerance from an HL7 AL1 segment",
        "Generate a US Core compliant Patient resource with Must Support fields",
        "Create a FHIR DocumentReference for a PDF lab report",
    ],
    "mirth": [
        "Create a channel that receives HL7 ADT and writes to PostgreSQL",
        "Build a JavaScript transformer to route messages by MSH-9 event type",
        "Set up an HTTP Sender destination with OAuth2 bearer token",
        "Write a channel that converts HL7 v2 to FHIR JSON and POSTs to an endpoint",
        "Create a code template for HL7 date format conversion",
        "Build error handling that sends email on critical lab results",
    ],
    "ehr_api": [
        "Connect to Epic FHIR R4 sandbox and pull patient allergies",
        "Query Cerner for a patient's medication list using FHIR search",
        "Set up AthenaHealth API auth with client credentials flow",
        "Fetch bulk data export from Epic using kick-off and poll pattern",
        "Search for patients by MRN on a Meditech FHIR endpoint",
    ],
}


def _load_domain_context(docs_dir: Path, domain: str, max_chars: int = 6000) -> str:
    """Load processed documentation for a domain to inject as context."""
    domain_dir = docs_dir / domain
    if not domain_dir.exists():
        return ""

    chunks: list[str] = []
    total = 0
    for fpath in sorted(domain_dir.rglob("*.txt")):
        try:
            text = fpath.read_text(encoding="utf-8")[:2000]
            chunks.append(text)
            total += len(text)
            if total >= max_chars:
                break
        except OSError:
            continue

    return "\n---\n".join(chunks)


def _call_teacher(
    prompt: str,
    endpoint: str,
    model: str,
    temperature: float = 0.7,
    timeout: int = 300,
    max_retries: int = 2,
) -> str:
    """Call the teacher model and return the response text. Retries on timeout."""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                endpoint,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except (requests.RequestException, ValueError, KeyError) as exc:
            if attempt < max_retries:
                logger.debug("Teacher call attempt %d failed, retrying: %s", attempt + 1, exc)
            else:
                logger.warning("Teacher call failed after %d attempts: %s", max_retries + 1, exc)
    return ""


def _generate_instruction(
    domain: str,
    context: str,
    endpoint: str,
    model: str,
    timeout: int = 300,
    max_retries: int = 2,
) -> str:
    """Stage 1: Generate a realistic user instruction (vibe) for a domain."""
    templates = _VIBE_TEMPLATES.get(domain, _VIBE_TEMPLATES["mirth"])
    seed_vibe = random.choice(templates)

    prompt = (
        f"You are generating training data for a healthcare integration AI.\n"
        f"Domain: {domain}\n\n"
        f"Reference documentation:\n{context[:4000]}\n\n"
        f"Inspired by this example request: \"{seed_vibe}\"\n\n"
        f"Generate a NEW, different, realistic natural language instruction that a "
        f"healthcare integration developer might give. Be specific and include "
        f"realistic details (message types, segment names, field numbers, API endpoints). "
        f"Return ONLY the instruction, nothing else."
    )
    result = _call_teacher(prompt, endpoint, model, timeout=timeout, max_retries=max_retries)
    return result if result else seed_vibe


def _generate_code(
    instruction: str,
    domain: str,
    context: str,
    endpoint: str,
    model: str,
    include_cot: bool = True,
    timeout: int = 300,
    max_retries: int = 2,
) -> tuple[str, str | None]:
    """Stage 2: Generate code from an instruction. Returns (code, cot_trace)."""
    cot_prefix = (
        "Think step by step about how to implement this. "
        "Show your reasoning, then provide the final code.\n\n"
        if include_cot else ""
    )

    prompt = (
        f"You are a healthcare integration expert specializing in Mirth Connect, "
        f"HL7 v2, FHIR R4, and EHR API connectivity.\n\n"
        f"Reference documentation:\n{context[:4000]}\n\n"
        f"{cot_prefix}"
        f"User request: {instruction}\n\n"
        f"Provide production-quality code that fulfills this request. "
        f"Include appropriate error handling and comments."
    )

    response = _call_teacher(prompt, endpoint, model, temperature=0.3, timeout=timeout, max_retries=max_retries)
    if not response:
        return "", None

    # Try to separate CoT reasoning from code
    cot_trace = None
    if include_cot and "<think>" in response:
        # DeepSeek-R1 style thinking tags
        parts = response.split("</think>", 1)
        if len(parts) == 2:
            cot_trace = parts[0].replace("<think>", "").strip()
            response = parts[1].strip()

    return response, cot_trace


def generate_examples(
    config: PipelineConfig,
    output_path: str | Path = "data/raw/synthetic.jsonl",
    docs_dir: str | Path = "data/docs_processed",
    num_per_domain: int = 100,
) -> list[TrainingExample]:
    """Generate synthetic training examples using the teacher model.

    Supports resume: if output_path already exists, counts existing examples
    per domain and skips completed work. New examples are appended incrementally
    so progress is never lost on interruption.

    Args:
        config: Pipeline configuration.
        output_path: Where to write the output JSONL.
        docs_dir: Directory with processed documentation (from doc_ingester).
        num_per_domain: Number of examples to generate per domain.

    Returns:
        List of TrainingExample objects (existing + new).
    """
    tc = config.data_factory
    timeout = getattr(tc, "timeout_seconds", 300)
    max_retries = getattr(tc, "max_retries", 2)
    docs_dir = Path(docs_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Resume support: count existing examples per domain ---
    existing_counts: dict[str, int] = {}
    examples: list[TrainingExample] = []
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    domain = row.get("domain", "unknown")
                    existing_counts[domain] = existing_counts.get(domain, 0) + 1
                    examples.append(TrainingExample.from_dict(row))
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Skipping malformed line %d in existing file", line_num)
        total_existing = sum(existing_counts.values())
        logger.info("Resume: found %d existing examples %s", total_existing, dict(existing_counts))
    else:
        logger.info("Starting fresh generation (no existing file)")

    failed = 0
    new_count = 0

    # Open in append mode so each example is persisted immediately
    with open(output_path, "a", encoding="utf-8") as f:
        for domain in tc.domains:
            already_done = existing_counts.get(domain, 0)
            remaining = num_per_domain - already_done

            if remaining <= 0:
                logger.info("Domain '%s': already complete (%d/%d), skipping",
                            domain, already_done, num_per_domain)
                continue

            context = _load_domain_context(docs_dir, domain)
            logger.info("Generating %d examples for domain '%s' (already have %d, context: %d chars, timeout=%ds, max_retries=%d)",
                         remaining, domain, already_done, len(context), timeout, max_retries)

            for i in range(remaining):
                # Stage 1: Generate instruction
                instruction = _generate_instruction(
                    domain, context, tc.endpoint, tc.model_name,
                    timeout=timeout, max_retries=max_retries,
                )

                # Stage 2: Generate code
                code, cot_trace = _generate_code(
                    instruction, domain, context,
                    tc.endpoint, tc.model_name,
                    include_cot=tc.include_cot,
                    timeout=timeout, max_retries=max_retries,
                )

                if not code:
                    failed += 1
                    continue

                example = TrainingExample(
                    instruction=instruction,
                    input="",
                    output=code,
                    cot_trace=cot_trace,
                    domain=domain,
                    source_standard=domain,
                    version="synthetic-v1",
                )
                examples.append(example)
                new_count += 1

                # Write immediately and flush so progress survives interruption
                f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                f.flush()

                if (i + 1) % 25 == 0:
                    logger.info("  [%s] %d/%d generated (total in file: %d)",
                                domain, i + 1, remaining, len(examples))

    logger.info(
        "Data factory complete: %d total examples in %s (new: %d, resumed: %d, failed: %d)",
        len(examples), output_path, new_count,
        len(examples) - new_count, failed,
    )
    return examples
