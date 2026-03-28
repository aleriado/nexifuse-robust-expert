"""Multi-GPU parallel healthcare data generation.

Distributes requests across multiple Ollama instances (one per GPU)
using round-robin endpoint selection for maximum throughput.
"""
import json
import logging
import random
import threading
import time
from itertools import cycle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
ENDPOINTS = [f"http://127.0.0.1:{p}/api/generate" for p in range(11434, 11442)]
MODEL = "llama3:8b"
OUTPUT_PATH = Path("data/raw/synthetic_run1.jsonl")
TARGET_PER_DOMAIN = 2100
NUM_WORKERS = 32  # 4 workers per GPU
TIMEOUT = 300
MAX_RETRIES = 3

# Round-robin endpoint iterator (thread-safe)
_endpoint_cycle = cycle(ENDPOINTS)
_endpoint_lock = threading.Lock()

def _next_endpoint() -> str:
    with _endpoint_lock:
        return next(_endpoint_cycle)

# --- Templates (from data_factory) ---
_VIBE_TEMPLATES: dict[str, list[str]] = {
    "hl7v2": [
        "Parse an HL7 v2.x ADT^A01 message and extract patient demographics",
        "Build a Mirth Connect channel that transforms ADT messages to FHIR Patient resources",
        "Write an HL7v2 ACK response generator with proper MSA segment error codes",
        "Create a PID segment builder that handles repeating fields and components",
        "Parse OBX segments from an ORU^R01 and map to FHIR Observation",
    ],
    "fhir_r4": [
        "Create a FHIR R4 Patient resource with extensions for race and ethnicity",
        "Build a FHIR Bundle transaction to create related resources atomically",
        "Write a FHIR search query to find all active MedicationRequests for a patient",
        "Implement FHIR Subscription for real-time ADT notifications",
        "Create a FHIR Provenance resource for audit trail compliance",
    ],
    "mirth": [
        "Write a Mirth Connect JavaScript transformer to parse HL7v2 segments",
        "Create a Mirth channel that routes messages based on MSH-9 event type",
        "Build a Mirth destination that posts FHIR resources to an EHR API",
        "Write a Mirth filter that drops messages missing required fields",
        "Create a Mirth channel for bidirectional ADT sync between two systems",
    ],
    "ehr_api": [
        "Build an Epic FHIR API client for patient search with OAuth2 authentication",
        "Create a Cerner Ignite API integration for lab result retrieval",
        "Write an allscripts API connector with proper token refresh logic",
        "Build a multi-EHR API aggregator that normalizes responses to FHIR R4",
        "Implement SMART on FHIR launch context for an EHR-embedded app",
    ],
    "ihe": [
        "Implement an IHE PIX (Patient Identifier Cross-Reference) query",
        "Build an IHE XDS.b Document Registry query for patient documents",
        "Create an IHE PDQ (Patient Demographics Query) consumer",
        "Write an ATNA audit message for IHE security compliance",
        "Implement IHE MHD (Mobile access to Health Documents) document submission",
    ],
    "dicom": [
        "Build a DICOM C-FIND query for patient study lookup",
        "Create a DICOM WADO-RS client for image retrieval",
        "Write a DICOMweb STOW-RS service for study upload",
        "Implement DICOM Modality Worklist (MWL) integration",
        "Build a DICOM SR (Structured Report) parser for radiology findings",
    ],
}


def _call_teacher(prompt: str, endpoint: str, temperature: float = 0.7) -> str | None:
    """Call Ollama teacher model with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                endpoint,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "24h",
                    "options": {"temperature": temperature},
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.warning("Teacher call failed after %d attempts on %s: %s", MAX_RETRIES, endpoint, e)
                return None
            time.sleep(2 ** attempt)
    return None


def _generate_one(domain: str) -> dict | None:
    """Generate a single training example for a domain."""
    endpoint = _next_endpoint()
    templates = _VIBE_TEMPLATES.get(domain, _VIBE_TEMPLATES["mirth"])
    seed = random.choice(templates)

    # Step 1: Generate instruction
    instruction_prompt = (
        f"You are generating training data for a healthcare integration AI.\n"
        f"Domain: {domain}\n\n"
        f"Inspired by this example: \"{seed}\"\n\n"
        f"Generate a NEW, different instruction in the same domain. "
        f"Be specific and technical. Return ONLY the instruction, nothing else."
    )
    instruction = _call_teacher(instruction_prompt, endpoint)
    if not instruction:
        return None

    # Step 2: Generate response (use potentially different endpoint for load balancing)
    response_endpoint = _next_endpoint()
    response_prompt = (
        f"You are a healthcare integration expert specializing in {domain}.\n\n"
        f"Task: {instruction}\n\n"
        f"Provide a detailed, production-quality response with code examples where appropriate. "
        f"Include error handling, security considerations, and compliance notes."
    )
    output = _call_teacher(response_prompt, response_endpoint, temperature=0.3)
    if not output:
        return None

    return {
        "instruction": instruction,
        "output": output,
        "domain": domain,
        "source_standard": domain,
        "version": "synthetic-v2-8b",
    }


def _count_existing() -> dict[str, int]:
    """Count existing examples per domain."""
    counts: dict[str, int] = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    domain = d.get("domain", "")
                    counts[domain] = counts.get(domain, 0) + 1
                except json.JSONDecodeError:
                    continue
    return counts


def _write_lock():
    """Thread lock for file writing."""
    return threading.Lock()


def main():
    existing = _count_existing()
    logger.info("Existing examples: %s", existing)

    # Build work queue: (domain, index) pairs
    work = []
    for domain in _VIBE_TEMPLATES:
        current = existing.get(domain, 0)
        needed = max(0, TARGET_PER_DOMAIN - current)
        if needed > 0:
            logger.info("Domain '%s': need %d more (have %d)", domain, needed, current)
            work.extend([(domain, i) for i in range(needed)])
        else:
            logger.info("Domain '%s': complete (%d/%d)", domain, current, TARGET_PER_DOMAIN)

    if not work:
        logger.info("All domains complete!")
        return

    random.shuffle(work)  # Mix domains for better GPU utilization
    logger.info("Total to generate: %d examples across %d workers", len(work), NUM_WORKERS)

    file_lock = threading.Lock()
    completed = [0]
    failed = [0]
    total = len(work)

    def process_item(item):
        domain, idx = item
        result = _generate_one(domain)
        if result:
            with file_lock:
                with open(OUTPUT_PATH, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed[0] += 1
                if completed[0] % 25 == 0:
                    logger.info("Progress: %d/%d done (failed: %d)", completed[0], total, failed[0])
        else:
            failed[0] += 1
        return result is not None

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(process_item, item): item for item in work}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error("Worker error: %s", e)
                failed[0] += 1

    logger.info("Generation complete: %d succeeded, %d failed out of %d total", completed[0], failed[0], total)


if __name__ == "__main__":
    main()
