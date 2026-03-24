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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from nexifuse.config import PipelineConfig
from nexifuse.models import ConversationExample, ConversationTurn, TrainingExample
from nexifuse.validator import validate_example

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

# ---------------------------------------------------------------------------
# General-purpose vibe templates — for restoring general assistant capability
# ---------------------------------------------------------------------------
_GENERAL_VIBE_TEMPLATES: dict[str, list[str]] = {
    "math": [
        "What is 247 + 389?",
        "Calculate 15% tip on a $67.50 bill",
        "Convert 72 degrees Fahrenheit to Celsius",
        "If a meeting starts at 2:45 PM and lasts 1 hour 20 minutes, when does it end?",
        "What is the area of a circle with radius 5?",
        "Solve: 3x + 7 = 22",
        "What is 12 factorial?",
        "A store has a 30% off sale. Original price is $85. What do you pay?",
    ],
    "general_knowledge": [
        "What is the difference between TCP and UDP?",
        "Explain how DNS resolution works",
        "What are the main differences between SQL and NoSQL databases?",
        "How does HTTPS encryption work?",
        "What is the CAP theorem?",
        "Explain what a hash table is and how it works",
        "What is the difference between a process and a thread?",
    ],
    "casual": [
        "Tell me a fun fact",
        "What should I consider when choosing a programming language for a new project?",
        "How do I stay productive while working remotely?",
        "What are some good practices for code reviews?",
        "Can you summarize the main ideas of clean code?",
        "What is pair programming and when is it useful?",
        "How do I write a good commit message?",
    ],
    "general_coding": [
        "Write a Python function to check if a string is a palindrome",
        "How do I read a CSV file in Python?",
        "Write a SQL query to find duplicate rows in a table",
        "Explain the difference between let, const, and var in JavaScript",
        "How do I handle exceptions in Python?",
        "Write a function to merge two sorted arrays",
        "What is a decorator in Python? Give an example",
        "Write a bash script that finds all files larger than 100MB",
    ],
    "reasoning": [
        "Compare REST APIs vs GraphQL - when would you choose each?",
        "Explain the pros and cons of microservices vs monolithic architecture",
        "What are the trade-offs of using an ORM vs raw SQL?",
        "Why is caching important and what are common caching strategies?",
        "When should you use a queue vs direct API calls?",
        "Explain eventual consistency vs strong consistency",
    ],
}

# ---------------------------------------------------------------------------
# Conceptual/explanation vibe templates — for non-code knowledge
# ---------------------------------------------------------------------------
_CONCEPTUAL_VIBE_TEMPLATES: dict[str, list[str]] = {
    "hl7v2_vs_fhir": [
        "What are the key differences between HL7 v2 and FHIR R4?",
        "When should a hospital migrate from HL7 v2 to FHIR?",
        "Explain the HL7 v2 message structure vs FHIR resource model",
        "How do HL7 v2 segments map to FHIR resources?",
        "What challenges arise when bridging HL7 v2 and FHIR systems?",
    ],
    "mirth_architecture": [
        "Explain the Mirth Connect channel processing pipeline",
        "How do Mirth source and destination connectors differ?",
        "What are Mirth code templates and when should they be used?",
        "Describe the Mirth message lifecycle from source to destination",
        "How does Mirth handle message queuing and retry logic?",
    ],
    "ihe_profiles": [
        "Explain the IHE PIX/PDQ profile and its use cases",
        "What is the IHE XDS.b profile and how does it support document sharing?",
        "Describe the IHE ATNA profile for audit logging",
        "How does IHE MHD relate to FHIR document sharing?",
        "What role does IHE CT (Consistent Time) play in interoperability?",
    ],
    "ehr_integration_patterns": [
        "Describe common patterns for real-time EHR integration",
        "What is the difference between point-to-point and hub-and-spoke integration?",
        "Explain the role of an integration engine in a hospital IT ecosystem",
        "How do you handle data reconciliation between EHR systems?",
        "What are best practices for EHR API rate limiting and throttling?",
    ],
    "security_compliance": [
        "What are the HIPAA requirements for healthcare data integration?",
        "How should PHI be handled in HL7 message logging?",
        "Explain TLS/mTLS requirements for healthcare API connections",
        "What audit logging is required for HIPAA compliance in integration engines?",
        "How do you implement data masking for PHI in development environments?",
    ],
    "fhir_resource_design": [
        "How do you design custom FHIR profiles for a specific use case?",
        "Explain FHIR extensions and when to create them",
        "What is the difference between FHIR contained resources and references?",
        "How do you model complex clinical workflows in FHIR?",
        "What are FHIR Implementation Guides and why are they important?",
    ],
}

# ---------------------------------------------------------------------------
# Raw HL7 message types for generation
# ---------------------------------------------------------------------------
_RAW_HL7_TYPES: list[str] = ["ADT", "ORU", "ORM", "SIU", "VXU"]

# ---------------------------------------------------------------------------
# Multi-turn conversation scenario outlines
# ---------------------------------------------------------------------------
_MULTITURN_SCENARIOS: dict[str, list[dict]] = {
    "debugging": [
        {
            "description": "Mirth transformer TypeError on null PID-3",
            "domain": "mirth",
            "turns_outline": [
                {"role": "user", "summary": "My Mirth JavaScript transformer throws a TypeError when PID-3 is empty in some ADT messages. Here is my code: var mrn = msg['PID']['PID.3']['PID.3.1'].toString();"},
                {"role": "assistant", "summary": "Identify the null-check issue, explain why PID-3 can be absent, provide a safe accessor with null check"},
                {"role": "user", "summary": "Thanks! But some messages have repeating PID-3 fields. How do I handle that?"},
                {"role": "assistant", "summary": "Show how to iterate over PID-3 repetitions safely using a for loop on the field length"},
            ],
        },
        {
            "description": "FHIR Bundle POST returning 422 Unprocessable Entity",
            "domain": "fhir_r4",
            "turns_outline": [
                {"role": "user", "summary": "I'm POSTing a FHIR transaction Bundle but getting 422 Unprocessable Entity. My Bundle has Patient and Encounter resources."},
                {"role": "assistant", "summary": "Ask if they included request.method and request.url in each entry, which are required for transaction Bundles"},
                {"role": "user", "summary": "I didn't include request.method! I added it and now Patient works but Encounter still fails."},
                {"role": "assistant", "summary": "Explain that Encounter needs a reference to Patient using a fullUrl temporary UUID, and show the correct Bundle structure"},
            ],
        },
        {
            "description": "HL7 segment index out of range with optional NK1",
            "domain": "hl7v2",
            "turns_outline": [
                {"role": "user", "summary": "My HL7 parser crashes with 'segment index out of range' when processing some ADT^A01 messages. It works for most messages."},
                {"role": "assistant", "summary": "Ask whether the failing messages are missing optional segments like NK1 or IN1, and suggest checking segment existence before accessing"},
                {"role": "user", "summary": "You're right, the failing messages don't have NK1 segments. How should I handle optional segments?"},
                {"role": "assistant", "summary": "Show a pattern to check if a segment exists before accessing it, and explain which ADT segments are optional vs required"},
            ],
        },
        {
            "description": "Python script fails to connect to Epic FHIR sandbox",
            "domain": "ehr_api",
            "turns_outline": [
                {"role": "user", "summary": "My Python script gets a 401 Unauthorized when trying to access Epic's FHIR R4 sandbox. I'm using requests with a Bearer token."},
                {"role": "assistant", "summary": "Ask about the auth method — Epic SMART Backend Services requires JWT-based auth, not simple Bearer tokens. Explain the difference."},
                {"role": "user", "summary": "I was using a static token. How do I implement the JWT-based SMART Backend Services flow?"},
                {"role": "assistant", "summary": "Walk through the steps: create a JWT with client_id, sign with private key, POST to token endpoint, use the returned access_token"},
            ],
        },
        {
            "description": "General Python debugging - list index error",
            "domain": "general_coding",
            "turns_outline": [
                {"role": "user", "summary": "My Python script crashes with IndexError: list index out of range when processing a CSV file. Not all rows have the same number of columns."},
                {"role": "assistant", "summary": "Explain that CSV rows can have variable column counts, suggest checking length before access or using csv.DictReader"},
                {"role": "user", "summary": "DictReader works great. But now some values are empty strings and my int() conversion fails. How do I handle missing values?"},
                {"role": "assistant", "summary": "Show a safe conversion pattern with a default value, or suggest using pandas for robust CSV handling with NaN support"},
            ],
        },
    ],
    "clarification": [
        {
            "description": "Vague FHIR conversion request",
            "domain": "fhir_r4",
            "turns_outline": [
                {"role": "user", "summary": "Convert this message to FHIR."},
                {"role": "assistant", "summary": "Ask what type of message (HL7 v2 ADT, ORU, ORM?), which segments to map, and which FHIR resources they need"},
                {"role": "user", "summary": "It's an ADT^A01. I need Patient and Encounter resources from PID and PV1 segments."},
                {"role": "assistant", "summary": "Provide the complete mapping code for PID -> FHIR Patient and PV1 -> FHIR Encounter with proper field mappings"},
            ],
        },
        {
            "description": "Vague HL7 listener request",
            "domain": "mirth",
            "turns_outline": [
                {"role": "user", "summary": "Set up an HL7 listener for me."},
                {"role": "assistant", "summary": "Ask about transport protocol (MLLP vs HTTP), port number, TLS requirements, and which message types to accept"},
                {"role": "user", "summary": "MLLP on port 6661, no TLS for now, ADT messages only."},
                {"role": "assistant", "summary": "Provide Mirth channel configuration with TCP Listener source connector on port 6661 with an MSH-9 filter for ADT messages"},
            ],
        },
        {
            "description": "Vague database request",
            "domain": "general_coding",
            "turns_outline": [
                {"role": "user", "summary": "Help me set up a database."},
                {"role": "assistant", "summary": "Ask what kind of data they're storing, expected volume, whether they need relational or document store, and what tech stack they're using"},
                {"role": "user", "summary": "I need to store patient appointment data, maybe 10k records, I'm using Python. Relational makes sense."},
                {"role": "assistant", "summary": "Recommend SQLite for simplicity at that scale, show a Python example with sqlite3 module creating an appointments table with proper schema"},
            ],
        },
    ],
    "iterative": [
        {
            "description": "Build up a Mirth ADT channel step by step",
            "domain": "mirth",
            "turns_outline": [
                {"role": "user", "summary": "Create a basic Mirth channel that receives HL7 ADT messages on MLLP port 6661."},
                {"role": "assistant", "summary": "Provide a basic channel XML with TCP Listener source connector on port 6661"},
                {"role": "user", "summary": "Now add a JavaScript transformer that routes A01 events to one destination and A08 to another."},
                {"role": "assistant", "summary": "Add a transformer step that reads MSH-9 and uses router.routeMessage() to send to different destinations"},
                {"role": "user", "summary": "Good. Now add PostgreSQL logging for all A01 admissions — save patient MRN and admission date."},
                {"role": "assistant", "summary": "Add a Database Writer destination with INSERT statement extracting PID-3 and PV1-44 into a patients table"},
                {"role": "user", "summary": "Finally, add error handling that sends an email alert if the database write fails."},
                {"role": "assistant", "summary": "Wrap the DB destination in a response transformer with error detection, add an SMTP destination for failures with alert details"},
            ],
        },
        {
            "description": "Build a FHIR Patient resource incrementally",
            "domain": "fhir_r4",
            "turns_outline": [
                {"role": "user", "summary": "Build a FHIR Patient resource from an HL7 PID segment with name and DOB."},
                {"role": "assistant", "summary": "Provide basic Patient with HumanName from PID-5 and birthDate from PID-7"},
                {"role": "user", "summary": "Add multiple identifiers from PID-3 repeating fields — MRN and SSN."},
                {"role": "assistant", "summary": "Add identifier array with MRN (type MR) and SSN (type SS) from PID-3 repetitions"},
                {"role": "user", "summary": "Now add US Core required extensions for race and ethnicity."},
                {"role": "assistant", "summary": "Add the us-core-race and us-core-ethnicity extensions with proper ombCategory coding"},
                {"role": "user", "summary": "Wrap it all in a Transaction Bundle with the Patient and a linked Encounter."},
                {"role": "assistant", "summary": "Create a Bundle of type transaction with Patient and Encounter entries, using fullUrl UUIDs for internal references"},
            ],
        },
        {
            "description": "Build a REST API incrementally",
            "domain": "general_coding",
            "turns_outline": [
                {"role": "user", "summary": "Create a basic Python Flask REST API with a /health endpoint."},
                {"role": "assistant", "summary": "Provide minimal Flask app with /health GET endpoint returning 200 OK"},
                {"role": "user", "summary": "Add a POST /users endpoint that accepts JSON with name and email."},
                {"role": "assistant", "summary": "Add the endpoint with request.get_json() parsing and basic validation"},
                {"role": "user", "summary": "Add input validation — name is required and email must be valid format."},
                {"role": "assistant", "summary": "Add validation with proper error responses (400) for missing name and invalid email regex"},
            ],
        },
    ],
    "code_review": [
        {
            "description": "Review Mirth transformer with issues",
            "domain": "mirth",
            "turns_outline": [
                {"role": "user", "summary": "Review this Mirth transformer: var pid3 = msg['PID']['PID.3']['PID.3.1'].toString(); channelMap.put('mrn', pid3); var name = msg['PID']['PID.5']['PID.5.1'].toString() + ' ' + msg['PID']['PID.5']['PID.5.2'].toString();"},
                {"role": "assistant", "summary": "Identify issues: no null checks (will crash on empty fields), no try/catch, hardcoded field access without repetition handling, name concatenation should use family/given order"},
                {"role": "user", "summary": "Good catch on the null checks. Can you show me the improved version?"},
                {"role": "assistant", "summary": "Provide the refactored code with null-safe accessors, try/catch, proper repetition handling, and correct name formatting"},
            ],
        },
        {
            "description": "Review Python code with inefficiencies",
            "domain": "general_coding",
            "turns_outline": [
                {"role": "user", "summary": "Review this Python function that finds duplicate items in a list using nested for loops."},
                {"role": "assistant", "summary": "Identify the O(n^2) complexity, suggest using a set or Counter for O(n) solution, note the function also modifies the input list which is a side effect"},
                {"role": "user", "summary": "Can you show me the improved version using Counter?"},
                {"role": "assistant", "summary": "Provide the optimized version using collections.Counter with explanation of the improvement"},
            ],
        },
    ],
    "migration": [
        {
            "description": "HL7 v2 to FHIR migration guidance",
            "domain": "fhir_r4",
            "turns_outline": [
                {"role": "user", "summary": "We're migrating our HL7 v2 ADT feeds to FHIR R4. Where do we start?"},
                {"role": "assistant", "summary": "Outline the migration: map ADT events to FHIR resources (PID->Patient, PV1->Encounter, NK1->RelatedPerson), identify what gets lost and what needs extensions"},
                {"role": "user", "summary": "What about custom Z-segments we use for internal routing? Those don't map to FHIR."},
                {"role": "assistant", "summary": "Explain options: FHIR extensions for data that needs to persist, message headers/meta for routing info, or a hybrid approach keeping Z-segment data in a custom extension namespace"},
                {"role": "user", "summary": "Should we do a big-bang cutover or run both in parallel?"},
                {"role": "assistant", "summary": "Recommend parallel running with a Mirth channel that fans out to both HL7 and FHIR destinations, with comparison logging to verify parity before cutover"},
            ],
        },
    ],
    "walkthrough": [
        {
            "description": "FHIR bulk data export walkthrough",
            "domain": "ehr_api",
            "turns_outline": [
                {"role": "user", "summary": "Walk me through implementing a FHIR bulk data export client for Epic."},
                {"role": "assistant", "summary": "Explain the 3-step pattern: kick-off request (GET /$export with _type parameter), poll the Content-Location URL for completion, download the NDJSON files"},
                {"role": "user", "summary": "How do I handle the polling? What status codes should I expect?"},
                {"role": "assistant", "summary": "Explain: 202 Accepted means still processing (check Retry-After header), 200 OK means complete (body contains download URLs), 4xx/5xx means error. Show polling loop with exponential backoff."},
                {"role": "user", "summary": "And how do I parse the NDJSON response files?"},
                {"role": "assistant", "summary": "Show Python code to download each file URL, parse line-by-line as JSON (each line is one FHIR resource), and process/store the resources with proper error handling for malformed lines"},
            ],
        },
        {
            "description": "Setting up a Python project walkthrough",
            "domain": "general_coding",
            "turns_outline": [
                {"role": "user", "summary": "Walk me through setting up a new Python project with proper structure."},
                {"role": "assistant", "summary": "Explain: create project directory, set up virtual environment, create src/ and tests/ layout, add pyproject.toml or setup.py, init git"},
                {"role": "user", "summary": "How should I manage dependencies?"},
                {"role": "assistant", "summary": "Recommend using a requirements.txt for simple projects or pyproject.toml with pip-tools for dependency pinning. Show the workflow: add to requirements.in, compile to requirements.txt, install."},
                {"role": "user", "summary": "What about testing setup?"},
                {"role": "assistant", "summary": "Show pytest configuration in pyproject.toml, create a sample test file, explain test discovery conventions, recommend pytest-cov for coverage reporting"},
            ],
        },
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
                    "keep_alive": "24h",
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


def _select_model(config: PipelineConfig, complex_task: bool = True) -> str:
    """Pick the right teacher model based on task complexity.

    Uses config.data_factory.model_name (primary teacher) for complex tasks
    like multi-turn conversations, CoT reasoning, and rejection sampling.
    Uses config.data_factory.bulk_model for simpler tasks like general Q&A
    and straightforward single-turn generation.
    """
    tc = config.data_factory
    if complex_task:
        return tc.model_name
    bulk = getattr(tc, "bulk_model", None)
    return bulk if bulk else tc.model_name


def _score_example(example: TrainingExample, config: PipelineConfig | None = None) -> float:
    """Score a training example using the validator. Returns 0.0-1.0."""
    result = validate_example(example, config)
    if not result.passed:
        return 0.0
    # Base score for passing validation
    score = 0.5
    output = example.output.strip()
    # Reward longer, more detailed outputs (up to a point)
    length = len(output)
    if length > 100:
        score += 0.15
    if length > 500:
        score += 0.1
    # Reward code blocks / structured content
    if "```" in output or "function " in output or "def " in output:
        score += 0.1
    # Reward comments / documentation
    if "//" in output or "#" in output or "/*" in output:
        score += 0.1
    # Penalize very short outputs
    if length < 50:
        score -= 0.2
    return max(0.0, min(1.0, score))


def _generate_with_rejection(
    prompt: str,
    domain: str,
    config: PipelineConfig,
    endpoint: str,
    model: str,
    simpo_path: Path | None = None,
    simpo_lock: threading.Lock | None = None,
    temperature: float = 0.7,
    timeout: int = 300,
    max_retries: int = 2,
) -> str:
    """Generate N completions via rejection sampling, return the best one.

    Also saves a SimPO preference pair (best vs worst) if simpo_path is set.

    Args:
        prompt: The generation prompt.
        domain: Domain for building TrainingExample for scoring.
        config: Pipeline config (used for rejection_samples count and validation).
        endpoint: Teacher model endpoint.
        model: Model name to use.
        simpo_path: Path to write SimPO pairs (optional).
        simpo_lock: Lock for thread-safe writes to simpo_path (optional).
        temperature: Sampling temperature.
        timeout: Request timeout.
        max_retries: Max retries per call.

    Returns:
        The best completion string, or empty string if all fail.
    """
    n_samples = getattr(config.data_factory, "rejection_samples", 5)
    if n_samples <= 1:
        return _call_teacher(prompt, endpoint, model, temperature=temperature,
                             timeout=timeout, max_retries=max_retries)

    completions: list[tuple[str, float]] = []
    for i in range(n_samples):
        # Vary temperature slightly for diversity
        temp_i = temperature + (i * 0.05)
        response = _call_teacher(prompt, endpoint, model, temperature=temp_i,
                                 timeout=timeout, max_retries=max_retries)
        if not response:
            continue
        # Score using validator
        example = TrainingExample(
            instruction=prompt, input="", output=response,
            domain=domain, source_standard=domain, version="rejection-v1",
        )
        score = _score_example(example, config)
        completions.append((response, score))

    if not completions:
        return ""

    # Sort by score descending
    completions.sort(key=lambda x: x[1], reverse=True)
    best_response, best_score = completions[0]
    worst_response, worst_score = completions[-1]

    logger.debug(
        "Rejection sampling: %d/%d valid, best=%.2f, worst=%.2f",
        len(completions), n_samples, best_score, worst_score,
    )

    # Save SimPO preference pair if we have distinct best/worst
    if (
        simpo_path is not None
        and simpo_lock is not None
        and len(completions) >= 2
        and best_score > worst_score
    ):
        pair = {
            "prompt": prompt,
            "chosen": best_response,
            "rejected": worst_response,
        }
        with simpo_lock:
            simpo_path.parent.mkdir(parents=True, exist_ok=True)
            with open(simpo_path, "a", encoding="utf-8") as sf:
                sf.write(json.dumps(pair, ensure_ascii=False) + "\n")
                sf.flush()

    return best_response


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
    num_workers: int = 8,
) -> list[TrainingExample]:
    """Generate synthetic training examples using the teacher model.

    Supports resume: if output_path already exists, counts existing examples
    per domain and skips completed work. New examples are appended incrementally
    so progress is never lost on interruption.

    Uses ThreadPoolExecutor with num_workers parallel requests to maximize
    GPU utilization across all available GPUs.

    Args:
        config: Pipeline configuration.
        output_path: Where to write the output JSONL.
        docs_dir: Directory with processed documentation (from doc_ingester).
        num_per_domain: Number of examples to generate per domain.
        num_workers: Number of parallel generation workers (default 8 for 8 GPUs).

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
    write_lock = threading.Lock()

    def _generate_one(domain: str, context: str) -> TrainingExample | None:
        """Generate a single example (runs in thread pool)."""
        instruction = _generate_instruction(
            domain, context, tc.endpoint, tc.model_name,
            timeout=timeout, max_retries=max_retries,
        )
        code, cot_trace = _generate_code(
            instruction, domain, context,
            tc.endpoint, tc.model_name,
            include_cot=tc.include_cot,
            timeout=timeout, max_retries=max_retries,
        )
        if not code:
            return None
        return TrainingExample(
            instruction=instruction, input="", output=code,
            cot_trace=cot_trace, domain=domain,
            source_standard=domain, version="synthetic-v1",
        )

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
            logger.info("Generating %d examples for domain '%s' with %d workers (already have %d)",
                         remaining, domain, num_workers, already_done)

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(_generate_one, domain, context): i
                    for i in range(remaining)
                }
                done_count = 0
                for future in as_completed(futures):
                    done_count += 1
                    example = future.result()
                    if example is None:
                        failed += 1
                        continue

                    with write_lock:
                        examples.append(example)
                        new_count += 1
                        f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                        f.flush()

                    if done_count % 25 == 0:
                        logger.info("  [%s] %d/%d done (total in file: %d)",
                                    domain, done_count, remaining, len(examples))

    logger.info(
        "Data factory complete: %d total examples in %s (new: %d, resumed: %d, failed: %d)",
        len(examples), output_path, new_count,
        len(examples) - new_count, failed,
    )
    return examples


# ---------------------------------------------------------------------------
# General-purpose data generation
# ---------------------------------------------------------------------------

def _generate_general_instruction(
    category: str,
    endpoint: str,
    model: str,
    timeout: int = 300,
    max_retries: int = 2,
) -> str:
    """Generate a general-purpose instruction for a category."""
    templates = _GENERAL_VIBE_TEMPLATES.get(category, _GENERAL_VIBE_TEMPLATES["casual"])
    seed = random.choice(templates)

    prompt = (
        f"You are generating diverse training data for an AI assistant.\n"
        f"Category: {category}\n\n"
        f"Inspired by this example: \"{seed}\"\n\n"
        f"Generate a NEW, different question or instruction in the same category. "
        f"Be specific and clear. Return ONLY the question/instruction, nothing else."
    )
    result = _call_teacher(prompt, endpoint, model, timeout=timeout, max_retries=max_retries)
    return result if result else seed


def _generate_general_response(
    instruction: str,
    endpoint: str,
    model: str,
    timeout: int = 300,
    max_retries: int = 2,
) -> str:
    """Generate a response to a general-purpose instruction."""
    prompt = (
        f"You are a helpful AI assistant. Answer the following clearly and concisely.\n\n"
        f"Question: {instruction}\n\n"
        f"Provide a helpful, accurate response."
    )
    return _call_teacher(prompt, endpoint, model, temperature=0.3, timeout=timeout, max_retries=max_retries)


def generate_general_examples(
    config: PipelineConfig,
    output_path: str | Path = "data/raw/general.jsonl",
    num_per_category: int = 1500,
    num_workers: int = 8,
    model_override: str | None = None,
) -> list[TrainingExample]:
    """Generate general-purpose training examples (math, coding, chat, reasoning).

    Uses ThreadPoolExecutor with num_workers parallel requests.
    """
    tc = config.data_factory
    model_name = model_override or tc.model_name
    timeout = getattr(tc, "timeout_seconds", 300)
    max_retries = getattr(tc, "max_retries", 2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Using model: %s for general data generation", model_name)

    # Resume support
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
                    cat = row.get("domain", "unknown")
                    existing_counts[cat] = existing_counts.get(cat, 0) + 1
                    examples.append(TrainingExample.from_dict(row))
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Skipping malformed line %d in %s", line_num, output_path)
        logger.info("Resume: found %d existing general examples %s", len(examples), dict(existing_counts))

    failed = 0
    new_count = 0
    write_lock = threading.Lock()

    def _generate_one(category: str) -> TrainingExample | None:
        """Generate a single general example (runs in thread pool)."""
        instruction = _generate_general_instruction(
            category, tc.endpoint, model_name,
            timeout=timeout, max_retries=max_retries,
        )
        response = _generate_general_response(
            instruction, tc.endpoint, model_name,
            timeout=timeout, max_retries=max_retries,
        )
        if not response:
            return None
        return TrainingExample(
            instruction=instruction, input="", output=response,
            domain=category, source_standard=category, version="general-v1",
        )

    with open(output_path, "a", encoding="utf-8") as f:
        for category in _GENERAL_VIBE_TEMPLATES:
            already_done = existing_counts.get(category, 0)
            remaining = num_per_category - already_done

            if remaining <= 0:
                logger.info("Category '%s': already complete (%d/%d), skipping",
                            category, already_done, num_per_category)
                continue

            logger.info("Generating %d general examples for '%s' with %d workers (already have %d)",
                        remaining, category, num_workers, already_done)

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(_generate_one, category): i
                    for i in range(remaining)
                }
                done_count = 0
                for future in as_completed(futures):
                    done_count += 1
                    example = future.result()
                    if example is None:
                        failed += 1
                        continue

                    with write_lock:
                        examples.append(example)
                        new_count += 1
                        f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                        f.flush()

                    if done_count % 25 == 0:
                        logger.info("  [%s] %d/%d done", category, done_count, remaining)

    logger.info(
        "General data factory complete: %d total in %s (new: %d, failed: %d)",
        len(examples), output_path, new_count, failed,
    )
    return examples


# ---------------------------------------------------------------------------
# Multi-turn conversation generation
# ---------------------------------------------------------------------------

def _generate_conversation(
    scenario: dict,
    endpoint: str,
    model: str,
    timeout: int = 300,
    max_retries: int = 2,
) -> ConversationExample | None:
    """Generate a single multi-turn conversation from a scenario outline.

    Two-phase approach:
      1. Generate expanded conversation from the outline in one pass
      2. Parse the response into individual turns
    """
    turns_outline = scenario["turns_outline"]
    domain = scenario.get("domain", "general")
    description = scenario.get("description", "")

    # Build the generation prompt with the outline
    outline_text = ""
    for t in turns_outline:
        outline_text += f"  {t['role'].upper()}: {t['summary']}\n"

    prompt = (
        f"You are generating a realistic multi-turn conversation for training a healthcare integration AI assistant.\n\n"
        f"Scenario: {description}\n"
        f"Domain: {domain}\n\n"
        f"Conversation outline:\n{outline_text}\n"
        f"Generate a complete, realistic conversation following this outline. "
        f"Each turn should be substantial and detailed. "
        f"For assistant turns that involve code, include actual working code.\n\n"
        f"Format your response EXACTLY as:\n"
        f"USER: <user message>\n"
        f"ASSISTANT: <assistant response>\n"
        f"USER: <user message>\n"
        f"ASSISTANT: <assistant response>\n"
        f"... and so on for all turns.\n\n"
        f"Start now:"
    )

    response = _call_teacher(
        prompt, endpoint, model, temperature=0.5,
        timeout=timeout, max_retries=max_retries,
    )
    if not response:
        return None

    # Parse the response into turns
    turns: list[ConversationTurn] = []
    current_role = None
    current_content: list[str] = []

    for line in response.split("\n"):
        stripped = line.strip()

        # Detect role markers
        if stripped.upper().startswith("USER:"):
            if current_role and current_content:
                turns.append(ConversationTurn(
                    role=current_role,
                    content="\n".join(current_content).strip(),
                ))
            current_role = "user"
            current_content = [stripped[5:].strip()]
        elif stripped.upper().startswith("ASSISTANT:"):
            if current_role and current_content:
                turns.append(ConversationTurn(
                    role=current_role,
                    content="\n".join(current_content).strip(),
                ))
            current_role = "assistant"
            current_content = [stripped[10:].strip()]
        elif current_role:
            current_content.append(line)

    # Flush last turn
    if current_role and current_content:
        turns.append(ConversationTurn(
            role=current_role,
            content="\n".join(current_content).strip(),
        ))

    # Validate: need at least 2 turns, must start with user, end with assistant
    if len(turns) < 2:
        return None
    if turns[0].role != "user":
        return None
    if turns[-1].role != "assistant":
        # Drop the last incomplete user turn
        turns = turns[:-1]
    if len(turns) < 2:
        return None
    # Reject any turn with empty content
    if any(not t.content.strip() for t in turns):
        return None

    return ConversationExample(
        turns=turns,
        domain=domain,
        scenario_type=next(
            (stype for stype, scenarios in _MULTITURN_SCENARIOS.items()
             if scenario in scenarios),
            "unknown",
        ),
        source_standard=domain,
        version="conversation-v1",
    )


def generate_conversations(
    config: PipelineConfig,
    output_path: str | Path = "data/raw/conversations.jsonl",
    num_per_scenario_domain: int = 70,
    num_workers: int = 8,
    model_override: str | None = None,
) -> list[ConversationExample]:
    """Generate multi-turn conversation training examples.

    For each scenario in _MULTITURN_SCENARIOS, generates num_per_scenario_domain
    variations. Uses ThreadPoolExecutor with num_workers parallel requests.
    """
    tc = config.data_factory
    model_name = model_override or tc.model_name
    timeout = getattr(tc, "timeout_seconds", 300)
    max_retries = getattr(tc, "max_retries", 2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Using model: %s for conversation generation", model_name)

    # Resume support — count by scenario_type+domain
    existing_counts: dict[str, int] = {}
    examples: list[ConversationExample] = []
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    key = f"{row.get('scenario_type', '')}:{row.get('domain', '')}"
                    existing_counts[key] = existing_counts.get(key, 0) + 1
                    examples.append(ConversationExample.from_dict(row))
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Skipping malformed line %d in %s", line_num, output_path)
        logger.info("Resume: found %d existing conversations %s", len(examples), dict(existing_counts))

    failed = 0
    new_count = 0
    write_lock = threading.Lock()

    def _generate_one(scenario: dict) -> ConversationExample | None:
        """Generate a single conversation (runs in thread pool)."""
        return _generate_conversation(
            scenario, tc.endpoint, model_name,
            timeout=timeout, max_retries=max_retries,
        )

    with open(output_path, "a", encoding="utf-8") as f:
        for scenario_type, scenarios in _MULTITURN_SCENARIOS.items():
            for scenario in scenarios:
                domain = scenario.get("domain", "general")
                key = f"{scenario_type}:{domain}"
                already_done = existing_counts.get(key, 0)
                remaining = num_per_scenario_domain - already_done

                if remaining <= 0:
                    logger.info("Scenario '%s' (%s): already complete (%d/%d), skipping",
                                scenario_type, domain, already_done, num_per_scenario_domain)
                    continue

                logger.info("Generating %d conversations for '%s:%s' with %d workers (already have %d)",
                            remaining, scenario_type, domain, num_workers, already_done)

                with ThreadPoolExecutor(max_workers=num_workers) as pool:
                    futures = {
                        pool.submit(_generate_one, scenario): i
                        for i in range(remaining)
                    }
                    done_count = 0
                    for future in as_completed(futures):
                        done_count += 1
                        conv = future.result()
                        if conv is None:
                            failed += 1
                            continue

                        with write_lock:
                            examples.append(conv)
                            new_count += 1
                            f.write(json.dumps(conv.to_dict(), ensure_ascii=False) + "\n")
                            f.flush()

                        if done_count % 10 == 0:
                            logger.info("  [%s:%s] %d/%d done",
                                        scenario_type, domain, done_count, remaining)

    logger.info(
        "Conversation factory complete: %d total in %s (new: %d, failed: %d)",
        len(examples), output_path, new_count, failed,
    )
    return examples


# ---------------------------------------------------------------------------
# Conceptual/explanation example generation
# ---------------------------------------------------------------------------

def generate_conceptual_examples(
    config: PipelineConfig,
    output_path: str | Path = "data/raw/conceptual.jsonl",
    num_per_category: int = 350,
    num_workers: int = 8,
    model_override: str | None = None,
) -> list[TrainingExample]:
    """Generate conceptual/explanation training examples (not code).

    Covers categories like hl7v2_vs_fhir, mirth_architecture, ihe_profiles, etc.
    These examples teach the model domain knowledge and reasoning rather than
    code generation.

    Supports resume: if output_path already exists, counts existing examples
    per category and generates only the remaining.

    Args:
        config: Pipeline configuration.
        output_path: Where to write the output JSONL.
        num_per_category: Number of examples to generate per category.
        num_workers: Number of parallel generation workers.
        model_override: Optional model name override.

    Returns:
        List of TrainingExample objects (existing + new).
    """
    tc = config.data_factory
    model_name = model_override or _select_model(config, complex_task=True)
    timeout = getattr(tc, "timeout_seconds", 300)
    max_retries = getattr(tc, "max_retries", 2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simpo_path = Path("data/simpo/pairs.jsonl")

    logger.info("Using model: %s for conceptual data generation", model_name)

    # Resume support: count existing examples per category
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
                    cat = row.get("domain", "unknown")
                    existing_counts[cat] = existing_counts.get(cat, 0) + 1
                    examples.append(TrainingExample.from_dict(row))
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Skipping malformed line %d in %s", line_num, output_path)
        logger.info("Resume: found %d existing conceptual examples %s",
                     len(examples), dict(existing_counts))

    failed = 0
    new_count = 0
    write_lock = threading.Lock()
    simpo_lock = threading.Lock()

    def _generate_one(category: str) -> TrainingExample | None:
        """Generate a single conceptual example (runs in thread pool)."""
        templates = _CONCEPTUAL_VIBE_TEMPLATES.get(
            category, list(_CONCEPTUAL_VIBE_TEMPLATES.values())[0]
        )
        seed = random.choice(templates)

        # Generate a new instruction inspired by the seed
        instr_prompt = (
            f"You are generating training data for a healthcare integration AI.\n"
            f"Category: {category}\n\n"
            f"Inspired by this example question: \"{seed}\"\n\n"
            f"Generate a NEW, different conceptual question about {category.replace('_', ' ')}. "
            f"The question should require an explanatory answer (not code). "
            f"Be specific and include realistic healthcare IT details. "
            f"Return ONLY the question, nothing else."
        )
        instruction = _call_teacher(
            instr_prompt, tc.endpoint, model_name,
            timeout=timeout, max_retries=max_retries,
        )
        if not instruction:
            instruction = seed

        # Generate the conceptual response using rejection sampling
        resp_prompt = (
            f"You are a healthcare integration expert. Answer the following question "
            f"with a clear, detailed explanation. Do NOT generate code — focus on "
            f"concepts, architecture, standards, and best practices.\n\n"
            f"Question: {instruction}\n\n"
            f"Provide a thorough, well-structured explanation."
        )
        response = _generate_with_rejection(
            resp_prompt, category, config,
            tc.endpoint, model_name,
            simpo_path=simpo_path, simpo_lock=simpo_lock,
            temperature=0.5,
            timeout=timeout, max_retries=max_retries,
        )
        if not response:
            return None
        return TrainingExample(
            instruction=instruction, input="", output=response,
            domain=category, source_standard=category,
            version="conceptual-v1",
        )

    with open(output_path, "a", encoding="utf-8") as f:
        for category in _CONCEPTUAL_VIBE_TEMPLATES:
            already_done = existing_counts.get(category, 0)
            remaining = num_per_category - already_done

            if remaining <= 0:
                logger.info("Category '%s': already complete (%d/%d), skipping",
                            category, already_done, num_per_category)
                continue

            logger.info("Generating %d conceptual examples for '%s' with %d workers (already have %d)",
                        remaining, category, num_workers, already_done)

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(_generate_one, category): i
                    for i in range(remaining)
                }
                done_count = 0
                for future in as_completed(futures):
                    done_count += 1
                    example = future.result()
                    if example is None:
                        failed += 1
                        continue

                    with write_lock:
                        examples.append(example)
                        new_count += 1
                        f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                        f.flush()

                    if done_count % 25 == 0:
                        logger.info("  [%s] %d/%d done", category, done_count, remaining)

    logger.info(
        "Conceptual factory complete: %d total in %s (new: %d, failed: %d)",
        len(examples), output_path, new_count, failed,
    )
    return examples


# ---------------------------------------------------------------------------
# Raw HL7 message generation
# ---------------------------------------------------------------------------

def generate_raw_hl7_examples(
    config: PipelineConfig,
    output_path: str | Path = "data/raw/raw_hl7.jsonl",
    num_examples: int = 500,
    num_workers: int = 8,
    model_override: str | None = None,
) -> list[TrainingExample]:
    """Generate raw pipe-delimited HL7 message examples.

    Generates realistic HL7 v2 messages (ADT, ORU, ORM, SIU, VXU) with
    proper segment structure, pipe delimiters, and realistic field values.

    Supports resume: if output_path already exists, counts existing examples
    per message type and generates only the remaining.

    Args:
        config: Pipeline configuration.
        output_path: Where to write the output JSONL.
        num_examples: Total number of examples to generate (distributed across types).
        num_workers: Number of parallel generation workers.
        model_override: Optional model name override.

    Returns:
        List of TrainingExample objects (existing + new).
    """
    tc = config.data_factory
    model_name = model_override or _select_model(config, complex_task=False)
    timeout = getattr(tc, "timeout_seconds", 300)
    max_retries = getattr(tc, "max_retries", 2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_per_type = max(1, num_examples // len(_RAW_HL7_TYPES))

    logger.info("Using model: %s for raw HL7 generation", model_name)

    # Resume support: count existing examples per HL7 type
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
                    hl7_type = row.get("source_standard", "unknown")
                    existing_counts[hl7_type] = existing_counts.get(hl7_type, 0) + 1
                    examples.append(TrainingExample.from_dict(row))
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Skipping malformed line %d in %s", line_num, output_path)
        logger.info("Resume: found %d existing raw HL7 examples %s",
                     len(examples), dict(existing_counts))

    failed = 0
    new_count = 0
    write_lock = threading.Lock()

    def _generate_one(hl7_type: str) -> TrainingExample | None:
        """Generate a single raw HL7 message example (runs in thread pool)."""
        # Instruction: ask for a raw HL7 message
        instruction = (
            f"Generate a realistic, complete HL7 v2.5.1 {hl7_type} message "
            f"with proper pipe-delimited segments. Include realistic but "
            f"fictional patient data (no real PHI)."
        )

        prompt = (
            f"You are an HL7 v2 message expert. Generate a complete, valid, "
            f"pipe-delimited HL7 v2.5.1 {hl7_type} message.\n\n"
            f"Requirements:\n"
            f"- Use proper HL7 encoding characters: |^~\\&\n"
            f"- Include MSH segment with correct message type\n"
            f"- Include all standard segments for {hl7_type} messages\n"
            f"- Use realistic but fictional patient data (no real PHI)\n"
            f"- Use realistic facility names and OIDs\n"
            f"- Include proper timestamps in HL7 format (YYYYMMDDHHMMSS)\n\n"
            f"Return ONLY the raw HL7 message with segments separated by newlines. "
            f"No explanation, no markdown, just the message."
        )

        response = _call_teacher(
            prompt, tc.endpoint, model_name,
            temperature=0.6, timeout=timeout, max_retries=max_retries,
        )
        if not response:
            return None

        return TrainingExample(
            instruction=instruction, input="", output=response,
            domain="hl7v2", source_standard=hl7_type,
            version="raw-hl7-v1",
        )

    with open(output_path, "a", encoding="utf-8") as f:
        for hl7_type in _RAW_HL7_TYPES:
            already_done = existing_counts.get(hl7_type, 0)
            remaining = num_per_type - already_done

            if remaining <= 0:
                logger.info("HL7 type '%s': already complete (%d/%d), skipping",
                            hl7_type, already_done, num_per_type)
                continue

            logger.info("Generating %d raw HL7 %s messages with %d workers (already have %d)",
                        remaining, hl7_type, num_workers, already_done)

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(_generate_one, hl7_type): i
                    for i in range(remaining)
                }
                done_count = 0
                for future in as_completed(futures):
                    done_count += 1
                    example = future.result()
                    if example is None:
                        failed += 1
                        continue

                    with write_lock:
                        examples.append(example)
                        new_count += 1
                        f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                        f.flush()

                    if done_count % 25 == 0:
                        logger.info("  [%s] %d/%d done", hl7_type, done_count, remaining)

    logger.info(
        "Raw HL7 factory complete: %d total in %s (new: %d, failed: %d)",
        len(examples), output_path, new_count, failed,
    )
    return examples
