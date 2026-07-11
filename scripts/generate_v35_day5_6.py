"""V3.5 Day 5-6: Generate error handling and debug training data.

Error Handling: 5,000 new + 10,000 retrofit from existing examples.
Debug with Code: 3,000 structured debugging examples.
"""
import argparse
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ENDPOINT = "http://127.0.0.1:11434/api/generate"
MODEL_70B = "llama3:70b"
MODEL_8B = "llama3:8b"
TIMEOUT = 600
MAX_RETRIES = 3
NUM_WORKERS_70B = 6
NUM_WORKERS_8B = 8

_file_lock = threading.Lock()


def _call(prompt, temperature=0.3, model=None):
    """Call Ollama endpoint with retry logic."""
    model = model or MODEL_70B
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                ENDPOINT,
                json={
                    "model": model,
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
                logger.warning("Failed after %d attempts: %s", MAX_RETRIES, e)
                return None
            time.sleep(2 ** attempt)


def _write(path, example):
    """Thread-safe append of a JSON example to file."""
    with _file_lock:
        with open(path, "a") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def _count_lines(path):
    """Count existing lines in a file for resume support."""
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for _ in open(p))


# ---------------------------------------------------------------------------
# Error Handling Prompts (5,000 new examples)
# ---------------------------------------------------------------------------

LANGS = ["Python", "Java", "JavaScript", "C#"]

HEALTHCARE_CODE_ERROR_PROMPTS = [
    "Write a {lang} function that retrieves patient records from a database. Include comprehensive try/catch with meaningful error messages and structured logging for each failure mode (connection, timeout, not found, auth).",
    "Write a {lang} function that processes HL7 ADT^A01 admission messages and stores them. Wrap every I/O operation in try/catch with meaningful error messages and logging. Handle parse errors, DB errors, and validation failures.",
    "Write a {lang} function that reads a FHIR Patient resource and creates a local record. Include try/catch around the HTTP call, JSON parsing, and DB insert, with meaningful error messages and logging at each step.",
    "Write a {lang} class that synchronizes patient demographics between two systems. Every method must have try/catch with meaningful error messages and logging. Handle network, parsing, and data conflict errors.",
    "Write a {lang} function that processes a batch of lab results (OBX segments). Wrap each result's processing in try/catch so one failure doesn't stop the batch. Log meaningful error messages with context.",
    "Write a {lang} service that accepts HL7 ORM^O01 order messages via TCP. Include try/catch for socket errors, HL7 parse errors, and downstream API failures, with meaningful error messages and logging.",
    "Write a {lang} function that converts CDA documents to FHIR R4 bundles. Include try/catch for XML parse errors, mapping failures, and validation errors, with meaningful error messages and structured logging.",
    "Write a {lang} medication reconciliation function that compares two medication lists. Include try/catch with meaningful error messages for parsing, comparison logic, and output generation.",
    "Write a {lang} function that sends patient discharge summaries to a HIE. Include try/catch with meaningful error messages and logging for serialization, network, and acknowledgment failures.",
    "Write a {lang} insurance eligibility checker that calls a payer API. Include try/catch with meaningful error messages and logging for request building, HTTP errors, response parsing, and business rule failures.",
    "Write a {lang} function that generates a CCDA document from structured patient data. Include try/catch with meaningful error messages at each generation step: header, sections, entries, and XML serialization.",
    "Write a {lang} patient matching algorithm that compares demographics across systems. Include try/catch with meaningful error messages for data access, comparison, scoring, and result storage.",
    "Write a {lang} function to process HL7 SIU^S12 scheduling messages. Include try/catch with meaningful error messages and logging for parse errors, calendar conflicts, and notification failures.",
    "Write a {lang} clinical decision support function that evaluates medication interactions. Include try/catch with meaningful error messages for drug DB lookups, rule evaluation, and alert generation.",
    "Write a {lang} function that archives old patient records to cold storage. Include try/catch with meaningful error messages for read, compress, upload, and delete operations with full logging.",
]

API_INTEGRATION_ERROR_PROMPTS = [
    "Write a {lang} HTTP client class for a healthcare API that implements exponential backoff with jitter, a circuit breaker (open after 5 failures, half-open after 30s), and configurable timeout handling. Include logging.",
    "Write a {lang} FHIR API client with retry logic using exponential backoff (base 2s, max 60s), a circuit breaker pattern that stops requests after consecutive failures, and per-request timeout handling.",
    "Write a {lang} EHR integration client that calls multiple endpoints. Implement exponential backoff for transient errors (429, 503), a circuit breaker that trips after N failures, and request-level timeout handling.",
    "Write a {lang} API gateway for routing requests to multiple EHR vendors. Include exponential backoff per vendor, independent circuit breakers for each backend, and cascading timeout handling.",
    "Write a {lang} webhook delivery system for healthcare events. Implement exponential backoff for failed deliveries, a circuit breaker per destination, timeout handling, and dead-letter queue for persistent failures.",
    "Write a {lang} bulk FHIR export client ($export) that polls for completion. Include exponential backoff on poll intervals, circuit breaker for repeated server errors, and overall operation timeout handling.",
    "Write a {lang} OAuth2 token manager for healthcare APIs. Implement token refresh with exponential backoff on auth failures, circuit breaker for the token endpoint, and timeout handling for token requests.",
    "Write a {lang} real-time HL7 MLLP client with connection pooling. Include reconnect with exponential backoff, circuit breaker for unreachable hosts, and timeout handling for message send/receive.",
]

HL7_FHIR_PARSING_ERROR_PROMPTS = [
    "Write a {lang} HL7 v2 parser that defensively handles malformed messages: missing segment terminators, wrong field counts, invalid date formats, unexpected encoding characters. Log each anomaly and continue parsing.",
    "Write a {lang} FHIR R4 resource validator that defensively parses JSON with missing required fields, wrong data types, invalid references, and unknown extensions. Return structured error reports without crashing.",
    "Write a {lang} HL7 ADT message parser with defensive parsing for: truncated messages, duplicate segments, out-of-order segments, non-ASCII characters, and mismatched message types. Recover gracefully from each.",
    "Write a {lang} function that parses HL7 ORU^R01 lab results defensively. Handle malformed OBX segments, missing units, unparseable numeric values, and corrupt encoding. Extract what's valid, log what's not.",
    "Write a {lang} FHIR Bundle parser that handles malformed entries: invalid resource types, circular references, missing IDs, oversized payloads, and duplicate entries. Parse what's valid, skip and log what's broken.",
    "Write a {lang} HL7 ORM^O01 order parser with defensive handling for: missing ORC segments, malformed OBR fields, invalid order control codes, and inconsistent segment grouping.",
    "Write a {lang} FHIR Observation parser that defensively handles: missing valueQuantity, invalid LOINC codes, nested extensions with missing URLs, and dateTime fields in unexpected formats.",
    "Write a {lang} HL7 MDM^T02 document parser with defensive parsing for: base64-encoded content that's corrupt, missing TXA segments, OBX-5 with mixed types, and oversized embedded documents.",
    "Write a {lang} function that converts HL7 v2 messages to FHIR. Defensively handle mapping failures: unmappable code systems, missing required source fields, and structural mismatches. Log each fallback used.",
    "Write a {lang} FHIR Medication resource parser that defensively handles: missing coding systems, ambiguous RxNorm codes, conflicting dosage instructions, and incomplete ingredient lists.",
]

# ---------------------------------------------------------------------------
# Debug with Code Prompts (3,000 examples)
# ---------------------------------------------------------------------------

DEBUG_INSTRUCTION = (
    "Respond with exactly 4 sections: "
    "1) Problem Analysis (explain likely root causes), "
    "2) Diagnostic Commands (shell/API commands to gather info), "
    "3) Fix Code (working code that resolves the issue), "
    "4) Verification Steps (how to confirm the fix works)."
)

MIRTH_DEBUG_PROMPTS = [
    "My Mirth Connect channel is dropping HL7 messages silently. The source is a TCP Listener on port 6661 and the destination is a Database Writer. No errors in the dashboard. " + DEBUG_INSTRUCTION,
    "Mirth Connect channel shows 'QUEUED' status for all messages but none are being sent to the destination HTTP endpoint. The channel was working yesterday. " + DEBUG_INSTRUCTION,
    "My Mirth transformer JavaScript is throwing 'Cannot read property of undefined' on msg['PID']['PID.5']. The channel processes ADT^A01 messages. " + DEBUG_INSTRUCTION,
    "Mirth Connect channel has a memory leak. Java heap keeps growing until OutOfMemoryError crashes the service. The channel processes HL7 to FHIR conversions. " + DEBUG_INSTRUCTION,
    "My Mirth channel's response transformer isn't firing. I set up an auto-ACK but the sending system reports no acknowledgment. TCP Listener source. " + DEBUG_INSTRUCTION,
    "Mirth channel destination is a FHIR REST endpoint but getting 401 Unauthorized. OAuth token refresh seems to fail intermittently. " + DEBUG_INSTRUCTION,
    "Mirth Connect database reader source is re-processing the same records repeatedly. I'm using a polling query with a processed flag. " + DEBUG_INSTRUCTION,
    "My Mirth channel's JavaScript filter is causing the channel to slow down dramatically. It queries a database for each message. " + DEBUG_INSTRUCTION,
    "Mirth channel fails to start after server restart. Error: 'Channel XX could not be deployed: null'. No other details in the log. " + DEBUG_INSTRUCTION,
    "Mirth Connect HTTP Listener channel receives POST requests but the body is always empty in the transformer. Content-Type is application/json. " + DEBUG_INSTRUCTION,
    "My Mirth channel's source queue is growing unboundedly. Destination is slower than source. Need to implement backpressure without losing messages. " + DEBUG_INSTRUCTION,
    "Mirth channel converts HL7 to XML but special characters (&, <, >) in patient names break the output XML. " + DEBUG_INSTRUCTION,
    "My Mirth Connect channel duplicates messages when the destination is temporarily unavailable and then comes back. " + DEBUG_INSTRUCTION,
    "Mirth channel's JavaScript transformer runs out of Rhino script engine instances under heavy load. " + DEBUG_INSTRUCTION,
    "Mirth Connect channel with File Writer destination creates files with wrong encoding. Expected UTF-8 but getting ISO-8859-1. " + DEBUG_INSTRUCTION,
    "My Mirth channel needs to route messages to different destinations based on MSH-9 message type but the router logic isn't working. " + DEBUG_INSTRUCTION,
]

HL7_DEBUG_PROMPTS = [
    "HL7 parser throws 'Segment PID not found' on messages that clearly contain PID. The message comes from an Epic system and uses \\r\\n line endings. " + DEBUG_INSTRUCTION,
    "HL7 ACK message has error code AE with 'Segment sequence error'. My ADT^A08 message has PV1 before PID. How to diagnose and fix. " + DEBUG_INSTRUCTION,
    "HL7 message receiver reports 'Invalid encoding characters' for messages from a Cerner system. MSH-2 contains ^~\\& but receiver expects ^~\\\\&. " + DEBUG_INSTRUCTION,
    "HL7 OBX segments are losing decimal precision. Value '0.50' becomes '0.5' after parsing and re-serializing. Lab system requires exact format. " + DEBUG_INSTRUCTION,
    "HL7 Z-segment (ZPD) custom fields are being stripped by my parser. Need to preserve custom segments during message transformation. " + DEBUG_INSTRUCTION,
    "HL7 message contains repeating IN1 segments but my parser only reads the first insurance. Patient has dual coverage. " + DEBUG_INSTRUCTION,
    "HL7 date fields have inconsistent formats: some YYYYMMDD, some YYYYMMDDHHMMSS, some with timezone offsets. Parser crashes on mixed formats. " + DEBUG_INSTRUCTION,
    "HL7 MLLP connection drops after every 100 messages exactly. Sending system gets connection reset. Receiver is a Java TCP server. " + DEBUG_INSTRUCTION,
    "HL7 message has embedded escape sequences (\\E\\, \\T\\, \\R\\) that aren't being decoded properly in field values. " + DEBUG_INSTRUCTION,
    "HL7 batch file processing fails on message 500 of 1000. BatchTrailerMessage (BTS) count doesn't match actual message count. " + DEBUG_INSTRUCTION,
    "HL7 NK1 (next of kin) segment has phone numbers in inconsistent formats across different sending systems. Parser validation fails. " + DEBUG_INSTRUCTION,
    "HL7 message routing based on MSH-5 (receiving application) sends messages to wrong destination when the value contains sub-components. " + DEBUG_INSTRUCTION,
]

FHIR_DEBUG_PROMPTS = [
    "FHIR $everything operation on Patient returns 413 Payload Too Large for patients with extensive history. Server is HAPI FHIR. " + DEBUG_INSTRUCTION,
    "FHIR search returns empty bundle when searching Patient?identifier=MRN|12345 but the patient exists. Works with Patient?_id=xxx. " + DEBUG_INSTRUCTION,
    "FHIR SMART on FHIR launch fails with 'invalid_scope' when requesting patient/*.read. The EHR is Epic and was working last week. " + DEBUG_INSTRUCTION,
    "FHIR Subscription notification webhook receives duplicate notifications for the same resource change. Server is HAPI FHIR R4. " + DEBUG_INSTRUCTION,
    "FHIR Batch bundle processing returns OperationOutcome with 'Reference not found' for resources within the same bundle. Order matters? " + DEBUG_INSTRUCTION,
    "FHIR CDS Hooks service returns suggestions but the EHR ignores them. No errors in the EHR log. Hook type is patient-view. " + DEBUG_INSTRUCTION,
    "FHIR Bulk Data Export ($export) job stays in 'in-progress' state indefinitely. No errors in server log. HAPI FHIR server. " + DEBUG_INSTRUCTION,
    "FHIR Consent resource isn't being enforced. Patient opted out but their data still appears in search results. " + DEBUG_INSTRUCTION,
    "FHIR _include parameter returns 500 Internal Server Error when including Practitioner references from Encounter resources. " + DEBUG_INSTRUCTION,
    "FHIR PATCH operation on Patient resource fails with 'Conflict' when using JSON Patch to update address. Works with PUT. " + DEBUG_INSTRUCTION,
    "FHIR DocumentReference search returns results but the Binary content URLs return 403 Forbidden even with valid token. " + DEBUG_INSTRUCTION,
    "FHIR ValueSet $expand operation times out for large code systems like SNOMED CT. Need it for terminology validation. " + DEBUG_INSTRUCTION,
]

EHR_DEBUG_PROMPTS = [
    "Epic FHIR API returns 'Token expired' even though the token was refreshed 5 minutes ago. Using backend services auth with JWT. " + DEBUG_INSTRUCTION,
    "Epic MyChart patient-facing app gets 'insufficient_scope' on appointment read. Scopes were approved during registration. " + DEBUG_INSTRUCTION,
    "Cerner Ignite sandbox returns different data structure than production for Patient resource. Code works in sandbox but fails in prod. " + DEBUG_INSTRUCTION,
    "Athena Health API returns paginated results but the next page URL returns 400 Bad Request after the third page. " + DEBUG_INSTRUCTION,
    "MEDITECH Expanse API connection drops intermittently. TLS handshake fails with 'certificate unknown' error every few hours. " + DEBUG_INSTRUCTION,
    "Epic Interconnect web service returns SOAP fault 'Access denied' for a specific patient. Other patients work fine. Break-the-glass? " + DEBUG_INSTRUCTION,
    "Cerner Millennium CCDA document query returns XML with invalid namespace declarations. Downstream parser fails. " + DEBUG_INSTRUCTION,
    "Allscripts Unity API returns HTTP 200 but the response body is HTML error page instead of JSON. Happens during peak hours. " + DEBUG_INSTRUCTION,
    "Epic CDS Hooks integration fires but the returned cards don't show in the EHR. No errors in Epic's CDS log. " + DEBUG_INSTRUCTION,
    "Oracle Health (Cerner) R4 FHIR endpoint returns _revinclude results without Provenance even though it's requested. " + DEBUG_INSTRUCTION,
]

GENERAL_INTEGRATION_DEBUG_PROMPTS = [
    "Healthcare integration engine processes 50 messages/sec but suddenly drops to 5/sec. CPU usage is low, memory is fine. Database query times increased. " + DEBUG_INSTRUCTION,
    "TLS mutual authentication (mTLS) between two healthcare systems fails with 'bad certificate'. Both certs are valid and not expired. " + DEBUG_INSTRUCTION,
    "Message queue (RabbitMQ) for healthcare messages has growing dead-letter queue. Consumer keeps rejecting messages with no error logged. " + DEBUG_INSTRUCTION,
    "Healthcare data warehouse ETL job fails midway through daily load. Duplicate key violation on patient dimension table. " + DEBUG_INSTRUCTION,
    "API rate limiting is triggered even though we're well under the documented limit. Healthcare vendor says our usage looks normal. " + DEBUG_INSTRUCTION,
    "SFTP file transfer of healthcare data fails intermittently with 'connection reset'. File sizes vary from 1KB to 500MB. " + DEBUG_INSTRUCTION,
    "OAuth2 client credentials flow works for 23 hours then fails until manual token cache clear. Token lifetime is 3600s. " + DEBUG_INSTRUCTION,
    "Healthcare event bus (Kafka) consumer group rebalances frequently causing duplicate message processing. " + DEBUG_INSTRUCTION,
    "Docker container running HAPI FHIR server crashes with OOM despite 8GB heap allocation. JVM memory settings seem correct. " + DEBUG_INSTRUCTION,
    "Healthcare data sync between two sites shows patient records updated in site A but not reflected in site B for hours. " + DEBUG_INSTRUCTION,
]


# ---------------------------------------------------------------------------
# Generation Functions
# ---------------------------------------------------------------------------

def generate_error_handling(output_path, target_count=5000):
    """Generate 5,000 new error handling examples."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_lines(output_path)
    if existing >= target_count:
        logger.info("Error handling: already have %d/%d, skipping", existing, target_count)
        return existing

    remaining = target_count - existing
    logger.info("Error handling: generating %d examples (%d existing)", remaining, existing)

    all_prompts = []

    # 3,000 healthcare code with try/catch
    for _ in range(3000):
        lang = random.choice(LANGS)
        template = random.choice(HEALTHCARE_CODE_ERROR_PROMPTS)
        all_prompts.append(("healthcare_error_handling", template.format(lang=lang), "healthcare"))

    # 1,000 API integration with backoff/circuit breaker/timeout
    for _ in range(1000):
        lang = random.choice(LANGS)
        template = random.choice(API_INTEGRATION_ERROR_PROMPTS)
        all_prompts.append(("api_resilience", template.format(lang=lang), "api_integration"))

    # 1,000 HL7/FHIR defensive parsing
    for _ in range(1000):
        lang = random.choice(LANGS)
        template = random.choice(HL7_FHIR_PARSING_ERROR_PROMPTS)
        all_prompts.append(("defensive_parsing", template.format(lang=lang), "hl7_fhir"))

    random.shuffle(all_prompts)
    all_prompts = all_prompts[:remaining]

    completed = [0]
    failed = [0]

    def process(item):
        category, prompt, domain = item
        output = _call(prompt, temperature=0.3, model=MODEL_70B)
        if output and len(output) > 100:
            example = {
                "instruction": prompt,
                "output": output,
                "domain": domain,
                "source_standard": category,
                "version": "v3.5-error-handling",
            }
            _write(output_path, example)
            completed[0] += 1
            if completed[0] % 50 == 0:
                logger.info("Error handling: %d/%d done (failed: %d)", completed[0], remaining, failed[0])
        else:
            failed[0] += 1

    with ThreadPoolExecutor(max_workers=NUM_WORKERS_70B) as pool:
        list(pool.map(process, all_prompts))

    logger.info("Error handling complete: %d succeeded, %d failed", completed[0], failed[0])
    return existing + completed[0]


def generate_error_retrofit(input_path, output_path, target_count=10000):
    """Retrofit existing examples with error handling using 8B model."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 0

    existing = _count_lines(output_path)
    if existing >= target_count:
        logger.info("Error retrofit: already have %d/%d, skipping", existing, target_count)
        return existing

    # Load source examples
    source_examples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    source_examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not source_examples:
        logger.error("No valid examples in %s", input_path)
        return 0

    remaining = target_count - existing
    logger.info("Error retrofit: generating %d examples (%d existing, %d source)", remaining, existing, len(source_examples))

    # If we have fewer source examples than target, cycle through them
    if len(source_examples) < remaining:
        import itertools
        examples_to_process = list(itertools.islice(itertools.cycle(source_examples), remaining))
    else:
        examples_to_process = random.sample(source_examples, remaining)

    # Skip already-processed examples for resume support
    examples_to_process = examples_to_process[existing:]
    if not examples_to_process:
        # Reset: we already counted existing above, just process remaining
        examples_to_process = examples_to_process[:remaining]

    completed = [0]
    failed = [0]

    def process(src_example):
        existing_output = src_example.get("output", "")
        if not existing_output or len(existing_output) < 50:
            failed[0] += 1
            return

        retrofit_prompt = (
            "Add proper error handling (try/catch) with meaningful error messages "
            "and logging to this code: " + existing_output
        )

        output = _call(retrofit_prompt, temperature=0.3, model=MODEL_8B)
        if output and len(output) > 100:
            example = {
                "instruction": src_example.get("instruction", ""),
                "output": output,
                "domain": src_example.get("domain", "healthcare"),
                "source_standard": "error_retrofit",
                "version": "v3.5-error-retrofit",
            }
            _write(output_path, example)
            completed[0] += 1
            if completed[0] % 50 == 0:
                logger.info("Error retrofit: %d/%d done (failed: %d)", completed[0], remaining, failed[0])
        else:
            failed[0] += 1

    with ThreadPoolExecutor(max_workers=NUM_WORKERS_8B) as pool:
        list(pool.map(process, examples_to_process))

    logger.info("Error retrofit complete: %d succeeded, %d failed", completed[0], failed[0])
    return existing + completed[0]


def generate_debug(output_path, target_count=3000):
    """Generate 3,000 structured debug examples."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_lines(output_path)
    if existing >= target_count:
        logger.info("Debug: already have %d/%d, skipping", existing, target_count)
        return existing

    remaining = target_count - existing
    logger.info("Debug: generating %d examples (%d existing)", remaining, existing)

    all_prompts = []

    # 800 Mirth channel debugging
    for prompt in MIRTH_DEBUG_PROMPTS:
        count_per = 800 // len(MIRTH_DEBUG_PROMPTS)
        for _ in range(count_per):
            all_prompts.append(("mirth_debug", prompt, "mirth"))
    # top up to 800
    while sum(1 for p in all_prompts if p[0] == "mirth_debug") < 800:
        all_prompts.append(("mirth_debug", random.choice(MIRTH_DEBUG_PROMPTS), "mirth"))

    # 600 HL7 message debugging
    for prompt in HL7_DEBUG_PROMPTS:
        count_per = 600 // len(HL7_DEBUG_PROMPTS)
        for _ in range(count_per):
            all_prompts.append(("hl7_debug", prompt, "hl7v2"))
    while sum(1 for p in all_prompts if p[0] == "hl7_debug") < 600:
        all_prompts.append(("hl7_debug", random.choice(HL7_DEBUG_PROMPTS), "hl7v2"))

    # 600 FHIR API debugging
    for prompt in FHIR_DEBUG_PROMPTS:
        count_per = 600 // len(FHIR_DEBUG_PROMPTS)
        for _ in range(count_per):
            all_prompts.append(("fhir_debug", prompt, "fhir"))
    while sum(1 for p in all_prompts if p[0] == "fhir_debug") < 600:
        all_prompts.append(("fhir_debug", random.choice(FHIR_DEBUG_PROMPTS), "fhir"))

    # 500 EHR integration debugging
    for prompt in EHR_DEBUG_PROMPTS:
        count_per = 500 // len(EHR_DEBUG_PROMPTS)
        for _ in range(count_per):
            all_prompts.append(("ehr_debug", prompt, "ehr"))
    while sum(1 for p in all_prompts if p[0] == "ehr_debug") < 500:
        all_prompts.append(("ehr_debug", random.choice(EHR_DEBUG_PROMPTS), "ehr"))

    # 500 general integration debugging
    for prompt in GENERAL_INTEGRATION_DEBUG_PROMPTS:
        count_per = 500 // len(GENERAL_INTEGRATION_DEBUG_PROMPTS)
        for _ in range(count_per):
            all_prompts.append(("general_debug", prompt, "integration"))
    while sum(1 for p in all_prompts if p[0] == "general_debug") < 500:
        all_prompts.append(("general_debug", random.choice(GENERAL_INTEGRATION_DEBUG_PROMPTS), "integration"))

    random.shuffle(all_prompts)
    all_prompts = all_prompts[:remaining]

    completed = [0]
    failed = [0]

    def process(item):
        category, prompt, domain = item
        output = _call(prompt, temperature=0.3, model=MODEL_70B)
        if output and len(output) > 150:
            example = {
                "instruction": prompt,
                "output": output,
                "domain": domain,
                "source_standard": category,
                "version": "v3.5-debug",
            }
            _write(output_path, example)
            completed[0] += 1
            if completed[0] % 50 == 0:
                logger.info("Debug: %d/%d done (failed: %d)", completed[0], remaining, failed[0])
        else:
            failed[0] += 1

    with ThreadPoolExecutor(max_workers=NUM_WORKERS_70B) as pool:
        list(pool.map(process, all_prompts))

    logger.info("Debug complete: %d succeeded, %d failed", completed[0], failed[0])
    return existing + completed[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3.5 Day 5-6: Error handling and debug data generation")
    parser.add_argument("--error-output", default="data/raw/v35_error_handling.jsonl",
                        help="Output path for new error handling examples")
    parser.add_argument("--retrofit-input", default="data/raw/synthetic_run1.jsonl",
                        help="Input path for retrofit source examples")
    parser.add_argument("--retrofit-output", default="data/raw/v35_error_retrofit.jsonl",
                        help="Output path for retrofit examples")
    parser.add_argument("--debug-output", default="data/raw/v35_debug.jsonl",
                        help="Output path for debug examples")
    parser.add_argument("--error-count", type=int, default=5000,
                        help="Target count for new error handling examples")
    parser.add_argument("--retrofit-count", type=int, default=10000,
                        help="Target count for retrofit examples")
    parser.add_argument("--debug-count", type=int, default=3000,
                        help="Target count for debug examples")
    parser.add_argument("--skip-error", action="store_true", help="Skip new error handling generation")
    parser.add_argument("--skip-retrofit", action="store_true", help="Skip retrofit generation")
    parser.add_argument("--skip-debug", action="store_true", help="Skip debug generation")
    args = parser.parse_args()

    logger.info("=== V3.5 Day 5-6: Error Handling + Debug Generation ===")

    if not args.skip_error:
        logger.info("--- Phase 1: New Error Handling Examples (5,000) ---")
        generate_error_handling(args.error_output, args.error_count)

    if not args.skip_retrofit:
        logger.info("--- Phase 2: Error Handling Retrofit (10,000) ---")
        generate_error_retrofit(args.retrofit_input, args.retrofit_output, args.retrofit_count)

    if not args.skip_debug:
        logger.info("--- Phase 3: Debug with Code (3,000) ---")
        generate_debug(args.debug_output, args.debug_count)

    logger.info("=== Day 5-6 generation complete ===")
