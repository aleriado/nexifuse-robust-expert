"""V3.5 Day 3-4: Bidirectional Standard Translation + Complete Mirth Channel Generation.

Generates 5,000 translation examples and 3,000 full Mirth channel examples
using a single Ollama endpoint with ThreadPoolExecutor(6 workers).
"""
import argparse
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ENDPOINT = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3:70b"
TIMEOUT = 600
MAX_RETRIES = 3
NUM_WORKERS = 6
TEMPERATURE = 0.3

_file_lock = threading.Lock()


def _call(prompt):
    """Call Ollama endpoint with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                ENDPOINT,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "24h",
                    "options": {"temperature": TEMPERATURE},
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
    """Thread-safe append of a JSON line."""
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
# Bidirectional Standard Translation (5,000 examples)
# ---------------------------------------------------------------------------

# HL7 v2 message types used in prompts
HL7_MSG_TYPES = [
    "ADT^A01 (Admit)", "ADT^A02 (Transfer)", "ADT^A03 (Discharge)",
    "ADT^A04 (Register)", "ADT^A08 (Update Patient Info)",
    "ADT^A11 (Cancel Admit)", "ADT^A13 (Cancel Discharge)",
    "ORM^O01 (Order)", "ORU^R01 (Observation Result)",
    "SIU^S12 (Schedule)", "MDM^T02 (Document Notification)",
    "DFT^P03 (Charge/Billing)", "BAR^P01 (Add Patient Account)",
    "RDE^O11 (Pharmacy Order)", "VXU^V04 (Vaccination Update)",
    "PPR^PC1 (Problem Add)", "MFN^M02 (Master File - Staff)",
]

# FHIR R4 resource types
FHIR_RESOURCES = [
    "Patient", "Encounter", "Observation", "DiagnosticReport",
    "Condition", "Procedure", "MedicationRequest", "AllergyIntolerance",
    "Immunization", "CarePlan", "CareTeam", "ServiceRequest",
    "DocumentReference", "Practitioner", "Organization", "Location",
    "Coverage", "Claim", "ExplanationOfBenefit", "Schedule", "Appointment",
]

# Clinical data scenarios for CSV imports
CSV_SCENARIOS = [
    "patient demographics (MRN, name, DOB, gender, SSN, address, phone, email, insurance)",
    "lab results (accession number, test code, test name, result value, units, reference range, status, collected date)",
    "medication list (patient ID, drug name, NDC code, dose, route, frequency, start date, end date, prescriber)",
    "vital signs (patient ID, encounter ID, date, BP systolic, BP diastolic, heart rate, temperature, SpO2, weight, height)",
    "appointment schedule (patient ID, appointment ID, provider, location, date, time, duration, type, status)",
    "insurance eligibility (member ID, plan name, group number, coverage start, coverage end, copay, deductible, PCP)",
    "immunization records (patient ID, vaccine name, CVX code, lot number, date administered, site, route, administering provider)",
    "problem list (patient ID, ICD-10 code, description, onset date, status, severity, diagnosing provider)",
    "allergy records (patient ID, allergen, reaction type, severity, onset date, verification status)",
    "surgical history (patient ID, procedure code, CPT code, description, date, surgeon, facility, laterality)",
]

# CDA document types
CDA_DOC_TYPES = [
    "CCD (Continuity of Care Document)",
    "Discharge Summary",
    "Progress Note",
    "History and Physical",
    "Consultation Note",
    "Operative Note",
    "Referral Note",
    "Transfer Summary",
]

# Edge case categories to inject variation
EDGE_CASES = [
    "Handle missing/null fields gracefully with appropriate defaults or omit optional elements.",
    "Handle special characters in patient names (hyphens, apostrophes, accented characters, suffixes like Jr./III).",
    "Handle multiple phone numbers, addresses, and identifiers per patient.",
    "Handle date format differences (YYYYMMDD, YYYY-MM-DD, MM/DD/YYYY, epoch timestamps).",
    "Handle repeated segments/resources (multiple diagnoses, multiple allergies, multiple insurance plans).",
    "Handle race/ethnicity coding differences between standards (CDC race codes vs HL7 table 0005 vs FHIR ValueSet).",
    "Handle pediatric patients with guardian/guarantor information and age-specific reference ranges.",
    "Handle deceased patients with death date/time and applicable status codes.",
    "Handle encounter status transitions (planned, arrived, in-progress, finished, cancelled).",
    "Handle code system mapping (ICD-10 to SNOMED CT, CPT to HCPCS, LOINC to local codes).",
]


def _build_translation_prompts(count):
    """Build a list of (category, source_standard, prompt) tuples for translation tasks."""
    prompts = []

    # --- HL7 v2 to FHIR R4 (1,250) ---
    for _ in range(1250):
        msg = random.choice(HL7_MSG_TYPES)
        resource = random.choice(FHIR_RESOURCES)
        edge = random.choice(EDGE_CASES)
        prompts.append((
            "hl7v2_to_fhir",
            "HL7v2",
            f"Write a complete Python function that converts an HL7 v2 {msg} message "
            f"to a FHIR R4 {resource} resource (JSON). "
            f"Include:\n"
            f"1. Full field-by-field mapping from HL7 segments (PID, PV1, OBX, ORC, etc.) to FHIR elements\n"
            f"2. Code system translation (HL7 Table values to FHIR ValueSet URIs)\n"
            f"3. Identifier system mapping (MRN, SSN, visit number)\n"
            f"4. Proper FHIR resource structure with meta, text narrative, and extensions where needed\n"
            f"5. Comprehensive error handling for malformed segments and missing required fields\n"
            f"6. {edge}\n"
            f"Return the complete working function with type hints, docstring, and example usage."
        ))

    # --- FHIR R4 to HL7 v2 (1,250) ---
    for _ in range(1250):
        resource = random.choice(FHIR_RESOURCES)
        msg = random.choice(HL7_MSG_TYPES)
        edge = random.choice(EDGE_CASES)
        prompts.append((
            "fhir_to_hl7v2",
            "FHIR_R4",
            f"Write a complete Python function that converts a FHIR R4 {resource} resource (JSON) "
            f"to an HL7 v2 {msg} message string. "
            f"Include:\n"
            f"1. Reverse mapping from FHIR elements to HL7 segments and fields (PID, PV1, OBX, ORC, etc.)\n"
            f"2. FHIR ValueSet URIs back to HL7 Table values\n"
            f"3. Proper HL7 encoding characters, segment terminators, and field separators\n"
            f"4. Component and sub-component delimiters for complex fields (XPN, XAD, CX, CWE)\n"
            f"5. Comprehensive error handling for missing FHIR elements and invalid data\n"
            f"6. {edge}\n"
            f"Return the complete working function with type hints, docstring, and example usage."
        ))

    # --- FHIR R4 to CDA (500) ---
    for _ in range(500):
        resource = random.choice(FHIR_RESOURCES)
        cda_doc = random.choice(CDA_DOC_TYPES)
        edge = random.choice(EDGE_CASES)
        prompts.append((
            "fhir_to_cda",
            "FHIR_R4",
            f"Write a complete Python function that converts a FHIR R4 {resource} resource "
            f"into a C-CDA {cda_doc} XML section. "
            f"Include:\n"
            f"1. Proper CDA XML structure with templateId, code (LOINC section codes), title, and text\n"
            f"2. Structured entries with correct Act, Observation, or Procedure clinical statements\n"
            f"3. Code translation from FHIR CodeableConcept to CDA CD data type with codeSystem OIDs\n"
            f"4. Narrative text block generation (human-readable HTML) from structured data\n"
            f"5. Comprehensive error handling and validation of required CDA elements\n"
            f"6. {edge}\n"
            f"Return the complete working function using lxml or xml.etree with docstring and example."
        ))

    # --- CDA to FHIR R4 (500) ---
    for _ in range(500):
        cda_doc = random.choice(CDA_DOC_TYPES)
        resource = random.choice(FHIR_RESOURCES)
        edge = random.choice(EDGE_CASES)
        prompts.append((
            "cda_to_fhir",
            "CDA",
            f"Write a complete Python function that parses a C-CDA {cda_doc} XML document "
            f"and extracts data into a FHIR R4 {resource} resource (JSON). "
            f"Include:\n"
            f"1. XPath-based extraction of CDA clinical statements (Acts, Observations, Procedures)\n"
            f"2. OID-to-URI translation for code systems (SNOMED CT, LOINC, ICD-10, RxNorm)\n"
            f"3. CDA effectiveTime parsing (TS, IVL_TS) to FHIR dateTime/Period\n"
            f"4. CDA participant/performer mapping to FHIR references (Practitioner, Organization)\n"
            f"5. Comprehensive error handling for missing elements, namespace issues, and malformed XML\n"
            f"6. {edge}\n"
            f"Return the complete working function using lxml with docstring and example."
        ))

    # --- CSV to HL7 v2 (375) ---
    for _ in range(375):
        scenario = random.choice(CSV_SCENARIOS)
        msg = random.choice(HL7_MSG_TYPES)
        edge = random.choice(EDGE_CASES)
        prompts.append((
            "csv_to_hl7v2",
            "CSV",
            f"Write a complete Python function that reads a CSV file containing {scenario} "
            f"and converts each row into an HL7 v2 {msg} message. "
            f"Include:\n"
            f"1. CSV parsing with configurable column mapping (header names to HL7 field positions)\n"
            f"2. Proper MSH segment generation with sending/receiving application and facility\n"
            f"3. Data type formatting (dates to YYYYMMDD, phone to XTN, address to XAD, name to XPN)\n"
            f"4. Batch file output with FHS/BHS/BTS/FTS wrapping for multiple messages\n"
            f"5. Data validation and error reporting for each row with line numbers\n"
            f"6. {edge}\n"
            f"Return the complete working function with CLI support, docstring, and example."
        ))

    # --- CSV to FHIR R4 (375) ---
    for _ in range(375):
        scenario = random.choice(CSV_SCENARIOS)
        resource = random.choice(FHIR_RESOURCES)
        edge = random.choice(EDGE_CASES)
        prompts.append((
            "csv_to_fhir",
            "CSV",
            f"Write a complete Python function that reads a CSV file containing {scenario} "
            f"and converts each row into a FHIR R4 {resource} resource, "
            f"then bundles them into a FHIR Bundle (type: transaction). "
            f"Include:\n"
            f"1. CSV parsing with configurable column-to-FHIR-path mapping\n"
            f"2. Proper FHIR resource generation with meta.profile, identifier systems, and coding\n"
            f"3. Bundle entry creation with fullUrl (UUID-based), resource, and request (method/url)\n"
            f"4. Data type conversion (CSV strings to FHIR dateTime, Coding, Quantity, HumanName, Address)\n"
            f"5. Validation against FHIR profiles and error collection per row\n"
            f"6. {edge}\n"
            f"Return the complete working function with docstring and example usage."
        ))

    # --- HL7 v2 version upgrades (375) ---
    v2_versions = [
        ("2.1", "2.3"), ("2.3", "2.3.1"), ("2.3", "2.5"), ("2.3.1", "2.5"),
        ("2.3.1", "2.5.1"), ("2.5", "2.5.1"), ("2.5", "2.7"), ("2.5.1", "2.7"),
        ("2.5.1", "2.8"), ("2.7", "2.8"),
    ]
    for _ in range(375):
        src_ver, dst_ver = random.choice(v2_versions)
        msg = random.choice(HL7_MSG_TYPES)
        edge = random.choice(EDGE_CASES)
        prompts.append((
            "hl7v2_version_upgrade",
            f"HL7v2_{src_ver}",
            f"Write a complete Python function that upgrades an HL7 v2.{src_ver} {msg} message "
            f"to HL7 v2.{dst_ver} format. "
            f"Include:\n"
            f"1. MSH version field update and segment structure changes between versions\n"
            f"2. New required fields/segments added in v{dst_ver} with appropriate defaults\n"
            f"3. Deprecated field handling (map old fields to new locations or remove)\n"
            f"4. Data type changes between versions (e.g., ST to CWE, CM to CQ)\n"
            f"5. Backward-compatibility mode flag to keep optional old fields\n"
            f"6. {edge}\n"
            f"Return the complete working function with version mapping table, docstring, and example."
        ))

    # --- FHIR STU3 to R4 (375) ---
    stu3_breaking = [
        ("MedicationStatement", "MedicationStatement (taken field removed, statusReason added)"),
        ("MedicationRequest", "MedicationRequest (requester changed from BackboneElement to Reference)"),
        ("Immunization", "Immunization (notGiven removed, status=not-done added)"),
        ("DocumentReference", "DocumentReference (class renamed to category, indexed renamed to date)"),
        ("CapabilityStatement", "CapabilityStatement (multiple structural changes)"),
        ("Patient", "Patient (animal component removed, link type changes)"),
        ("Encounter", "Encounter (class changed from Coding to single code)"),
        ("Observation", "Observation (component.referenceRange structure change)"),
        ("Condition", "Condition (abatement renamed, stage changes)"),
        ("Procedure", "Procedure (notDone removed, status expanded)"),
    ]
    for _ in range(375):
        resource_stu3, description = random.choice(stu3_breaking)
        edge = random.choice(EDGE_CASES)
        prompts.append((
            "fhir_stu3_to_r4",
            "FHIR_STU3",
            f"Write a complete Python function that migrates a FHIR STU3 {description} resource "
            f"to FHIR R4 format. "
            f"Include:\n"
            f"1. All breaking changes between STU3 and R4 for this resource type\n"
            f"2. Renamed, removed, and restructured elements with proper mapping\n"
            f"3. ValueSet/CodeSystem URI updates from STU3 to R4\n"
            f"4. Extension handling for STU3 fields that became first-class R4 elements\n"
            f"5. Validation of the output against R4 StructureDefinition\n"
            f"6. {edge}\n"
            f"Return the complete working function with migration notes, docstring, and example."
        ))

    random.shuffle(prompts)
    return prompts[:count]


# ---------------------------------------------------------------------------
# Complete Mirth Channel Generation (3,000 examples)
# ---------------------------------------------------------------------------

# Source connector types
SOURCE_TYPES = [
    ("TCP Listener", "MLLP", "HL7 v2 messages over TCP/MLLP"),
    ("HTTP Listener", "REST", "FHIR JSON/XML over HTTP"),
    ("File Reader", "File", "HL7/CSV/XML files from a directory"),
    ("Database Reader", "JDBC", "polling a database table for new records"),
    ("DICOM Listener", "DICOM", "DICOM images and metadata"),
    ("JMS Listener", "JMS", "messages from a JMS/ActiveMQ queue"),
    ("Web Service Listener", "SOAP", "SOAP/WSDL web service requests"),
    ("Channel Reader", "Channel", "messages from another Mirth channel"),
]

# Destination connector types
DEST_TYPES = [
    ("TCP Sender", "MLLP", "send HL7 v2 over MLLP to a remote system"),
    ("HTTP Sender", "REST", "POST FHIR resources to a REST API endpoint"),
    ("File Writer", "File", "write transformed messages to a file system directory"),
    ("Database Writer", "JDBC", "INSERT/UPDATE records in a database"),
    ("SMTP Sender", "Email", "send email notifications or reports"),
    ("Document Writer", "PDF", "generate PDF documents from message data"),
    ("Channel Writer", "Channel", "route messages to another Mirth channel"),
    ("JMS Sender", "JMS", "publish messages to a JMS/ActiveMQ queue"),
    ("Web Service Sender", "SOAP", "call a SOAP/WSDL web service"),
]

# Clinical integration scenarios for channels
CHANNEL_SCENARIOS = [
    "ADT patient registration feed from Epic to a downstream analytics database",
    "lab results (ORU^R01) interface from a LIS to an EHR with result normalization",
    "pharmacy order (RDE^O11) routing from CPOE to multiple pharmacy systems",
    "radiology order routing from EHR to RIS with DICOM worklist generation",
    "patient transfer notifications between two hospitals in a health system",
    "clinical document (CCD/CDA) ingestion from an HIE and conversion to FHIR",
    "real-time vital signs streaming from bedside monitors to the EHR",
    "insurance eligibility verification via X12 270/271 with HL7 response",
    "public health reporting (ELR) for lab results to state DOH",
    "syndromic surveillance feed to CDC BioSense platform",
    "immunization submission (VXU^V04) to a state immunization registry (IIS)",
    "blood bank order interface between EHR and blood bank system",
    "discharge summary generation from ADT^A03 with CDA document creation",
    "appointment scheduling interface between patient portal and EHR",
    "medication reconciliation feed from pharmacy benefit manager to EHR",
    "patient merge/unmerge handling (ADT^A40/A41) across all downstream systems",
    "clinical trial patient screening alerts based on diagnosis codes",
    "pathology report routing from LIS to oncology EHR system",
    "home health referral orders from hospital EHR to home health agency system",
    "newborn screening results feed from state lab to hospital EHR",
]

# Transformer operations
TRANSFORMER_OPS = [
    "patient identifier cross-referencing (MPI/EMPI lookup)",
    "code translation (ICD-10 to SNOMED CT, CPT to HCPCS, local codes to standard)",
    "date/time format normalization across time zones",
    "patient name formatting (proper case, suffix/prefix handling)",
    "address standardization and geocoding",
    "insurance eligibility code mapping",
    "physician NPI lookup and validation",
    "message filtering based on patient location or service line",
    "duplicate message detection using control ID and hash",
    "PHI redaction for non-production environments",
    "HL7 segment reordering for destination system requirements",
    "custom Z-segment generation for proprietary data",
]

# Error handling patterns
ERROR_PATTERNS = [
    "dead letter queue with configurable retry count and exponential backoff",
    "error notification via email to the interface team with message context",
    "automatic retry with circuit breaker pattern for destination timeouts",
    "error channel routing for manual review with a web dashboard",
    "graceful degradation with store-and-forward when destination is down",
    "duplicate detection and idempotent processing to prevent double-sends",
]


def _build_mirth_prompts(count):
    """Build a list of (category, source_standard, prompt) tuples for Mirth channel tasks."""
    prompts = []

    # --- Full Channel XML (1,200) ---
    for _ in range(1200):
        scenario = random.choice(CHANNEL_SCENARIOS)
        src_name, src_proto, src_desc = random.choice(SOURCE_TYPES)
        dst_name, dst_proto, dst_desc = random.choice(DEST_TYPES)
        transformer = random.choice(TRANSFORMER_OPS)
        error = random.choice(ERROR_PATTERNS)
        prompts.append((
            "full_channel",
            "MirthConnect",
            f"Generate a COMPLETE Mirth Connect channel XML configuration for: {scenario}.\n\n"
            f"Source connector: {src_name} ({src_desc})\n"
            f"Destination connector: {dst_name} ({dst_desc})\n\n"
            f"The channel XML MUST include ALL of these sections:\n"
            f"1. <channel> root with id, name, description, revision, version\n"
            f"2. <sourceConnector> with transportName='{src_name}', properties for {src_proto}, "
            f"   inboundDataType, outboundDataType, and a <transformer> with JavaScript steps for: {transformer}\n"
            f"3. <destinationConnectors> with at least one <connector> including transportName='{dst_name}', "
            f"   properties for {dst_proto}, and a <responseTransformer>\n"
            f"4. <preprocessingScript> that validates incoming messages and logs receipt\n"
            f"5. <postprocessingScript> that updates processing statistics and triggers alerts\n"
            f"6. <deployScript> and <undeployScript> for channel lifecycle management\n"
            f"7. Error handling: {error}\n\n"
            f"Provide the COMPLETE channel XML -- not a snippet. Include realistic property values, "
            f"JavaScript transformer code, and production-ready error handling."
        ))

    # --- Multi-Destination Fan-Out (600) ---
    for _ in range(600):
        scenario = random.choice(CHANNEL_SCENARIOS)
        src_name, src_proto, src_desc = random.choice(SOURCE_TYPES)
        num_dests = random.choice([2, 3, 4, 5])
        dests = random.sample(DEST_TYPES, min(num_dests, len(DEST_TYPES)))
        dest_desc = "\n".join(
            f"   - Destination {i+1}: {d[0]} ({d[2]})" for i, d in enumerate(dests)
        )
        error = random.choice(ERROR_PATTERNS)
        prompts.append((
            "fan_out",
            "MirthConnect",
            f"Generate a COMPLETE Mirth Connect channel XML for a multi-destination fan-out pattern: "
            f"{scenario}.\n\n"
            f"Source: {src_name} ({src_desc})\n"
            f"Destinations ({num_dests} total):\n{dest_desc}\n\n"
            f"The channel XML MUST include:\n"
            f"1. <channel> root with full metadata\n"
            f"2. <sourceConnector> with {src_name} properties and inbound/outbound data types\n"
            f"3. <destinationConnectors> with {num_dests} separate <connector> elements, each with:\n"
            f"   - Unique name, transportName, and connector-specific properties\n"
            f"   - Individual <filter> rules to control which messages go to each destination\n"
            f"   - Individual <transformer> steps for destination-specific data mapping\n"
            f"4. <preprocessingScript> for message validation\n"
            f"5. <postprocessingScript> for aggregate status tracking across all destinations\n"
            f"6. Error handling: {error}\n\n"
            f"Provide the COMPLETE channel XML with realistic filter logic and transformer JavaScript."
        ))

    # --- Channel Chains (400) ---
    chain_scenarios = [
        ("ingestion", "normalization", "routing", "ADT messages from multiple source systems"),
        ("receive", "transform", "deliver", "lab results with code translation"),
        ("intake", "validate", "store", "clinical documents from an HIE"),
        ("listen", "enrich", "distribute", "orders with MPI lookup and code mapping"),
        ("capture", "deduplicate", "archive", "billing transactions with duplicate detection"),
    ]
    for _ in range(400):
        ch1, ch2, ch3, desc = random.choice(chain_scenarios)
        error = random.choice(ERROR_PATTERNS)
        prompts.append((
            "channel_chain",
            "MirthConnect",
            f"Generate COMPLETE Mirth Connect XML for a 3-channel chain pattern for {desc}.\n\n"
            f"Channel 1 ({ch1}): Receives raw messages, performs initial validation, "
            f"routes to Channel 2 via Channel Writer.\n"
            f"Channel 2 ({ch2}): Transforms/enriches messages, performs data mapping, "
            f"routes to Channel 3 via Channel Writer.\n"
            f"Channel 3 ({ch3}): Delivers to final destination(s) with acknowledgment handling.\n\n"
            f"For EACH of the 3 channels, provide COMPLETE XML including:\n"
            f"1. <channel> root with id, name referencing the chain (e.g., '{desc} - {ch1}')\n"
            f"2. <sourceConnector> (Channel 1: external source; Channels 2-3: Channel Reader)\n"
            f"3. <destinationConnectors> (Channels 1-2: Channel Writer; Channel 3: external destination)\n"
            f"4. <preprocessingScript> and <postprocessingScript>\n"
            f"5. Channel-to-channel routing using channelId references\n"
            f"6. Error handling across the chain: {error}\n\n"
            f"Provide ALL THREE complete channel XMLs, clearly separated and labeled."
        ))

    # --- Error Channel Patterns (400) ---
    for _ in range(400):
        scenario = random.choice(CHANNEL_SCENARIOS)
        error = random.choice(ERROR_PATTERNS)
        prompts.append((
            "error_channel",
            "MirthConnect",
            f"Generate a COMPLETE Mirth Connect error handling channel XML for: {scenario}.\n\n"
            f"This channel receives failed messages from other channels and implements: {error}\n\n"
            f"The channel XML MUST include:\n"
            f"1. <channel> root configured as a dedicated error handler\n"
            f"2. <sourceConnector> as Channel Reader (receives from global error routing)\n"
            f"3. <destinationConnectors> with:\n"
            f"   - Database Writer to log errors with full context (source channel, error message, "
            f"     original message, timestamp, retry count)\n"
            f"   - SMTP Sender for critical error alerts to the interface team\n"
            f"   - Channel Writer to re-queue messages for retry (with max retry check)\n"
            f"4. <preprocessingScript> that classifies error severity (CRITICAL, WARNING, INFO)\n"
            f"5. <postprocessingScript> that updates error dashboard metrics\n"
            f"6. Transformer logic for:\n"
            f"   - Error classification based on error code and destination\n"
            f"   - Retry eligibility check (max retries, circuit breaker state)\n"
            f"   - Alert throttling (don't spam emails for recurring errors)\n"
            f"   - Error context extraction and enrichment\n\n"
            f"Provide the COMPLETE channel XML with production-ready JavaScript in all transformers."
        ))

    # --- Production Templates (400) ---
    template_types = [
        ("HIPAA-compliant audit logging channel", "All message events are logged with user context, "
         "PHI is masked in logs, and audit records are written to a tamper-proof store"),
        ("high-availability failover channel", "Primary and secondary destinations with automatic "
         "failover, health checks, and reconnection logic"),
        ("message throttling/rate-limiting channel", "Controls message throughput to protect "
         "destination systems, with configurable rate limits and backpressure handling"),
        ("message archival and replay channel", "Archives all messages to a database with full "
         "metadata, supports replaying messages by date range or filter criteria"),
        ("data quality monitoring channel", "Validates messages against configurable rules, "
         "tracks data quality metrics, and alerts on quality degradation"),
        ("HL7 ACK/NAK generation channel", "Generates proper HL7 acknowledgment messages (ACK/NAK) "
         "based on processing results with appropriate error codes (AE, AR, CE, CR)"),
        ("FHIR Subscription notification channel", "Implements FHIR R4 Subscription with "
         "REST-hook callback, handles subscription lifecycle and notification delivery"),
        ("cross-environment message router", "Routes messages between DEV/TEST/STAGING/PROD "
         "environments with environment-specific transformations and PHI scrubbing for non-prod"),
    ]
    for _ in range(400):
        template_name, template_desc = random.choice(template_types)
        error = random.choice(ERROR_PATTERNS)
        prompts.append((
            "production_template",
            "MirthConnect",
            f"Generate a COMPLETE production-ready Mirth Connect channel XML template for: "
            f"{template_name}.\n\n"
            f"Description: {template_desc}\n\n"
            f"The channel XML MUST include:\n"
            f"1. <channel> root with comprehensive metadata and description\n"
            f"2. <sourceConnector> with appropriate connector type and configuration\n"
            f"3. <destinationConnectors> with all necessary destinations\n"
            f"4. <preprocessingScript> with input validation and message enrichment\n"
            f"5. <postprocessingScript> with completion handling and metrics\n"
            f"6. <deployScript> with initialization logic (DB connections, config loading)\n"
            f"7. <undeployScript> with cleanup logic (connection closing, final flush)\n"
            f"8. Detailed JavaScript in all transformer steps with comments\n"
            f"9. Error handling: {error}\n\n"
            f"This should be a drop-in template that a Mirth administrator can import and "
            f"configure with minimal changes. Include TODO comments for site-specific values."
        ))

    random.shuffle(prompts)
    return prompts[:count]


# ---------------------------------------------------------------------------
# Generation runners
# ---------------------------------------------------------------------------

def generate_translation(output_path, target_count=5000, num_workers=NUM_WORKERS):
    """Generate bidirectional standard translation examples."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_lines(output_path)
    if existing >= target_count:
        logger.info("Translation: already have %d/%d, skipping", existing, target_count)
        return existing

    remaining = target_count - existing
    logger.info("Translation: generating %d examples (%d existing)", remaining, existing)

    all_prompts = _build_translation_prompts(remaining)
    completed = [0]
    failed = [0]

    def process(item):
        category, source_std, prompt = item
        output = _call(prompt)
        if output and len(output) > 200:
            example = {
                "instruction": prompt,
                "output": output,
                "domain": "standard_translation",
                "source_standard": f"{category}|{source_std}",
                "version": "v3.5-translation",
            }
            _write(output_path, example)
            completed[0] += 1
            if completed[0] % 50 == 0:
                logger.info(
                    "Translation: %d/%d done (failed: %d)",
                    completed[0], remaining, failed[0],
                )
        else:
            failed[0] += 1

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(process, all_prompts))

    logger.info(
        "Translation complete: %d succeeded, %d failed", completed[0], failed[0]
    )
    return existing + completed[0]


def generate_mirth_channels(output_path, target_count=3000, num_workers=NUM_WORKERS):
    """Generate complete Mirth channel XML examples."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_lines(output_path)
    if existing >= target_count:
        logger.info("Mirth channels: already have %d/%d, skipping", existing, target_count)
        return existing

    remaining = target_count - existing
    logger.info("Mirth channels: generating %d examples (%d existing)", remaining, existing)

    all_prompts = _build_mirth_prompts(remaining)
    completed = [0]
    failed = [0]

    def process(item):
        category, source_std, prompt = item
        output = _call(prompt)
        if output and len(output) > 300:
            example = {
                "instruction": prompt,
                "output": output,
                "domain": "mirth_channel",
                "source_standard": f"{category}|{source_std}",
                "version": "v3.5-mirth-channel",
            }
            _write(output_path, example)
            completed[0] += 1
            if completed[0] % 50 == 0:
                logger.info(
                    "Mirth channels: %d/%d done (failed: %d)",
                    completed[0], remaining, failed[0],
                )
        else:
            failed[0] += 1

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(process, all_prompts))

    logger.info(
        "Mirth channels complete: %d succeeded, %d failed", completed[0], failed[0]
    )
    return existing + completed[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3.5 Day 3-4: Bidirectional Translation + Mirth Channel Generation"
    )
    parser.add_argument(
        "--translation-output",
        default="data/raw/v35_translation.jsonl",
        help="Output path for translation examples (default: data/raw/v35_translation.jsonl)",
    )
    parser.add_argument(
        "--mirth-output",
        default="data/raw/v35_mirth_channels.jsonl",
        help="Output path for Mirth channel examples (default: data/raw/v35_mirth_channels.jsonl)",
    )
    parser.add_argument(
        "--translation-count",
        type=int,
        default=5000,
        help="Number of translation examples to generate (default: 5000)",
    )
    parser.add_argument(
        "--mirth-count",
        type=int,
        default=3000,
        help="Number of Mirth channel examples to generate (default: 3000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of parallel workers (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip translation generation",
    )
    parser.add_argument(
        "--skip-mirth",
        action="store_true",
        help="Skip Mirth channel generation",
    )
    args = parser.parse_args()

    workers = args.workers

    logger.info("=== V3.5 Day 3-4: Translation + Mirth Channels ===")
    logger.info("Workers: %d | Endpoint: %s | Model: %s", workers, ENDPOINT, MODEL)

    if not args.skip_translation:
        generate_translation(args.translation_output, args.translation_count, workers)

    if not args.skip_mirth:
        generate_mirth_channels(args.mirth_output, args.mirth_count, workers)

    logger.info("=== Day 3-4 generation complete ===")
