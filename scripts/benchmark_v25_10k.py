#!/usr/bin/env python3
"""NexiFuse V2.5 Comprehensive Benchmark: 10,000 test prompts across 10 categories.

Generates 1,000 prompts per category using seed prompts + parametric expansion.
Supports resume: if results file exists, skips already-completed prompts.
"""

import json
import time
import random
import requests
import sys
import re
from pathlib import Path
from datetime import datetime

API_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "nexifuse-robust-expert"
RESULTS_FILE = Path("data/benchmark_v25_10k_results.json")
REPORT_FILE = Path("report_v25_benchmark_10k.md")
MAX_TOKENS = 512
TEMPERATURE = 0.1
TIMEOUT = 120

# ---------------------------------------------------------------------------
# Prompt templates: each category has seed templates that get expanded
# with random parameters to create 1,000 unique prompts per category
# ---------------------------------------------------------------------------

MIRTH_ACTIONS = [
    "Write a Mirth Connect JavaScript transformer that",
    "Create a Mirth Connect source transformer that",
    "Write a Mirth Connect destination transformer that",
    "Create a Mirth channel configuration that",
    "How do I configure a Mirth Connect channel to",
    "Write a Mirth preprocessor script that",
    "Create a Mirth filter that",
    "Write a Mirth postprocessor that",
    "How do I set up Mirth Connect to",
    "Write error handling code for a Mirth channel that",
]

MIRTH_OBJECTS = [
    "extracts patient demographics from an HL7 ADT {event} PID segment",
    "parses HL7 ORU R01 OBX segments into a flat key-value map",
    "converts a FHIR {resource} JSON to an HL7 {msg_type} message",
    "merges data from {source1} and {source2} messages into a single outbound message",
    "adds a custom Z-segment ({zseg}) with {data_type} information",
    "strips all PHI from HL7 messages before routing to a {dest}",
    "maps HL7 v{v1} messages to HL7 v{v2} format handling field differences",
    "validates all required fields in {seg1}, {seg2}, and {seg3} segments",
    "extracts {data_type} data from {segment} segments and creates a FHIR {resource} resource",
    "handles HL7 escape sequences and special characters in {field} fields",
    "routes {msg_type} messages to different destinations based on {criteria}",
    "implements message deduplication using {method}",
    "polls a {source_type} for {data_format} files and converts to HL7",
    "configures TLS/SSL for {direction} HL7 MLLP connections",
    "handles connection timeouts in {dest_type} destinations",
    "optimizes channel performance for {volume} msg/hour processing",
    "implements store-and-forward pattern for unreliable {dest_type}",
    "logs all {direction} messages for HIPAA audit compliance",
    "splits a FHIR Bundle into individual {resource} resources",
    "builds an HL7 ACK message with proper MSA segment based on {criteria}",
    "converts CDA XML sections to individual HL7 v2 {segment} segments",
    "implements a dead letter queue for messages that fail processing {n} times",
    "rate-limits messages to {rate} per minute per sending facility",
    "enriches incoming messages with data from an external {api_type} API",
    "handles batch HL7 messages (BHS/BTS wrapped) processing each individually",
]

HL7_PROMPTS_SEEDS = [
    "Generate an HL7 v2.{ver} {msg_type} {event} message for {scenario}",
    "Write a parser that handles HL7 v2 {feature} in {language}",
    "Explain the structure of HL7 v2 {segment} segment with all field definitions",
    "Write code to validate HL7 v2 {msg_type} messages checking {validation}",
    "How do you handle HL7 v2 {feature} in production systems?",
    "Write code to convert HL7 v2 {segment} to FHIR {resource} in {language}",
    "Create an HL7 v2 {msg_type} message builder in {language}",
    "Write code to handle HL7 v2 {feature} edge cases",
    "Explain the difference between HL7 v2 {concept1} and {concept2}",
    "Write a {language} function to extract {data} from HL7 v2 {segment} segment",
]

FHIR_PROMPTS_SEEDS = [
    "Create a FHIR R4 {resource} resource for {scenario}",
    "Write code to implement FHIR R4 {operation} operation on {resource} resources in {language}",
    "Write a FHIR R4 search query to find {resource} with {criteria}",
    "Explain the FHIR R4 {concept} and how it applies to {use_case}",
    "Create a FHIR R4 Bundle of type {bundle_type} containing {resources}",
    "Write code to validate a FHIR R4 {resource} against {profile} profile in {language}",
    "How do you implement FHIR R4 {feature} in a production server?",
    "Write a FHIR R4 {resource} to HL7 v2 {segment} converter in {language}",
    "Create a FHIR R4 CapabilityStatement for a server supporting {resources}",
    "Write code to handle FHIR R4 {feature} with proper error handling in {language}",
]

EHR_PROMPTS_SEEDS = [
    "Write code to implement {vendor} FHIR API {operation} in {language}",
    "How do you handle {vendor} OAuth {flow} authentication flow?",
    "Write code to {action} using {vendor}'s {api_type} API in {language}",
    "How do you troubleshoot {vendor} FHIR API {error} errors?",
    "Write code to implement SMART on FHIR {feature} for {vendor}",
    "Create a {vendor} integration that {action} with proper error handling",
    "How do you handle {vendor}'s rate limiting and throttling?",
    "Write code to sync {data_type} data between {vendor} and {dest}",
    "How do you implement bulk data export using {vendor}'s FHIR API?",
    "Write a patient matching service using {vendor}'s {api_type} API",
]

DEBUG_PROMPTS_SEEDS = [
    "Mirth Connect {component} is {symptom}. {context}. How do I diagnose and fix this?",
    "My HL7 {msg_type} messages are {symptom}. {context}. What's the root cause?",
    "FHIR API {endpoint} returns {error}. {context}. How do I troubleshoot?",
    "My {system} integration is {symptom} after {event}. What should I check?",
    "Performance issue: {component} is {symptom}. {metrics}. How do I optimize?",
    "Data quality issue: {data_type} has {symptom}. How do I identify and fix affected records?",
    "Authentication failing for {vendor} API: {error}. {context}. How do I resolve?",
    "Message ordering is {symptom} in {component}. How do I ensure correct sequencing?",
    "My {component} stopped working after {event}. Logs show {log_msg}. What's wrong?",
    "Intermittent failures in {component}: {symptom} every {frequency}. How do I diagnose?",
]

CONCEPTUAL_PROMPTS_SEEDS = [
    "Explain the difference between {concept1} and {concept2} in healthcare interoperability",
    "What is {concept} and how does it apply to healthcare integration?",
    "Compare {approach1} vs {approach2} for {use_case}",
    "What are the security implications of {topic} in healthcare systems?",
    "Explain {standard} and its role in {context}",
    "What is the {pattern} pattern and why is it important for healthcare integration?",
    "How does {regulation} affect healthcare data exchange?",
    "What are best practices for {topic} in healthcare IT?",
    "Explain the architecture of {system} and how it integrates with {other}",
    "What is the future of {topic} in healthcare interoperability?",
]

GENERAL_CODE_SEEDS = [
    "Write a {language} function that {action}",
    "Implement a {pattern} in {language} with proper error handling",
    "Create a {language} class that {action}",
    "Write a {framework} endpoint that {action}",
    "Write a SQL query to {action} in a {db_type} database",
    "Create a {language} script that {action}",
    "Write a {tool} configuration for {action}",
    "Implement {algorithm} in {language} with O({complexity}) complexity",
    "Write a {test_type} test for {component} in {language}",
    "Create a {language} utility that {action}",
]

MATH_PROMPTS_SEEDS = [
    "Calculate {expr}. Show your work step by step.",
    "A hospital processes {n} {msg_type} messages per {period}. Each message averages {size}. {question}",
    "If system uptime is {uptime}%, how many {unit} of downtime per {period}?",
    "{n} servers each handle {rate} requests/sec. With {redundancy} redundancy, what's the max throughput?",
    "Network latency is {lat}ms. {n} API calls are needed. What's the total round-trip time?",
    "Storage grows at {rate} per {period}. Current usage is {current}. When will it reach {limit}?",
    "Convert {value} {unit1} to {unit2}. Show your calculation.",
    "A FHIR server uses {mem} per {n} resources. For {total} resources, how much {resource_type} is needed?",
    "Message queue has {n} messages. Processing rate is {rate}/sec. How long to clear the queue?",
    "If {n}% of messages fail validation, and you process {total} per day, how many need manual review per {period}?",
]

IDENTITY_PROMPTS_SEEDS = [
    "What is your name?", "Who are you?", "What are you?", "Tell me about yourself",
    "Are you ChatGPT?", "Are you GPT-4?", "Are you Claude?", "Are you Gemini?",
    "What can you help with?", "What do you specialize in?",
    "Hello!", "Hi there", "Good morning", "Thanks for your help",
    "What model are you based on?", "How were you trained?",
    "Can you write code?", "Do you know about FHIR?", "Can you help with Mirth Connect?",
    "What's your context window?", "Are you open source?",
    "Can you help with non-healthcare tasks?", "What languages do you support?",
    "Who created you?", "What version are you?",
    "Are you HIPAA compliant?", "Can you access patient data?",
    "How accurate are you?", "What are your limitations?",
    "Can you learn from our conversations?", "Do you store our conversation?",
    "What's your response time?", "Can you work offline?",
    "How do you compare to other AI assistants?",
    "Can you generate HL7 messages?", "Can you debug Mirth channels?",
    "Can you create FHIR resources?", "Can you help with Epic integration?",
    "What healthcare standards do you know?", "Can you explain IHE profiles?",
]

SECURITY_PROMPTS_SEEDS = [
    "Write code to {action} for PHI protection in {language}",
    "How do you implement {security_feature} for healthcare {system}?",
    "Create a {language} audit logger for {event_type} events with HIPAA compliance",
    "Write code to {action} using {encryption} encryption in {language}",
    "How do you configure {component} for {compliance} compliance?",
    "Write a security test that checks for {vulnerability} in {component}",
    "Create a {language} function for {security_action} of {data_type}",
    "How do you implement {auth_type} authentication for {system}?",
    "Write code to monitor {metric} for security anomaly detection",
    "How do you handle {incident_type} in a healthcare integration environment?",
]

# ---------------------------------------------------------------------------
# Parameter pools for template expansion
# ---------------------------------------------------------------------------

PARAMS = {
    "event": ["A01", "A02", "A03", "A04", "A08", "A11", "A12", "A13", "A28", "A31", "A34", "A40"],
    "resource": ["Patient", "Encounter", "Observation", "Condition", "MedicationRequest", "Procedure",
                  "AllergyIntolerance", "DiagnosticReport", "Immunization", "CarePlan", "ServiceRequest",
                  "Practitioner", "Organization", "Location", "Device", "Coverage", "Claim",
                  "ExplanationOfBenefit", "DocumentReference", "Provenance", "Consent", "Goal"],
    "msg_type": ["ADT", "ORU", "ORM", "SIU", "RDE", "MDM", "DFT", "BAR", "VXU", "MFN"],
    "segment": ["PID", "PV1", "OBX", "OBR", "MSH", "NK1", "IN1", "DG1", "RXA", "AL1", "GT1", "EVN", "SCH", "AIG"],
    "language": ["Python", "JavaScript", "Java", "TypeScript", "C#"],
    "vendor": ["Epic", "Cerner", "Athena", "MEDITECH", "Allscripts"],
    "source1": ["ADT", "ORU", "ORM"], "source2": ["ORU", "SIU", "DFT"],
    "zseg": ["ZPI", "ZDX", "ZIN", "ZRF", "ZCD"],
    "data_type": ["insurance", "diagnosis", "medication", "allergy", "immunization", "lab result", "vital signs"],
    "dest": ["research database", "data warehouse", "analytics platform", "reporting system", "billing system"],
    "v1": ["2.3", "2.3.1", "2.4"], "v2": ["2.5", "2.5.1", "2.7", "2.8"],
    "seg1": ["PID", "MSH", "OBR"], "seg2": ["PV1", "NK1", "OBX"], "seg3": ["IN1", "DG1", "AL1"],
    "field": ["OBX.5", "PID.11", "NK1.2", "IN1.3", "PV1.7", "MSH.9"],
    "criteria": ["MSH.9 event type", "MSH.4 sending facility", "PID.3 patient ID", "message priority"],
    "method": ["MSH.10 Message Control ID lookup", "content hash comparison", "database key check"],
    "source_type": ["directory", "SFTP server", "S3 bucket", "database table", "REST API"],
    "data_format": ["CSV", "JSON", "XML", "HL7", "FHIR Bundle", "CDA"],
    "direction": ["inbound", "outbound", "bidirectional"],
    "dest_type": ["TCP", "HTTP", "SFTP", "database", "REST API", "SOAP"],
    "volume": ["1,000", "5,000", "10,000", "50,000", "100,000"],
    "rate": ["100", "500", "1,000", "5,000"],
    "n": ["3", "5", "10", "50", "100", "1,000", "10,000"],
    "api_type": ["FHIR", "REST", "SOAP", "proprietary", "bulk data"],
    "ver": ["3", "3.1", "4", "5", "5.1", "7", "8"],
    "scenario": [
        "a patient admission with insurance and next-of-kin",
        "a lab result with numeric and text observations",
        "a medication order with dosage instructions",
        "a scheduled appointment with practitioner details",
        "a patient discharge with diagnosis summary",
        "a patient transfer between departments",
        "an emergency department visit",
        "a vaccination administration record",
        "a radiology order with contrast instructions",
        "a pathology report with multiple specimens",
    ],
    "feature": ["repeating segments", "escape sequences", "continuation segments", "batch processing",
                 "message acknowledgments", "segment groups", "Z-segments", "encoding characters",
                 "message profiles", "conformance claims"],
    "validation": ["required fields", "data types", "code tables", "segment ordering", "field lengths"],
    "concept1": ["HL7 v2", "FHIR REST", "CDA", "ESB", "point-to-point", "synchronous"],
    "concept2": ["FHIR R4", "FHIR Messaging", "FHIR Documents", "API gateway", "hub-and-spoke", "asynchronous"],
    "data": ["patient name", "MRN", "date of birth", "address", "phone number", "insurance ID"],
    "operation": ["$validate", "$everything", "$match", "$translate", "$expand", "search", "create", "update"],
    "bundle_type": ["transaction", "searchset", "document", "message", "batch", "collection"],
    "resources": ["Patient and Encounter", "Observation and DiagnosticReport", "MedicationRequest and Medication"],
    "profile": ["US Core", "Da Vinci", "IHE MHD", "CARIN Blue Button", "Argonaut"],
    "flow": ["authorization code", "client credentials", "JWT bearer", "PKCE"],
    "action": [
        "fetch patient demographics", "search for encounters", "create observations",
        "update medication records", "bulk export data", "validate resources",
        "sync patient records", "query lab results", "manage appointments",
        "process claims", "generate reports", "monitor interfaces",
    ],
    "error": ["401 Unauthorized", "403 Forbidden", "404 Not Found", "422 Unprocessable Entity",
              "429 Too Many Requests", "500 Internal Server Error", "503 Service Unavailable"],
    "endpoint": ["/Patient", "/Encounter", "/Observation", "/MedicationRequest", "/Condition"],
    "component": ["channel", "transformer", "destination", "source connector", "filter", "database writer",
                   "TCP listener", "HTTP sender", "message queue", "thread pool"],
    "symptom": [
        "extremely slow", "dropping messages", "returning errors intermittently",
        "consuming excessive memory", "failing silently", "producing duplicate records",
        "timing out frequently", "corrupting data", "losing message ordering",
        "rejecting valid messages", "not processing after restart",
    ],
    "context": [
        "This started after a system upgrade", "No configuration changes were made recently",
        "The issue only occurs during peak hours", "It works in dev but fails in production",
        "Logs show no errors but messages are missing", "CPU usage spikes to 100% periodically",
    ],
    "system": ["Mirth Connect", "Epic", "Cerner", "HL7 interface", "FHIR server", "integration engine"],
    "metrics": ["CPU at 95%, memory at 80%", "Queue depth growing by 1000/hour", "Response time >30 seconds"],
    "event": ["upgrade", "config change", "certificate renewal", "network migration", "failover test"],
    "log_msg": ["Connection refused", "Timeout exceeded", "OutOfMemoryError", "SSL handshake failed"],
    "frequency": ["5 minutes", "1 hour", "randomly", "during batch processing", "under heavy load"],
    "concept": [
        "TEFCA", "Carequality", "CommonWell", "Direct messaging", "USCDI",
        "Clinical Decision Support", "SMART on FHIR", "Bulk Data Access",
        "IHE XDS.b", "IHE PIX/PDQ", "HL7 CCDA", "openEHR", "OMOP CDM",
    ],
    "approach1": ["microservices", "ESB", "point-to-point", "API gateway", "event-driven"],
    "approach2": ["monolith", "message broker", "hub-and-spoke", "service mesh", "batch processing"],
    "use_case": ["patient data exchange", "lab result routing", "medication reconciliation",
                 "care coordination", "claims processing", "clinical trials"],
    "standard": ["IHE XDS.b", "IHE PIX", "IHE PDQ", "IEEE 11073", "NCPDP SCRIPT", "X12 835/837"],
    "pattern": ["circuit breaker", "saga", "CQRS", "event sourcing", "retry with backoff", "bulkhead"],
    "regulation": ["HIPAA", "21st Century Cures Act", "GDPR", "HITRUST", "SOC 2", "ONC regulations"],
    "topic": ["API security", "data governance", "real-time monitoring", "disaster recovery",
              "cloud migration", "AI in healthcare", "patient matching", "consent management"],
    "framework": ["FastAPI", "Flask", "Express", "Spring Boot", "Django"],
    "db_type": ["PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch"],
    "pattern_code": ["singleton", "factory", "observer", "strategy", "decorator", "adapter"],
    "algorithm": ["binary search", "BFS", "DFS", "Dijkstra", "A*", "merge sort", "LRU cache"],
    "complexity": ["1", "log n", "n", "n log n", "n^2"],
    "tool": ["Docker", "Kubernetes", "Terraform", "GitHub Actions", "Prometheus", "Grafana"],
    "test_type": ["unit", "integration", "end-to-end", "load", "security", "regression"],
    "expr": [
        "247 * 18 + 356", "1547 * 23 + 891", "2^16 - 1", "987654 / 321",
        "log2(1048576)", "sqrt(144) + sqrt(256)", "15! / (10! * 5!)",
        "0.99^365", "(3/7) + (2/5)", "sum of integers from 1 to 100",
    ],
    "period": ["hour", "day", "week", "month", "year"],
    "size": ["2 KB", "3.2 KB", "5 KB", "10 KB", "50 KB"],
    "question": [
        "How much storage is needed for 30 days of archival?",
        "What bandwidth is required?",
        "How many processing threads are needed?",
        "What's the peak throughput requirement?",
    ],
    "uptime": ["99.9", "99.99", "99.95", "99.999"],
    "unit": ["minutes", "hours"],
    "lat": ["5", "10", "20", "50", "100"],
    "redundancy": ["N+1", "2N", "active-passive", "active-active"],
    "current": ["500 GB", "1 TB", "2 TB", "5 TB"],
    "limit": ["10 TB", "20 TB", "50 TB"],
    "mem": ["500MB RAM", "1GB RAM", "2GB RAM"],
    "total": ["100,000", "500,000", "1,000,000", "10,000,000"],
    "resource_type": ["RAM", "disk", "CPU cores"],
    "value": ["1024", "2048", "4096", "8192"],
    "unit1": ["MB", "KB", "GB"], "unit2": ["GB", "MB", "TB"],
    "security_feature": ["mutual TLS", "OAuth 2.0", "RBAC", "ABAC", "field-level encryption",
                          "data masking", "audit logging", "intrusion detection"],
    "encryption": ["AES-256", "RSA-2048", "TLS 1.3", "SHA-256"],
    "compliance": ["HIPAA", "HITRUST", "SOC 2", "GDPR", "NIST"],
    "event_type": ["PHI access", "authentication", "data export", "configuration change"],
    "vulnerability": ["SQL injection", "XSS", "CSRF", "path traversal", "insecure deserialization"],
    "security_action": ["tokenization", "de-identification", "redaction", "anonymization", "pseudonymization"],
    "auth_type": ["OAuth 2.0", "SAML", "JWT", "mTLS", "API key", "SMART on FHIR"],
    "metric": ["failed login attempts", "unusual data access patterns", "API rate anomalies"],
    "incident_type": ["data breach", "unauthorized access", "PHI exposure", "ransomware", "DDoS attack"],
}


def fill_template(template: str) -> str:
    """Fill a template string with random parameters."""
    result = template
    for key, values in PARAMS.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def generate_prompts(category: str, seeds: list, count: int = 1000) -> list[str]:
    """Generate `count` unique prompts from seed templates."""
    prompts = set()
    attempts = 0
    max_attempts = count * 20

    while len(prompts) < count and attempts < max_attempts:
        seed = random.choice(seeds)
        prompt = fill_template(seed)
        prompts.add(prompt)
        attempts += 1

    result = list(prompts)[:count]
    random.shuffle(result)
    return result


def generate_mirth_prompts(count=1000):
    seeds = [f"{a} {o}" for a in MIRTH_ACTIONS for o in MIRTH_OBJECTS]
    return generate_prompts("mirth_connect", seeds, count)

def generate_hl7_prompts(count=1000):
    return generate_prompts("hl7v2", HL7_PROMPTS_SEEDS, count)

def generate_fhir_prompts(count=1000):
    return generate_prompts("fhir_r4", FHIR_PROMPTS_SEEDS, count)

def generate_ehr_prompts(count=1000):
    return generate_prompts("ehr_api", EHR_PROMPTS_SEEDS, count)

def generate_debug_prompts(count=1000):
    return generate_prompts("multi_turn_debugging", DEBUG_PROMPTS_SEEDS, count)

def generate_conceptual_prompts(count=1000):
    return generate_prompts("conceptual_explanation", CONCEPTUAL_PROMPTS_SEEDS, count)

def generate_general_code_prompts(count=1000):
    return generate_prompts("general_coding", GENERAL_CODE_SEEDS, count)

def generate_math_prompts(count=1000):
    return generate_prompts("math_reasoning", MATH_PROMPTS_SEEDS, count)

def generate_identity_prompts(count=1000):
    # Identity uses fixed prompts with repetition + variations
    base = IDENTITY_PROMPTS_SEEDS[:]
    variations = [
        "Tell me, " + p.lower() if not p.startswith("Tell") else p for p in base
    ] + [
        "I'm curious, " + p.lower() if p.endswith("?") else p for p in base
    ] + [
        "Quick question: " + p.lower() for p in base if p.endswith("?")
    ] + [
        "Please " + p.lower().rstrip("?!.") for p in base
    ]
    all_prompts = list(set(base + variations))
    random.shuffle(all_prompts)
    # Repeat to reach count
    while len(all_prompts) < count:
        all_prompts.extend(all_prompts[:count - len(all_prompts)])
    return all_prompts[:count]

def generate_security_prompts(count=1000):
    return generate_prompts("security_compliance", SECURITY_PROMPTS_SEEDS, count)


def call_api(prompt: str, timeout: int = TIMEOUT) -> dict:
    """Call the NexiFuse API and return result dict."""
    start = time.time()
    try:
        resp = requests.post(API_URL, json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are NexiFuse, a healthcare integration expert."},
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        latency = time.time() - start
        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})
        return {
            "response": content,
            "latency_sec": round(latency, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "status": "ok",
        }
    except Exception as e:
        return {
            "response": "",
            "latency_sec": round(time.time() - start, 2),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "status": f"error: {str(e)[:200]}",
        }


def run_benchmark():
    """Run the full 10,000-prompt benchmark with resume support."""
    random.seed(42)  # Reproducible prompt generation

    # Generate all prompts
    print("Generating 10,000 test prompts...")
    categories = {
        "mirth_connect": generate_mirth_prompts(1000),
        "hl7v2": generate_hl7_prompts(1000),
        "fhir_r4": generate_fhir_prompts(1000),
        "ehr_api": generate_ehr_prompts(1000),
        "multi_turn_debugging": generate_debug_prompts(1000),
        "conceptual_explanation": generate_conceptual_prompts(1000),
        "general_coding": generate_general_code_prompts(1000),
        "math_reasoning": generate_math_prompts(1000),
        "identity": generate_identity_prompts(1000),
        "security_compliance": generate_security_prompts(1000),
    }

    total_prompts = sum(len(v) for v in categories.values())
    print(f"Generated {total_prompts} prompts across {len(categories)} categories")

    # Resume support: load existing results
    existing_results = []
    completed_keys = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing_results = json.load(f)
        for r in existing_results:
            completed_keys.add((r["category"], r["index"]))
        print(f"Resuming: {len(existing_results)} already completed")

    results = existing_results[:]
    total_done = len(results)
    total_errors = sum(1 for r in results if r["status"] != "ok")

    print(f"\nNexiFuse V2.5 Benchmark: 10,000 prompts across 10 categories")
    print("=" * 70)

    for cat_name, prompts in categories.items():
        cat_done = sum(1 for r in results if r["category"] == cat_name)
        cat_total = len(prompts)
        print(f"\n=== {cat_name} ({cat_done}/{cat_total} done) ===")

        for i, prompt in enumerate(prompts):
            if (cat_name, i) in completed_keys:
                continue

            result = call_api(prompt)
            entry = {
                "category": cat_name,
                "index": i,
                "prompt": prompt[:120],
                "response_length": len(result["response"]),
                "latency_sec": result["latency_sec"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "has_code": "```" in result["response"],
                "response_preview": result["response"][:300],
                "full_response": result["response"],
                "status": result["status"],
            }
            results.append(entry)
            total_done += 1

            if result["status"] != "ok":
                total_errors += 1

            # Progress every 100 prompts
            if (total_done) % 100 == 0:
                print(f"  [{total_done}/10000] {result['status'][:2].upper()} "
                      f"{result['latency_sec']:.1f}s {len(result['response'])} chars "
                      f"-- {prompt[:60]}...")

            # Save every 200 prompts (resume point)
            if total_done % 200 == 0:
                RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)

    # Final save
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    ok_results = [r for r in results if r["status"] == "ok"]
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK COMPLETE: {len(ok_results)}/{total_done} passed, {total_errors} errors")
    print(f"{'=' * 70}")

    if ok_results:
        avg_lat = sum(r["latency_sec"] for r in ok_results) / len(ok_results)
        avg_len = sum(r["response_length"] for r in ok_results) / len(ok_results)
        avg_tok = sum(r["completion_tokens"] for r in ok_results) / len(ok_results)
        code_pct = sum(1 for r in ok_results if r["has_code"]) / len(ok_results) * 100
        print(f"Avg latency:  {avg_lat:.2f}s")
        print(f"Avg response: {avg_len:.0f} chars / {avg_tok:.0f} tokens")
        print(f"Code blocks:  {code_pct:.0f}%")

    # Per-category breakdown
    print(f"\nPer-category breakdown:")
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat and r["status"] == "ok"]
        cat_errors = sum(1 for r in results if r["category"] == cat and r["status"] != "ok")
        if cat_results:
            cat_lat = sum(r["latency_sec"] for r in cat_results) / len(cat_results)
            cat_len = sum(r["response_length"] for r in cat_results) / len(cat_results)
            cat_code = sum(1 for r in cat_results if r["has_code"]) / len(cat_results) * 100
            print(f"  {cat:30s} {len(cat_results):>4}/{len(cat_results)+cat_errors} ok "
                  f" lat={cat_lat:.1f}s  len={cat_len:.0f}  code={cat_code:.0f}%")

    print(f"\nResults saved to {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    run_benchmark()
