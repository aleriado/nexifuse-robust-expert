#!/usr/bin/env python3
"""NexiFuse v2 Benchmark — 100 test prompts across 10 categories."""

import json
import time
import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "nexifuse-robust-expert"

# 10 categories × 10 prompts = 100 total
TEST_PROMPTS = {
    "mirth_connect": [
        "Write a Mirth Connect JavaScript transformer that extracts patient demographics from an HL7 ADT A01 message and maps them to a JSON object.",
        "Create a Mirth Connect channel filter that drops messages where PID.3 (patient ID) is empty.",
        "Write a Mirth Connect JavaScript destination transformer that converts an HL7 ORU R01 result to a FHIR DiagnosticReport resource.",
        "How do I configure a Mirth Connect HTTP Listener source to receive FHIR Bundle resources?",
        "Write a Mirth Connect JavaScript transformer to merge two PID segments from different ADT messages into a single patient record.",
        "Create a Mirth Connect preprocessor script that validates incoming HL7 messages have valid MSH segment encoding characters.",
        "Write a Mirth Connect channel that routes HL7 messages to different destinations based on MSH.9 message type.",
        "How do I handle HL7 acknowledgment (ACK/NACK) responses in a Mirth Connect channel?",
        "Write a Mirth Connect JavaScript transformer that adds a custom Z-segment to an outgoing HL7 message.",
        "Create a Mirth Connect deployment script that exports channel configurations as XML.",
    ],
    "hl7v2": [
        "Generate a valid HL7 v2.5 ADT A01 message for a patient admission with full PID, PV1, and IN1 segments.",
        "Explain the structure of an HL7 v2 ORU R01 message and list all required segments.",
        "Write a parser that extracts all OBX segments from an HL7 ORU R01 message and returns them as a list.",
        "What is the difference between HL7 v2 encoding characters ^~\\& and how are they used in field repetitions?",
        "Generate an HL7 v2.5 ORM O01 order message for a CBC lab test.",
        "How do you handle HL7 v2 message fragmentation when messages exceed the maximum segment length?",
        "Write code to validate that an HL7 v2 ADT message contains all required segments for an A08 update.",
        "Explain HL7 v2 acknowledgment modes: original mode vs enhanced mode. When should each be used?",
        "Generate an HL7 v2 SIU S12 scheduling message for a new appointment.",
        "Write a function that converts HL7 v2 datetime format (YYYYMMDDHHMMSS) to ISO 8601.",
    ],
    "fhir_r4": [
        "Create a FHIR R4 Patient resource JSON with name, DOB, gender, address, and two identifiers (MRN and SSN).",
        "Write a FHIR R4 search query to find all Observations for a patient with a specific LOINC code.",
        "Create a FHIR R4 Bundle of type 'transaction' that creates a Patient and links an Encounter to it.",
        "Explain the difference between FHIR R4 contained resources and referenced resources. When should each be used?",
        "Write a FHIR R4 Medication Administration resource for an IV medication dose.",
        "How do you implement FHIR R4 pagination using Bundle links (next, previous)?",
        "Create a FHIR R4 Questionnaire resource for a patient intake form with 5 questions.",
        "Write code to validate a FHIR R4 resource against its StructureDefinition.",
        "Create a FHIR R4 Provenance resource that tracks who created a Patient resource and when.",
        "Explain FHIR R4 Capability Statement and write one for a simple Patient CRUD server.",
    ],
    "ehr_api": [
        "Write Python code to authenticate with the Epic FHIR API using OAuth 2.0 client credentials flow.",
        "How do you query patient data from Cerner's FHIR R4 API? Show the full request with headers.",
        "Write a script that syncs patient allergies between two EHR systems using FHIR APIs.",
        "Explain the SMART on FHIR launch sequence for an EHR-embedded application.",
        "Write Python code to bulk export patient data from a FHIR server using the $export operation.",
        "How do you handle rate limiting when making bulk API calls to Epic's FHIR endpoint?",
        "Write code to create a new encounter in an EHR system via FHIR API with proper authorization.",
        "Explain the difference between Epic's proprietary API and their FHIR API. When should each be used?",
        "Write a webhook handler that processes CDS Hooks requests from an EHR system.",
        "How do you implement single sign-on (SSO) between a custom healthcare app and an EHR using SAML?",
    ],
    "multi_turn_debugging": [
        "I have a Mirth Connect channel that's dropping messages silently. The source is a TCP listener and destination is a database writer. Where should I start debugging?",
        "My HL7 parser is throwing an error on messages with repeating fields in PID.3. The error says 'unexpected delimiter'. How do I fix this?",
        "I'm getting a 401 Unauthorized error when calling the Epic FHIR API. I've already set up my OAuth app. What could be wrong?",
        "My FHIR Bundle transaction is failing with a 422 error. The Patient resource validates fine individually. What's happening?",
        "Mirth Connect is running out of memory after processing about 10,000 messages. How do I diagnose and fix the memory leak?",
        "My HL7 to FHIR converter is producing invalid JSON for messages with special characters in patient names. How do I handle this?",
        "I'm seeing duplicate messages in my Mirth Connect destination. The source is a file reader polling a directory. What's causing this?",
        "My FHIR subscription is not receiving notifications when new resources are created. How do I troubleshoot?",
        "I have an HL7 ADT feed that sometimes sends messages out of order (A02 before A01). How should I handle this in my integration?",
        "My Mirth Connect channel works in development but fails in production. The error is 'connection refused' on the destination. What should I check?",
    ],
    "conceptual_explanation": [
        "Explain the key differences between HL7 v2, HL7 v3, and FHIR. Why did the industry move from v2 to FHIR?",
        "What is IHE (Integrating the Healthcare Enterprise) and how do IHE profiles relate to HL7 standards?",
        "Explain the concept of healthcare data interoperability and the challenges hospitals face in achieving it.",
        "What is the role of a healthcare integration engine like Mirth Connect in a hospital IT architecture?",
        "Explain HIPAA requirements for data in transit and at rest in the context of HL7 messaging.",
        "What is the difference between point-to-point integration and an enterprise service bus (ESB) pattern in healthcare?",
        "Explain the concept of clinical data repositories (CDR) and how they relate to FHIR APIs.",
        "What are the main security considerations when implementing FHIR APIs in a healthcare environment?",
        "Explain the difference between FHIR REST, FHIR Messaging, and FHIR Documents paradigms.",
        "What is the role of terminologies (SNOMED CT, LOINC, ICD-10) in healthcare interoperability?",
    ],
    "general_coding": [
        "Write a Python function that validates an email address using regex.",
        "Implement a binary search algorithm in JavaScript.",
        "Write a SQL query to find the top 5 most prescribed medications grouped by department.",
        "Create a REST API endpoint in Python Flask that accepts JSON and returns a filtered response.",
        "Write a function to convert a CSV file to JSON in Python.",
        "Implement a simple rate limiter using the token bucket algorithm in Python.",
        "Write a JavaScript function that debounces API calls with a configurable delay.",
        "Create a Docker compose file for a Node.js app with a PostgreSQL database.",
        "Write a Python script that monitors a directory for new files and processes them.",
        "Implement a simple LRU cache in Python with O(1) get and put operations.",
    ],
    "math_reasoning": [
        "What is 1547 * 23 + 891? Show your work step by step.",
        "A hospital processes 450 HL7 messages per minute. How many messages per day? Per year?",
        "If a FHIR server can handle 1000 requests/second and each request takes 50ms, what's the maximum concurrent connections?",
        "Calculate the storage needed for 10 million FHIR Patient resources if each averages 2KB.",
        "A Mirth channel processes messages with 99.9% uptime. How many minutes of downtime per year?",
        "If network latency is 5ms and a FHIR Bundle contains 100 resources, each requiring a separate API call at 20ms each, what's the total time?",
        "What is the probability of a collision in a UUID v4 after generating 1 billion IDs?",
        "A hospital has 5 EHR systems each sending 200 messages/hour. If the integration engine has a 500 msg/hr capacity, is it sufficient?",
        "Calculate the base64 encoding overhead for a 10MB CDA document.",
        "If HL7 message processing has a 2% error rate and we process 100,000 messages, how many will fail?",
    ],
    "identity": [
        "Who are you?",
        "What is your name?",
        "What can you help me with?",
        "Are you ChatGPT?",
        "What model are you based on?",
        "Tell me about yourself.",
        "What are your capabilities?",
        "Are you an AI assistant?",
        "Who created you?",
        "What is NexiFuse?",
    ],
    "security_compliance": [
        "Write a Mirth Connect preprocessor that sanitizes PHI from log messages before writing to the server log.",
        "How do you implement TLS mutual authentication for HL7 MLLP connections?",
        "Write a FHIR AuditEvent resource for tracking unauthorized access attempts to patient records.",
        "Explain how to configure Mirth Connect to encrypt messages at rest in the internal database.",
        "Write code to de-identify a FHIR Patient resource for research purposes following Safe Harbor guidelines.",
        "How do you implement role-based access control (RBAC) for a FHIR API server?",
        "Write a script that detects and flags potential PHI in free-text OBX segments of HL7 messages.",
        "Explain the HITRUST CSF framework and how it applies to healthcare integration systems.",
        "Write a Mirth Connect channel that logs all message access events for HIPAA audit trail compliance.",
        "How do you securely transmit HL7 messages over the internet between two hospitals?",
    ],
}

def test_prompt(prompt, category, idx):
    """Send a prompt to the API and return results."""
    start = time.time()
    try:
        resp = requests.post(API_URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 512,
        }, timeout=180)
        elapsed = time.time() - start
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "category": category,
            "index": idx,
            "prompt": prompt[:100],
            "response_length": len(content),
            "latency_sec": round(elapsed, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "has_code": "```" in content,
            "response_preview": content[:200],
            "full_response": content,
            "status": "ok",
        }
    except Exception as e:
        return {
            "category": category,
            "index": idx,
            "prompt": prompt[:100],
            "status": "error",
            "error": str(e),
            "latency_sec": round(time.time() - start, 2),
        }

def main():
    results = []
    total = sum(len(v) for v in TEST_PROMPTS.values())
    done = 0

    for category, prompts in TEST_PROMPTS.items():
        print(f"\n=== {category} ({len(prompts)} prompts) ===")
        for i, prompt in enumerate(prompts):
            done += 1
            result = test_prompt(prompt, category, i)
            results.append(result)
            status = "OK" if result["status"] == "ok" else "ERR"
            latency = result.get("latency_sec", 0)
            length = result.get("response_length", 0)
            print(f"  [{done}/{total}] {status} {latency:.1f}s {length} chars — {prompt[:60]}...")

    # Save raw results
    output_path = Path("data/benchmark_v2_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] != "ok"]

    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE: {len(ok)}/{total} passed, {len(err)} errors")
    print(f"{'='*60}")

    if ok:
        avg_latency = sum(r["latency_sec"] for r in ok) / len(ok)
        avg_length = sum(r["response_length"] for r in ok) / len(ok)
        avg_tokens = sum(r["completion_tokens"] for r in ok) / len(ok)
        code_pct = sum(1 for r in ok if r["has_code"]) / len(ok) * 100

        print(f"Avg latency:  {avg_latency:.2f}s")
        print(f"Avg response: {avg_length:.0f} chars / {avg_tokens:.0f} tokens")
        print(f"Code blocks:  {code_pct:.0f}%")

        # Per-category stats
        print(f"\nPer-category breakdown:")
        for cat in TEST_PROMPTS:
            cat_results = [r for r in ok if r["category"] == cat]
            if cat_results:
                cat_lat = sum(r["latency_sec"] for r in cat_results) / len(cat_results)
                cat_len = sum(r["response_length"] for r in cat_results) / len(cat_results)
                cat_code = sum(1 for r in cat_results if r["has_code"]) / len(cat_results) * 100
                print(f"  {cat:30s} lat={cat_lat:.1f}s  len={cat_len:.0f}  code={cat_code:.0f}%")

    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
