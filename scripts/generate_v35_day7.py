"""V3.5 Day 7: Generate verified math, clarification elimination, integration
architecture, and vendor-specific EHR training data.

Verified Math: 3,000 examples
Clarification Elimination: 2,000 examples
Integration Architecture: 2,000 examples
Vendor-Specific EHR: 3,000 examples
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
MODEL_70B = "llama3:70b"
TIMEOUT = 600
MAX_RETRIES = 3
NUM_WORKERS = 6

_file_lock = threading.Lock()


def _call(prompt, temperature=0.7, model=None):
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
# Verified Math Prompts (3,000 examples)
# ---------------------------------------------------------------------------

HEALTHCARE_MATH_PROMPTS = [
    "Calculate the correct IV drip rate in mL/hr for a patient weighing {weight}kg who needs {drug} at {dose} mcg/kg/min. The concentration is {conc} mg in {vol} mL. Show all steps.",
    "A hospital processes {msgs} HL7 messages per hour. Each message takes {ms}ms to parse and {ms2}ms to transform. With {workers} parallel workers, what is the maximum throughput in messages/second? Show the calculation.",
    "Calculate the dosage for a pediatric patient: weight {weight}kg, prescribed {drug} at {dose} mg/kg/day divided into {freq} doses. Available concentration is {conc} mg/{vol} mL. Show every step.",
    "A healthcare system needs to handle {users} concurrent users. Each session generates {queries} DB queries/minute averaging {ms}ms each. How many database connections are needed? Calculate the capacity.",
    "Calculate the storage capacity needed for {days} days of HL7 message archival. Average message size is {size} bytes, volume is {vol} messages/day. Include {overhead}% overhead for indexing. Show all math.",
    "A clinical trial has {patients} patients across {sites} sites. Each patient generates {obs} observations per visit with {visits} visits over {months} months. Calculate total FHIR Observation resources needed.",
    "Calculate the BMI for a patient who is {height_cm}cm tall and weighs {weight}kg. Classify the result. Then calculate ideal body weight using the Devine formula. Show all steps.",
    "A pharmacy dispenses {rx} prescriptions/day. Each takes {min} minutes to verify. With {staff} pharmacists working {hours} hour shifts, calculate utilization rate and wait time. Show calculations.",
    "Calculate the creatinine clearance using Cockcroft-Gault for a {age}-year-old {sex} patient weighing {weight}kg with serum creatinine of {cr} mg/dL. Show all steps and units.",
    "A health system migrates {records} million patient records. Each record averages {size}KB as FHIR JSON. With {bandwidth}Mbps upload speed and {overhead}% protocol overhead, how long will the migration take?",
    "Calculate the anion gap for a patient with Na+ {na} mEq/L, Cl- {cl} mEq/L, HCO3- {hco3} mEq/L. Interpret the result and suggest possible causes. Show the formula and calculation.",
    "A hospital ER has {beds} beds with average stay of {hours} hours. Arrival rate is {rate} patients/hour. Using Little's Law, calculate average occupancy and probability of wait. Show all math.",
]

GENERAL_MATH_PROMPTS = [
    "Solve: {a}x^2 + {b}x + {c} = 0. Show the discriminant calculation and both roots. Verify by substitution.",
    "Calculate the mean, median, standard deviation, and interquartile range of: {numbers}. Show each step.",
    "A dataset has values {numbers}. Perform a z-score normalization. Show the z-score for each value.",
    "Evaluate: ({a} * {b} + {c}) / {d} - {e}^{f}. Show order of operations step by step.",
    "Calculate compound interest: principal ${p}, rate {r}%, compounded {n} times per year, for {t} years. Show A = P(1 + r/n)^(nt) step by step.",
    "Find the probability of getting exactly {k} successes in {n} trials with probability {p} per trial (binomial). Show the formula and calculation.",
    "Convert {value} from base {base1} to base {base2}. Show the conversion process step by step.",
    "Calculate the derivative of f(x) = {a}x^{n} + {b}sin(x) + {c}e^x. Show each term's derivative and the final result.",
    "A server processes requests with response times (ms): {numbers}. Calculate P50, P95, P99 percentiles. Show the method.",
    "Solve the system of equations: {a1}x + {b1}y = {c1} and {a2}x + {b2}y = {c2}. Use substitution method. Verify the solution.",
]

CHAIN_OF_THOUGHT_MATH_PROMPTS = [
    "A hospital needs to scale its FHIR server. Current load: {load} requests/sec at {cpu}% CPU. They expect {growth}x growth in {months} months. Current server has {cores} cores. How many additional servers (same spec) are needed? Think through this step by step.",
    "An HL7 integration processes messages from {sources} source systems. Source A sends {a_rate}/min, Source B sends {b_rate}/min, Source C sends {c_rate}/min. Each message expands {factor}x when converted to FHIR. If the FHIR server handles {capacity} writes/sec, is the system capacity sufficient? Think step by step.",
    "A pharmacy system checks drug interactions. There are {drugs} drugs in the formulary. How many unique pairwise interactions need to be checked? If each check takes {ms}ms and we use {threads} threads, how long to check all interactions for a patient on {patient_drugs} medications? Think step by step.",
    "A data warehouse ingests patient data from {hospitals} hospitals. Hospital sizes: {sizes} beds. Average data per bed per day: {data_mb} MB. Storage costs ${cost}/GB/month. Calculate monthly storage cost after {months} months. Think through each step.",
    "An HIE serves {population} million patients. {percent}% have records at multiple facilities averaging {facilities} facilities each. Each cross-reference query costs {ms}ms. If {queries} queries/day reference multi-facility patients, what's the daily compute time? Think step by step.",
    "A clinical decision support system evaluates {rules} rules per patient encounter. Each rule has {conditions} conditions averaging {ms}ms to evaluate. If {encounters} encounters/day occur, and rules must complete in under {target}ms, how many parallel rule engines are needed? Think step by step.",
    "A health system backs up {tb}TB of data. Full backup takes {full_hrs}hrs. Incremental backup covers {pct}% of data and is {factor}x faster per GB. If they do full weekly and incremental daily, what's the total weekly backup time? Think step by step.",
    "A FHIR subscription service monitors {resources} resource types. Each has {subs} active subscriptions. When a resource changes, each subscription notification takes {ms}ms. If {changes} changes/hour occur across all types equally, what's the notification processing time per hour? Think step by step.",
]


def _fill_healthcare_math(template):
    """Fill healthcare math template with random realistic values."""
    drugs = ["dopamine", "norepinephrine", "dobutamine", "epinephrine", "phenylephrine"]
    return template.format(
        weight=random.randint(40, 120),
        drug=random.choice(drugs),
        dose=round(random.uniform(0.5, 20), 1),
        conc=random.choice([200, 400, 800, 1600]),
        vol=random.choice([250, 500, 1000]),
        ms=random.randint(5, 50),
        ms2=random.randint(10, 100),
        workers=random.choice([4, 8, 16, 32]),
        msgs=random.randint(1000, 50000),
        freq=random.choice([2, 3, 4, 6]),
        users=random.randint(100, 10000),
        queries=random.randint(5, 50),
        days=random.choice([30, 90, 180, 365]),
        size=random.randint(500, 5000),
        overhead=random.choice([10, 15, 20, 25]),
        patients=random.randint(50, 5000),
        sites=random.randint(5, 50),
        obs=random.randint(5, 30),
        visits=random.randint(4, 24),
        months=random.randint(6, 36),
        height_cm=random.randint(150, 200),
        rx=random.randint(100, 2000),
        min=round(random.uniform(2, 10), 1),
        staff=random.randint(2, 10),
        hours=random.choice([8, 10, 12]),
        age=random.randint(18, 90),
        sex=random.choice(["male", "female"]),
        cr=round(random.uniform(0.6, 4.0), 1),
        na=random.randint(130, 150),
        cl=random.randint(90, 115),
        hco3=random.randint(15, 30),
        beds=random.randint(20, 500),
        rate=round(random.uniform(2, 15), 1),
        records=random.randint(1, 50),
        bandwidth=random.choice([100, 500, 1000]),
    )


def _fill_general_math(template):
    """Fill general math template with random values."""
    numbers_list = sorted([random.randint(1, 100) for _ in range(random.randint(6, 12))])
    return template.format(
        a=random.randint(1, 10),
        b=random.randint(-10, 10),
        c=random.randint(-20, 20),
        d=random.randint(1, 10),
        e=random.randint(2, 5),
        f=random.randint(2, 4),
        n=random.randint(2, 5),
        numbers=", ".join(str(x) for x in numbers_list),
        value=random.randint(10, 1000),
        base1=random.choice([2, 8, 10, 16]),
        base2=random.choice([2, 8, 10, 16]),
        p=random.randint(1000, 50000),
        r=round(random.uniform(1, 12), 1),
        t=random.randint(1, 30),
        k=random.randint(1, 8),
        a1=random.randint(1, 5),
        b1=random.randint(1, 5),
        c1=random.randint(1, 20),
        a2=random.randint(1, 5),
        b2=random.randint(1, 5),
        c2=random.randint(1, 20),
    )


def _fill_cot_math(template):
    """Fill chain-of-thought math template with random values."""
    sizes = ", ".join(str(random.randint(50, 800)) for _ in range(random.randint(3, 6)))
    return template.format(
        load=random.randint(50, 500),
        cpu=random.randint(40, 85),
        growth=round(random.uniform(1.5, 5), 1),
        months=random.randint(3, 24),
        cores=random.choice([4, 8, 16, 32]),
        sources=random.randint(3, 10),
        a_rate=random.randint(10, 200),
        b_rate=random.randint(10, 200),
        c_rate=random.randint(10, 200),
        factor=round(random.uniform(2, 8), 1),
        capacity=random.randint(50, 500),
        drugs=random.randint(500, 5000),
        ms=round(random.uniform(0.1, 5), 1),
        threads=random.choice([4, 8, 16]),
        patient_drugs=random.randint(3, 15),
        hospitals=random.randint(3, 20),
        sizes=sizes,
        data_mb=round(random.uniform(0.5, 5), 1),
        cost=round(random.uniform(0.01, 0.05), 3),
        population=round(random.uniform(0.5, 10), 1),
        percent=random.randint(10, 40),
        facilities=round(random.uniform(2, 5), 1),
        queries=random.randint(1000, 100000),
        rules=random.randint(50, 500),
        conditions=random.randint(3, 10),
        encounters=random.randint(100, 5000),
        target=random.randint(100, 1000),
        tb=random.randint(5, 100),
        full_hrs=round(random.uniform(2, 24), 1),
        pct=random.randint(5, 20),
        resources=random.randint(5, 20),
        subs=random.randint(10, 200),
        changes=random.randint(100, 10000),
    )


# ---------------------------------------------------------------------------
# Clarification Elimination Prompts (2,000 examples)
# ---------------------------------------------------------------------------

JUST_DO_IT_PROMPTS = [
    "Build me a patient intake form backend.",
    "Create a healthcare data pipeline.",
    "Make an HL7 message router.",
    "Set up a FHIR server with custom resources.",
    "Write a healthcare chatbot.",
    "Build an appointment scheduling API.",
    "Create a medication management system.",
    "Make a clinical notes search engine.",
    "Build a lab results dashboard backend.",
    "Write a patient matching service.",
    "Create an insurance eligibility checker.",
    "Build a referral management API.",
    "Make a healthcare notification service.",
    "Set up a clinical data warehouse ETL.",
    "Write a patient portal backend.",
    "Create a drug interaction checker.",
    "Build a healthcare audit logging system.",
    "Make a DICOM image metadata extractor.",
    "Write a clinical trial enrollment system.",
    "Build a telehealth session manager.",
    "Create a pharmacy inventory tracker.",
    "Make a healthcare compliance monitor.",
    "Build a patient consent management API.",
    "Write a healthcare API gateway.",
    "Create a bed management system.",
]

JUST_DO_IT_INSTRUCTION = (
    "Do NOT ask any clarifying questions. Make reasonable assumptions and build a complete, "
    "working implementation. State your assumptions at the top as comments, then provide the full code."
)

COMPLEX_FHIR_HL7_PROMPTS = [
    "Convert an HL7 v2 ADT^A01 admission message into a FHIR R4 Bundle containing Patient, Encounter, Practitioner, Organization, and Coverage resources. Map all PID, PV1, IN1 segments. Include complete code.",
    "Build a bidirectional sync between an HL7 v2 ADT feed and a FHIR R4 server. Handle A01 (admit), A02 (transfer), A03 (discharge), A08 (update), A11 (cancel admit). Full working code.",
    "Create an HL7 v2 to FHIR R4 lab result translator that handles ORU^R01 messages with multiple OBR/OBX groups, nested components, repeating fields, and coded values. Map to DiagnosticReport and Observation.",
    "Build a FHIR Subscription processor that watches for new Encounter resources, fetches related Patient and Practitioner data, generates an HL7 ADT^A01, and sends it via MLLP. Complete implementation.",
    "Create a healthcare data lake ingestion pipeline that accepts both HL7 v2 (via MLLP) and FHIR R4 (via REST), normalizes to a common schema, and stores in parquet format. Full code.",
    "Build a clinical document exchange system that converts FHIR DocumentReference resources to CDA documents and vice versa. Handle narratives, coded entries, and attachments. Working code.",
    "Create a patient merge handler that processes HL7 ADT^A40 merge messages and updates all related FHIR resources (Patient, Encounter, Observation, etc.) maintaining referential integrity. Full implementation.",
    "Build a medication reconciliation pipeline: take HL7 RDE^O11 pharmacy messages and FHIR MedicationRequest resources, compare them, identify discrepancies, and generate alerts. Complete code.",
    "Create a multi-facility patient index that receives ADT messages from multiple HL7 feeds, links patients using probabilistic matching, and exposes a FHIR Patient/$match endpoint. Working implementation.",
    "Build an HL7 FHIR facade that presents an HL7 v2 interface (MLLP) to legacy systems while the backend is entirely FHIR R4. Translate QBP^Q22 queries to FHIR searches and return RSP^K22 responses.",
    "Create a real-time clinical dashboard backend that subscribes to HL7 ADT, ORU, and SIU messages, maintains current state in FHIR resources, and exposes WebSocket updates. Full working code.",
    "Build a referral workflow engine: accept HL7 REF^I12 referral messages, create FHIR ServiceRequest/Task resources, track status changes, and send HL7 acknowledgments. Complete implementation.",
]


# ---------------------------------------------------------------------------
# Integration Architecture Prompts (2,000 examples)
# ---------------------------------------------------------------------------

SYSTEM_DESIGN_PROMPTS = [
    "Design a healthcare integration platform architecture that handles 10,000 HL7 messages/minute from 50 source systems. Include message routing, transformation, error handling, monitoring, and disaster recovery. Provide detailed architecture with component diagrams described in text.",
    "Design a FHIR-first healthcare data platform for a 500-bed hospital. Include data ingestion from EHR, labs, pharmacy, and imaging. Cover API gateway, auth, rate limiting, audit, and scalability. Detailed architecture.",
    "Design a clinical data exchange architecture for a regional HIE connecting 20 hospitals. Include consent management, patient matching, document exchange, and real-time notifications. Provide full system design.",
    "Design a healthcare analytics platform that ingests real-time HL7 feeds and FHIR bulk exports. Include data lake, ETL pipeline, data warehouse, and visualization layer. Address HIPAA requirements.",
    "Design a healthcare microservices architecture for a patient engagement platform. Include appointment scheduling, messaging, care plans, and integrations with Epic and Cerner. Cover service mesh, auth, and observability.",
    "Design a disaster recovery architecture for a healthcare integration engine. Include RPO/RTO requirements, data replication, failover mechanisms, and testing strategies. Address regulatory requirements.",
    "Design a real-time clinical decision support architecture that processes ADT events, lab results, and medication orders. Include rule engine, alert routing, physician notification, and feedback loops.",
    "Design a healthcare IoT data platform that ingests vitals from bedside monitors (HL7/IEEE 11073), wearables (FHIR), and medical devices. Include edge processing, streaming, alerting, and long-term storage.",
]

MULTI_VENDOR_PROMPTS = [
    "Design and implement an integration architecture that connects Epic, Cerner, and Athena Health to a central analytics platform. Handle different FHIR versions, auth mechanisms, and data models. Provide code for the adapter layer.",
    "Build a multi-vendor EHR integration hub that normalizes patient data from Epic (FHIR R4), Cerner (FHIR R4), MEDITECH (HL7 v2), and Allscripts (proprietary API) into a unified FHIR format. Include adapter pattern code.",
    "Create a vendor-agnostic appointment scheduling API that works across Epic, Cerner, and Athena. Handle different booking workflows, availability models, and notification patterns. Full working code.",
    "Design a multi-vendor clinical document exchange: accept CCDAs from MEDITECH, FHIR DocumentReferences from Epic, and HL7 MDM messages from Cerner. Normalize and make searchable. Architecture and code.",
    "Build a multi-EHR medication reconciliation system that pulls medication lists from Epic, Cerner, and Allscripts using their respective APIs, reconciles duplicates, and presents a unified view. Include code.",
    "Create a unified patient search across Epic, Cerner, and Athena. Handle different identifier systems, name formats, and search capabilities. Implement fan-out search with result merging. Provide full code.",
    "Design a multi-vendor lab result aggregation system. Accept HL7 ORU from legacy systems and FHIR Observations from modern EHRs. Normalize LOINC coding, units, and reference ranges. Architecture and code.",
    "Build a multi-vendor clinical alert routing system that receives alerts from Epic BestPractice, Cerner PowerChart, and custom CDS. Normalize alert format, deduplicate, and route to appropriate clinicians.",
]

HL7_FHIR_MIGRATION_PROMPTS = [
    "Create a detailed migration plan and implementation for moving a hospital from HL7 v2 ADT interfaces to FHIR R4. Include parallel running strategy, data mapping, testing approach, and rollback plan. Provide migration code.",
    "Design and implement a phased migration from HL7 v2 order/result interfaces (ORM/ORU) to FHIR-based workflow. Include a dual-write bridge that maintains both interfaces during migration. Code included.",
    "Build an HL7 v2 to FHIR R4 migration toolkit: message converter, validation suite, comparison reports, and rollback capability. Handle ADT, ORM, ORU, SIU, and MDM message types. Full implementation.",
    "Create a migration strategy for moving 10 years of HL7 message archives to FHIR resources. Include batch conversion, validation, reference resolution, and provenance tracking. Provide code and architecture.",
    "Design a FHIR facade over existing HL7 v2 interfaces that allows incremental migration. New consumers use FHIR while legacy systems continue with HL7. Include routing logic and translation code.",
    "Build a testing framework for HL7 to FHIR migration. Generate test messages, compare round-trip conversions, report mapping gaps, and validate FHIR profile conformance. Full code.",
]

HIGH_AVAILABILITY_PROMPTS = [
    "Design a highly available healthcare integration engine with zero-downtime deployments. Include active-active clustering, message persistence, in-flight message handling during failover, and health monitoring. Provide configuration and code.",
    "Build a healthcare API gateway with HA: active-passive failover, connection draining, circuit breakers, rate limiting per client, and request replay for failed downstream calls. Include full code.",
    "Design a highly available MLLP listener cluster for HL7 messages. Handle TCP connection failover, message deduplication, ordered delivery guarantees, and split-brain prevention. Architecture and code.",
    "Create a resilient healthcare event streaming architecture using Kafka. Include multi-datacenter replication, consumer group management, exactly-once processing, and automated recovery. Provide code.",
    "Design HA for a FHIR server serving critical clinical data. Include database replication, read replicas, cache invalidation, search index sync, and automated failover. Architecture and configuration.",
    "Build a healthcare message queue system with guaranteed delivery. Handle broker failover, message persistence, consumer acknowledgment, dead-letter handling, and monitoring. Full implementation.",
]

COMPLIANCE_PROMPTS = [
    "Design a HIPAA-compliant healthcare integration architecture. Include encryption at rest and in transit, access controls, audit logging, BAA management, breach detection, and incident response automation. Provide code for the audit and encryption layers.",
    "Build a healthcare data governance framework: consent management, data retention policies, access audit, de-identification pipeline, and compliance reporting. Include FHIR Consent resource handling. Provide code.",
    "Design a compliance monitoring system for healthcare APIs. Track access patterns, detect anomalies, enforce minimum necessary access, generate compliance reports, and alert on violations. Architecture and code.",
    "Create a healthcare data de-identification pipeline following HIPAA Safe Harbor method. Process FHIR resources, HL7 messages, and clinical notes. Include 18 identifier types. Full implementation.",
    "Design an architecture for cross-border healthcare data exchange complying with HIPAA, GDPR, and local regulations. Include consent management, data localization, and audit trails. Architecture and code.",
    "Build a role-based access control system for healthcare APIs with break-the-glass emergency access. Include audit logging, time-limited elevated access, and automated review workflows. Full code.",
]

PERFORMANCE_PROMPTS = [
    "Design a high-performance HL7 message processing pipeline handling 50,000 messages/minute. Include connection pooling, batch processing, async I/O, and backpressure management. Provide benchmarking code.",
    "Build a FHIR search optimization layer: query planning, index management, result caching, pagination strategies, and _include/_revinclude optimization. Include performance testing code.",
    "Design a healthcare data caching architecture. Handle cache invalidation for clinical data (short TTL), reference data (long TTL), and real-time data (no cache). Include consistency guarantees and code.",
    "Create a load testing framework for healthcare integration engines. Generate realistic HL7 message patterns, simulate multiple source systems, measure throughput and latency, and identify bottlenecks. Full code.",
]


# ---------------------------------------------------------------------------
# Vendor-Specific EHR Prompts (3,000 examples)
# ---------------------------------------------------------------------------

EPIC_FHIR_PROMPTS = [
    "Write {lang} code to authenticate with Epic's FHIR R4 API using SMART Backend Services (JWT client credentials). Include key generation, JWT signing, token exchange, and token refresh. Handle all error cases.",
    "Write {lang} code to search for patients in Epic using the FHIR R4 Patient resource. Support search by MRN, name, DOB, and SSN. Handle pagination with Bundle.link. Include Epic-specific search parameters.",
    "Write {lang} code to read a patient's medication list from Epic FHIR R4. Use MedicationRequest search with patient context. Handle active, completed, and cancelled medications. Parse coded data (RxNorm).",
    "Write {lang} code to create a FHIR Appointment in Epic. Include participant lookup, slot availability search, booking confirmation, and status updates. Handle Epic's scheduling constraints.",
    "Write {lang} code to retrieve lab results from Epic FHIR R4 using Observation search. Filter by LOINC code, date range, and status. Handle component observations and reference ranges.",
    "Write {lang} code to submit a clinical note to Epic via FHIR DocumentReference. Include binary attachment upload, proper coding (LOINC document types), and status management.",
    "Write {lang} code to query Epic's FHIR R4 AllergyIntolerance resource. Handle clinical status, verification status, reaction details, and substance coding (RxNorm, SNOMED).",
    "Write {lang} code to perform a FHIR $everything operation on an Epic Patient. Handle large response bundles, pagination, and selective resource type inclusion.",
    "Write {lang} code to manage care teams in Epic via FHIR CareTeam resource. Include member management, role assignment, period handling, and status transitions.",
    "Write {lang} code to read and write Problem List entries in Epic via FHIR Condition resource. Handle clinical status, verification, onset, abatement, and SNOMED CT coding.",
    "Write {lang} code to integrate with Epic's CDS Hooks. Implement patient-view and order-select hooks. Return suggestion cards with SMART app launch links. Handle prefetch data.",
    "Write {lang} code to read vital signs from Epic FHIR R4. Use Observation search with vital-signs category. Handle blood pressure (component observations), BMI, and temperature.",
]

EPIC_MYCHART_PROMPTS = [
    "Write {lang} code for a MyChart-integrated patient-facing app using SMART on FHIR. Implement the EHR launch flow, standalone launch, and token refresh. Handle patient context and scopes.",
    "Write {lang} code for a MyChart patient portal integration that lets patients view their upcoming appointments, lab results, and medications. Use FHIR R4 with patient-level scopes.",
    "Write {lang} code to implement MyChart messaging integration. Use FHIR Communication resource to send and receive messages between patients and care teams. Handle attachments and threading.",
    "Write {lang} code for a MyChart questionnaire integration. Use FHIR Questionnaire and QuestionnaireResponse resources for patient intake forms. Handle validation and submission.",
    "Write {lang} code for a MyChart-connected remote monitoring app. Submit patient-generated vital signs (blood pressure, glucose, weight) as FHIR Observations with appropriate categories and device references.",
]

CERNER_PROMPTS = [
    "Write {lang} code to authenticate with Cerner Ignite (Oracle Health) FHIR R4 API. Implement both authorization code flow and system account access. Handle tenant-specific endpoints.",
    "Write {lang} code to search and read Patient resources from Cerner FHIR R4. Handle Cerner's custom extensions, identifier systems, and search parameter differences from Epic.",
    "Write {lang} code to manage orders in Cerner via FHIR ServiceRequest. Create, update, and cancel lab and radiology orders. Handle Cerner's order workflow and required extensions.",
    "Write {lang} code to read clinical documents from Cerner via FHIR DocumentReference. Handle Cerner's document type coding, binary content retrieval, and access controls.",
    "Write {lang} code to manage encounters in Cerner FHIR R4. Create, update, and discharge encounters. Handle Cerner's encounter types, location management, and participant tracking.",
    "Write {lang} code to submit lab results to Cerner via FHIR DiagnosticReport and Observation. Handle Cerner's required coding systems, value types, and reference ranges.",
    "Write {lang} code to integrate with Cerner's real-time event notifications (FHIR Subscription). Subscribe to patient admissions, discharges, and lab result events. Handle webhook delivery.",
    "Write {lang} code to query Cerner's FHIR Provenance resources to build an audit trail of clinical data changes. Track who changed what and when.",
    "Write {lang} code to manage immunization records in Cerner via FHIR Immunization resource. Handle CVX coding, dose sequencing, and contraindication checking.",
]

ATHENA_PROMPTS = [
    "Write {lang} code to authenticate with Athena Health's API (athenaNET). Implement OAuth2 flow, handle practice-level authentication, and manage API keys. Include rate limit handling.",
    "Write {lang} code to search and manage patients in Athena Health. Handle demographics, insurance information, custom fields, and Athena's proprietary patient matching.",
    "Write {lang} code to manage appointments in Athena Health. Create, update, cancel appointments. Handle provider schedules, appointment types, and reminder preferences.",
    "Write {lang} code to submit and retrieve clinical documents via Athena Health API. Handle document categories, upload attachments, and manage document statuses.",
    "Write {lang} code to manage orders and results in Athena Health. Create lab orders, receive results, and handle order-result matching. Include HL7 result interface.",
    "Write {lang} code to handle billing and claims data via Athena Health API. Manage charge entry, claim submission, ERA processing, and payment posting.",
    "Write {lang} code to integrate with Athena Health's patient portal. Handle patient self-scheduling, messaging, and document sharing through the API.",
    "Write {lang} code to manage clinical encounters in Athena Health. Handle encounter creation, HPI documentation, assessment/plan, and encounter closing workflow.",
]

MEDITECH_PROMPTS = [
    "Write {lang} code to integrate with MEDITECH Expanse via its FHIR R4 API. Handle authentication, patient search, and clinical data retrieval. Address MEDITECH-specific FHIR profile differences.",
    "Write {lang} code to receive HL7 v2 messages from MEDITECH and transform to FHIR R4. Handle MEDITECH's custom Z-segments, encoding quirks, and field mapping differences.",
    "Write {lang} code to query MEDITECH's clinical data repository. Handle orders, results, medications, and allergies. Address MEDITECH's specific data model and API patterns.",
    "Write {lang} code for a MEDITECH Expanse integration that syncs patient demographics bidirectionally. Handle MEDITECH's MRN format, name conventions, and address structures.",
    "Write {lang} code to process MEDITECH discharge summaries. Parse the document structure, extract coded data, and create FHIR DocumentReference and Composition resources.",
    "Write {lang} code to integrate with MEDITECH's pharmacy module. Handle medication orders, dispensing events, and medication administration records via HL7 and FHIR interfaces.",
]

ALLSCRIPTS_PROMPTS = [
    "Write {lang} code to integrate with Allscripts/Veradigm Unity API. Handle authentication (GetSecurityToken), patient search, and clinical data retrieval. Include error handling for Unity-specific error codes.",
    "Write {lang} code to manage clinical encounters in Allscripts Professional. Handle encounter creation, documentation, diagnosis coding, and encounter finalization via the Unity API.",
    "Write {lang} code to retrieve and submit medication data via Allscripts/Veradigm. Handle e-prescribing workflows, medication reconciliation, and formulary checking.",
    "Write {lang} code to integrate with Veradigm's clinical data exchange platform. Handle CCDA document exchange, patient matching, and data normalization.",
]

MULTI_VENDOR_EHR_PROMPTS = [
    "Write {lang} code for a unified EHR API client that abstracts differences between Epic, Cerner, Athena, and MEDITECH. Use adapter pattern. Handle auth, patient search, and clinical data retrieval for each vendor.",
    "Write {lang} code to synchronize patient demographics across Epic, Cerner, and Athena. Handle vendor-specific identifier systems, name formats, and update workflows. Include conflict resolution.",
    "Write {lang} code for a multi-vendor appointment aggregator. Query Epic, Cerner, and Athena for patient appointments. Normalize data models, handle timezone differences, and merge results.",
    "Write {lang} code to build a unified medication view across Epic (FHIR), Cerner (FHIR), and MEDITECH (HL7). Reconcile duplicates using RxNorm coding. Handle different status models.",
    "Write {lang} code for a multi-vendor clinical document aggregator. Collect documents from Epic, Cerner, Allscripts, and MEDITECH. Normalize metadata, deduplicate, and present a unified timeline.",
    "Write {lang} code to implement vendor-neutral patient matching across Epic, Cerner, Athena, and MEDITECH. Use probabilistic matching with configurable thresholds per vendor.",
    "Write {lang} code for a multi-vendor lab result aggregator. Normalize LOINC codes, units, and reference ranges from Epic, Cerner, and MEDITECH. Handle vendor-specific result formats.",
    "Write {lang} code for a unified referral management system that works with Epic, Cerner, and Athena referral workflows. Handle different state machines and notification patterns.",
    "Write {lang} code for a multi-vendor immunization registry integration. Submit and query immunization records across Epic, Cerner, and MEDITECH using IIS HL7 standards.",
    "Write {lang} code for cross-vendor care plan aggregation. Pull care plans from Epic, Cerner, and Athena, merge overlapping goals, and present a unified patient care plan.",
]


# ---------------------------------------------------------------------------
# Generation Functions
# ---------------------------------------------------------------------------

def generate_verified_math(output_path, target_count=3000):
    """Generate 3,000 verified math examples."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_lines(output_path)
    if existing >= target_count:
        logger.info("Verified math: already have %d/%d, skipping", existing, target_count)
        return existing

    remaining = target_count - existing
    logger.info("Verified math: generating %d examples (%d existing)", remaining, existing)

    all_prompts = []

    # 1,000 healthcare math
    for _ in range(1000):
        template = random.choice(HEALTHCARE_MATH_PROMPTS)
        try:
            prompt = _fill_healthcare_math(template)
        except (KeyError, IndexError):
            prompt = template
        all_prompts.append(("healthcare_math", prompt, "healthcare"))

    # 1,000 general math
    for _ in range(1000):
        template = random.choice(GENERAL_MATH_PROMPTS)
        try:
            prompt = _fill_general_math(template)
        except (KeyError, IndexError):
            prompt = template
        all_prompts.append(("general_math", prompt, "math"))

    # 1,000 chain-of-thought
    for _ in range(1000):
        template = random.choice(CHAIN_OF_THOUGHT_MATH_PROMPTS)
        try:
            prompt = _fill_cot_math(template)
        except (KeyError, IndexError):
            prompt = template
        all_prompts.append(("chain_of_thought", prompt, "math"))

    random.shuffle(all_prompts)
    all_prompts = all_prompts[:remaining]

    completed = [0]
    failed = [0]

    def process(item):
        category, prompt, domain = item
        output = _call(prompt, temperature=0.7, model=MODEL_70B)
        if output and len(output) > 100:
            example = {
                "instruction": prompt,
                "output": output,
                "domain": domain,
                "source_standard": category,
                "version": "v3.5-verified-math",
            }
            _write(output_path, example)
            completed[0] += 1
            if completed[0] % 50 == 0:
                logger.info("Verified math: %d/%d done (failed: %d)", completed[0], remaining, failed[0])
        else:
            failed[0] += 1

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        list(pool.map(process, all_prompts))

    logger.info("Verified math complete: %d succeeded, %d failed", completed[0], failed[0])
    return existing + completed[0]


def generate_clarification_elimination(output_path, target_count=2000):
    """Generate 2,000 clarification elimination examples."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_lines(output_path)
    if existing >= target_count:
        logger.info("Clarification elimination: already have %d/%d, skipping", existing, target_count)
        return existing

    remaining = target_count - existing
    logger.info("Clarification elimination: generating %d examples (%d existing)", remaining, existing)

    all_prompts = []

    # 1,000 "just do it" responses
    for _ in range(1000):
        base_prompt = random.choice(JUST_DO_IT_PROMPTS)
        prompt = base_prompt + " " + JUST_DO_IT_INSTRUCTION
        all_prompts.append(("just_do_it", prompt, "healthcare"))

    # 1,000 complex FHIR/HL7 combination tasks
    for _ in range(1000):
        lang = random.choice(LANGS)
        template = random.choice(COMPLEX_FHIR_HL7_PROMPTS)
        prompt = template + " " + JUST_DO_IT_INSTRUCTION
        all_prompts.append(("complex_fhir_hl7", prompt, "hl7_fhir"))

    random.shuffle(all_prompts)
    all_prompts = all_prompts[:remaining]

    completed = [0]
    failed = [0]

    def process(item):
        category, prompt, domain = item
        output = _call(prompt, temperature=0.3, model=MODEL_70B)
        if output and len(output) > 200:
            example = {
                "instruction": prompt,
                "output": output,
                "domain": domain,
                "source_standard": category,
                "version": "v3.5-clarification-elimination",
            }
            _write(output_path, example)
            completed[0] += 1
            if completed[0] % 50 == 0:
                logger.info("Clarification elim: %d/%d done (failed: %d)", completed[0], remaining, failed[0])
        else:
            failed[0] += 1

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        list(pool.map(process, all_prompts))

    logger.info("Clarification elimination complete: %d succeeded, %d failed", completed[0], failed[0])
    return existing + completed[0]


def generate_integration_architecture(output_path, target_count=2000):
    """Generate 2,000 integration architecture examples."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_lines(output_path)
    if existing >= target_count:
        logger.info("Integration architecture: already have %d/%d, skipping", existing, target_count)
        return existing

    remaining = target_count - existing
    logger.info("Integration architecture: generating %d examples (%d existing)", remaining, existing)

    all_prompts = []

    # 500 system design
    for _ in range(500):
        prompt = random.choice(SYSTEM_DESIGN_PROMPTS)
        all_prompts.append(("system_design", prompt, "architecture"))

    # 400 multi-vendor EHR integration
    for _ in range(400):
        prompt = random.choice(MULTI_VENDOR_PROMPTS)
        all_prompts.append(("multi_vendor", prompt, "architecture"))

    # 300 HL7 to FHIR migration
    for _ in range(300):
        prompt = random.choice(HL7_FHIR_MIGRATION_PROMPTS)
        all_prompts.append(("hl7_fhir_migration", prompt, "architecture"))

    # 300 high availability
    for _ in range(300):
        prompt = random.choice(HIGH_AVAILABILITY_PROMPTS)
        all_prompts.append(("high_availability", prompt, "architecture"))

    # 300 compliance architecture
    for _ in range(300):
        prompt = random.choice(COMPLIANCE_PROMPTS)
        all_prompts.append(("compliance", prompt, "architecture"))

    # 200 performance patterns
    for _ in range(200):
        prompt = random.choice(PERFORMANCE_PROMPTS)
        all_prompts.append(("performance", prompt, "architecture"))

    random.shuffle(all_prompts)
    all_prompts = all_prompts[:remaining]

    completed = [0]
    failed = [0]

    def process(item):
        category, prompt, domain = item
        output = _call(prompt, temperature=0.7, model=MODEL_70B)
        if output and len(output) > 200:
            example = {
                "instruction": prompt,
                "output": output,
                "domain": domain,
                "source_standard": category,
                "version": "v3.5-integration-architecture",
            }
            _write(output_path, example)
            completed[0] += 1
            if completed[0] % 50 == 0:
                logger.info("Integration arch: %d/%d done (failed: %d)", completed[0], remaining, failed[0])
        else:
            failed[0] += 1

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        list(pool.map(process, all_prompts))

    logger.info("Integration architecture complete: %d succeeded, %d failed", completed[0], failed[0])
    return existing + completed[0]


def generate_vendor_ehr(output_path, target_count=3000):
    """Generate 3,000 vendor-specific EHR examples."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_lines(output_path)
    if existing >= target_count:
        logger.info("Vendor EHR: already have %d/%d, skipping", existing, target_count)
        return existing

    remaining = target_count - existing
    logger.info("Vendor EHR: generating %d examples (%d existing)", remaining, existing)

    all_prompts = []

    # 800 Epic FHIR API (non-MyChart)
    for _ in range(800):
        lang = random.choice(LANGS)
        template = random.choice(EPIC_FHIR_PROMPTS)
        all_prompts.append(("epic_fhir", template.format(lang=lang), "epic"))

    # 200 Epic MyChart
    for _ in range(200):
        lang = random.choice(LANGS)
        template = random.choice(EPIC_MYCHART_PROMPTS)
        all_prompts.append(("epic_mychart", template.format(lang=lang), "epic"))

    # 600 Cerner Ignite / Oracle Health
    for _ in range(600):
        lang = random.choice(LANGS)
        template = random.choice(CERNER_PROMPTS)
        all_prompts.append(("cerner_ignite", template.format(lang=lang), "cerner"))

    # 400 Athena Health
    for _ in range(400):
        lang = random.choice(LANGS)
        template = random.choice(ATHENA_PROMPTS)
        all_prompts.append(("athena_health", template.format(lang=lang), "athena"))

    # 300 MEDITECH Expanse
    for _ in range(300):
        lang = random.choice(LANGS)
        template = random.choice(MEDITECH_PROMPTS)
        all_prompts.append(("meditech_expanse", template.format(lang=lang), "meditech"))

    # 200 Allscripts/Veradigm
    for _ in range(200):
        lang = random.choice(LANGS)
        template = random.choice(ALLSCRIPTS_PROMPTS)
        all_prompts.append(("allscripts_veradigm", template.format(lang=lang), "allscripts"))

    # 500 multi-vendor scenarios
    for _ in range(500):
        lang = random.choice(LANGS)
        template = random.choice(MULTI_VENDOR_EHR_PROMPTS)
        all_prompts.append(("multi_vendor_ehr", template.format(lang=lang), "multi_vendor"))

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
                "version": "v3.5-vendor-ehr",
            }
            _write(output_path, example)
            completed[0] += 1
            if completed[0] % 50 == 0:
                logger.info("Vendor EHR: %d/%d done (failed: %d)", completed[0], remaining, failed[0])
        else:
            failed[0] += 1

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        list(pool.map(process, all_prompts))

    logger.info("Vendor EHR complete: %d succeeded, %d failed", completed[0], failed[0])
    return existing + completed[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3.5 Day 7: Math, clarification, architecture, vendor EHR")
    parser.add_argument("--math-output", default="data/raw/v35_verified_math.jsonl",
                        help="Output path for verified math examples")
    parser.add_argument("--clarification-output", default="data/raw/v35_clarification_elimination.jsonl",
                        help="Output path for clarification elimination examples")
    parser.add_argument("--architecture-output", default="data/raw/v35_integration_architecture.jsonl",
                        help="Output path for integration architecture examples")
    parser.add_argument("--vendor-output", default="data/raw/v35_vendor_ehr.jsonl",
                        help="Output path for vendor-specific EHR examples")
    parser.add_argument("--math-count", type=int, default=3000,
                        help="Target count for verified math examples")
    parser.add_argument("--clarification-count", type=int, default=2000,
                        help="Target count for clarification elimination examples")
    parser.add_argument("--architecture-count", type=int, default=2000,
                        help="Target count for integration architecture examples")
    parser.add_argument("--vendor-count", type=int, default=3000,
                        help="Target count for vendor-specific EHR examples")
    parser.add_argument("--skip-math", action="store_true", help="Skip verified math generation")
    parser.add_argument("--skip-clarification", action="store_true", help="Skip clarification elimination")
    parser.add_argument("--skip-architecture", action="store_true", help="Skip integration architecture")
    parser.add_argument("--skip-vendor", action="store_true", help="Skip vendor-specific EHR")
    args = parser.parse_args()

    logger.info("=== V3.5 Day 7: Math + Clarification + Architecture + Vendor EHR ===")

    if not args.skip_math:
        logger.info("--- Phase 1: Verified Math (3,000) ---")
        generate_verified_math(args.math_output, args.math_count)

    if not args.skip_clarification:
        logger.info("--- Phase 2: Clarification Elimination (2,000) ---")
        generate_clarification_elimination(args.clarification_output, args.clarification_count)

    if not args.skip_architecture:
        logger.info("--- Phase 3: Integration Architecture (2,000) ---")
        generate_integration_architecture(args.architecture_output, args.architecture_count)

    if not args.skip_vendor:
        logger.info("--- Phase 4: Vendor-Specific EHR (3,000) ---")
        generate_vendor_ehr(args.vendor_output, args.vendor_count)

    logger.info("=== Day 7 generation complete ===")
