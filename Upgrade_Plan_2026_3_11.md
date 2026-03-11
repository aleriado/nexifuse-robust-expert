# NexiFuse Robust Expert — Training Data & Teacher Model Strategy

**Date:** 2026-03-11
**Role:** Principal AI Architect
**Project:** NexiFuse Health — On-Prem Healthcare Interoperability LLM
**Student Model:** DeepSeek-R1-Distill-Llama-8B (LoRA, 4-bit NF4, DGX Spark)
**Constraint:** Fully local, fully free. No paid APIs. All data generation runs on-prem.

---

## Executive Summary

The current NexiFuse model is catastrophically overfit to healthcare integration code generation. It cannot answer "What is 2 + 4?" because 100% of training data was domain-specific. This document defines the complete dataset strategy, teacher model stack, synthetic data generation plan, and execution timeline to produce a Robust Expert that is both a competent general assistant and a best-in-class healthcare integration specialist.

The core thesis: **a well-curated 25k-example dataset with the right mixture will outperform a 100k-example dataset with the wrong mixture on an 8B model.** Data quality, mixture ratio, and validation rigor are the moat — not volume.

All teacher models run locally on DGX Spark via Ollama. Zero API costs. Zero data leaves the premises.

---

## 1. Recommended Total Dataset Size

### Why These Numbers

An 8B parameter model with LoRA rank 32 has limited capacity to absorb new knowledge. Too little data and the model underfits the domain. Too much data (especially low-quality synthetic data) and the model memorizes patterns rather than learning generalizable behavior. The sweet spots below are calibrated for LoRA fine-tuning specifically — full fine-tuning would require 3-5x more data.

### Size Targets

| Stage | Examples | Approx. Tokens | Training Time (DGX Spark) | Purpose |
|---|---|---|---|---|
| **MVP** | 8,000 - 12,000 | 8M - 15M | 4 - 8 hours | Prove the mixture works. Model answers general questions AND generates correct Mirth XML. Deployable for internal demo. |
| **Usable v1** | 20,000 - 30,000 | 20M - 40M | 12 - 24 hours | Production-ready for the core slice: NL -> Mirth XML, Rhino JS, HL7 v2 parsing, basic FHIR. Handles multi-turn debugging. |
| **Strong Production** | 50,000 - 80,000 | 50M - 100M | 2 - 4 days | Full coverage of all EHR vendors, IHE profiles, DICOM, complex FHIR workflows, DPO alignment, edge-case hardening. |

### Token Budget Per Example

| Data Type | Avg Tokens/Example | Reason |
|---|---|---|
| General assistant (single-turn) | 300 - 600 | Short Q&A: math, facts, casual chat |
| Healthcare code generation (single-turn) | 800 - 1,500 | Instruction + full code output with comments |
| Multi-turn conversation | 1,500 - 3,000 | 3-6 turns of context-dependent dialogue |
| Mirth channel XML generation | 1,200 - 2,500 | XML is verbose; full channel configs are long |

### Critical Constraint: max_seq_length

Current training config uses `max_seq_length: 2048`. This is too short for multi-turn conversations and full Mirth channel XML outputs. **Recommendation: increase to 4096 for v1, consider 8192 for production.** This doubles memory per example but is essential for the model to learn complete outputs rather than truncated fragments.

---

## 2. Recommended Dataset Composition

### The Mixture

| Category | % of Dataset | MVP (10k) | v1 (25k) | Production (65k) | Rationale |
|---|---|---|---|---|---|
| **General assistant** | 25-30% | 2,500 | 7,000 | 18,000 | Prevents catastrophic forgetting. Preserves math, reasoning, casual chat, general coding. |
| **Healthcare domain (single-turn)** | 40-45% | 4,500 | 11,000 | 28,000 | Core value prop. NL -> Mirth XML, HL7 parsing, FHIR mapping, EHR API workflows. |
| **Multi-turn interactive** | 15-20% | 1,500 | 4,500 | 13,000 | Debugging, clarification, iterative building, code review. Makes the model usable in real workflows. |
| **Identity & behavioral anchors** | 3-5% | 500 | 1,000 | 2,500 | "Who are you?", safety boundaries, refusal patterns, NexiFuse personality. |
| **DPO preference pairs** | 0% (MVP), 5% (v1+) | 0 | 1,500 | 3,500 | Chosen/rejected pairs for alignment. Not needed for MVP but critical for production quality. |

### Why This Ratio Is Correct

**25-30% general data is not wasted capacity** — it is structural load-bearing. Research from Llama 3, Qwen 2.5, and DeepSeek-V3 teams consistently shows that fine-tuning an instruction model on pure domain data destroys 40-60% of base model capability within 1-2 epochs. The general data acts as a regularizer. It does not compete with domain learning — it preserves the foundation that domain learning builds on.

**40-45% healthcare domain is the minimum for reliable code generation.** Healthcare integration outputs must be syntactically correct, structurally valid, and semantically accurate (correct segment names, correct FHIR resource types, correct API endpoints). This requires high density of domain examples. Below 35%, the model generates plausible-looking but incorrect outputs (hallucinated segment names, wrong FHIR resource structures).

**15-20% multi-turn is what separates a code generator from an assistant.** A model that can only answer one-shot questions is a fancy autocomplete. Real developers iterate: "make a channel" -> "add routing" -> "it's throwing a null pointer on PID-3" -> "now add TLS." Without multi-turn training, the model treats each message as independent and loses context.

**3-5% identity anchors are cheap insurance.** Without them, the model randomly adopts personas from training data ("As an AI language model..." or echoing teacher model identity). A small set of well-crafted identity examples locks the model into the NexiFuse persona.

### Sub-Category Breakdown for Healthcare Domain Data (v1 target: 11,000)

| Sub-Category | Count | Priority |
|---|---|---|
| Mirth Connect channel XML generation | 2,000 | P0 — highest value, most complex output |
| Rhino JavaScript transformers | 2,000 | P0 — core Mirth workflow |
| HL7 v2 message parsing & transformation | 1,500 | P0 — foundational |
| HL7 v2 to FHIR R4 conversion | 1,500 | P0 — most requested workflow |
| FHIR R4 resource creation & bundles | 1,200 | P1 |
| EHR vendor API integration (Epic, Cerner, Athena, Meditech, Veradigm) | 1,200 | P1 |
| Error handling & validation patterns | 800 | P1 |
| Security, PHI-safe logging, compliance | 500 | P2 |
| IHE profiles & DICOM (production only) | 300 | P2 |

### Sub-Category Breakdown for General Assistant Data (v1 target: 7,000)

| Sub-Category | Count | Purpose |
|---|---|---|
| Math & arithmetic | 1,200 | Restore basic reasoning the model lost |
| General coding (Python, JS, SQL) | 1,500 | Code generation is a transferable skill |
| Computer science & technical Q&A | 1,200 | Networking, databases, OS, security concepts |
| Reasoning & comparison | 1,000 | Trade-off analysis, architecture decisions |
| Casual conversation & advice | 800 | Personality, productivity, general helpfulness |
| Summarization & explanation | 300 | "Explain X to me like..." |

---

## 3. Recommended Teacher Model Strategy

### The Core Principle

No single teacher model excels at everything. The optimal strategy is a **2-model local stack** where each teacher is assigned to the task where it has the strongest comparative advantage. Both models run on DGX Spark via Ollama. Total cost: zero.

### The Fully Local Stack

#### Primary Teacher: DeepSeek-R1 70B (via Ollama, quantized)

**Assign to:** Complex healthcare code generation, multi-turn conversations, general reasoning, DPO chosen responses, vendor API workflows.

**Why DeepSeek-R1 70B is the anchor teacher:**
- **Reasoning traces.** DeepSeek-R1's `<think>` tags produce step-by-step reasoning that transfers directly to the student model (same architecture family). The student literally learns how to think through HL7 segment parsing.
- **Architecture alignment.** The student model (DeepSeek-R1-Distill-Llama-8B) was literally distilled from this teacher family. The knowledge transfer pathway is maximally efficient. No other local model has this advantage.
- **Strong code generation.** DeepSeek-R1 70B matches or exceeds GPT-4-level on code benchmarks, particularly for structured output generation (XML, JSON, JavaScript).
- **200k+ context window.** Can ingest full Mirth channel XML, HL7 spec excerpts, and FHIR profiles as reference context during generation.
- **Multi-turn capability.** Among local models, DeepSeek-R1 70B has the strongest ability to maintain coherence across conversation turns. Not as natural as Claude, but sufficient with good prompting.
- **Free and local.** No API costs. Full control. No data leaves the premises.

**Trade-offs:** Slow. On DGX Spark with 128GB unified memory, expect 2-5 minutes per complex generation (Mirth XML, multi-turn conversations). For the v1 dataset (25k examples, ~40% via this teacher = ~10k generations), budget approximately **14-35 days of continuous generation** or run multiple quantized instances in parallel. Plan for this wall time in the schedule.

**Mitigation for speed:** Use DeepSeek-R1 70B only for P0 (complex) tasks. Route all simpler tasks to the faster Qwen 32B teacher.

**Recommended Ollama model:** `deepseek-r1:70b` (Q4_K_M quantization, ~40GB VRAM)

#### Secondary Teacher: Qwen 2.5 Coder 32B (via Ollama, quantized)

**Assign to:** Bulk domain code generation, general assistant data, simple HL7 parsing, Rhino JS templates, error handling patterns, math and general knowledge Q&A.

**Why Qwen 2.5 Coder 32B as the workhorse:**
- **3-5x faster than DeepSeek-R1 70B.** At Q4_K_M quantization (~18GB VRAM), Qwen 32B generates responses in 20-60 seconds vs. minutes for DeepSeek-R1. This makes it viable for bulk generation of thousands of examples.
- **Strong code output.** Qwen 2.5 Coder is specifically trained for code generation. It produces cleaner, more concise code than general-purpose models. Excellent for JavaScript, Python, SQL, XML.
- **Good instruction following.** Qwen 2.5 follows structured prompts well, making it reliable for templated generation (math Q&A, general knowledge, simple code tasks).
- **Runs alongside DeepSeek-R1.** On DGX Spark with 128GB, both models can be loaded simultaneously (~40GB + ~18GB = ~58GB), enabling parallel generation pipelines.

**Trade-offs:** Weaker reasoning than DeepSeek-R1, especially on complex multi-step problems. Shorter effective context window for coherent output. Does not produce reasoning traces. Multi-turn conversation quality is noticeably weaker — conversations tend to be generic and lack the diagnostic depth that DeepSeek-R1 achieves.

**Current setup uses qwen2.5-coder:7b.** This is too weak. The 7B teacher is similar in capacity to the 8B student — the student cannot learn much from a teacher of equal size. **Upgrading to 32B is mandatory.**

**Recommended Ollama model:** `qwen2.5-coder:32b` (Q4_K_M quantization, ~18GB VRAM)

#### DPO Preference Pairs: Self-Play + DeepSeek-R1 70B

For DPO alignment (v1 and production), no paid API is needed:
- **Rejected responses:** Generated by the student model itself (outputs that fail validation)
- **Chosen responses:** Generated by DeepSeek-R1 70B for the same prompts

This creates a natural quality gap. The student's failures (syntax errors, hallucinated segments, wrong FHIR types) paired with DeepSeek-R1's correct outputs teach the student to prefer correct behavior. This is the standard self-play DPO approach used by DeepSeek's own team.

### Teacher Stack Summary

| Teacher | Role | Data Types | Volume | Speed | VRAM |
|---|---|---|---|---|---|
| **DeepSeek-R1 70B** | Complex reasoning + multi-turn | Mirth XML, complex HL7, FHIR conversion, vendor APIs, multi-turn conversations, DPO chosen | 40% of generation | Slow (2-5 min/example) | ~40GB |
| **Qwen 2.5 Coder 32B** | Bulk generation + general data | General assistant (math, coding, Q&A), simple HL7 parsing, Rhino JS, error handling, PHI logging | 60% of generation | Fast (20-60 sec/example) | ~18GB |
| **Student model (8B)** | DPO rejected responses | Preference pairs — student failures | DPO only (<5%) | Very fast | ~6GB |

### Generation Time Budget (v1: 25,000 examples)

| Teacher | Examples | Time/Example | Total Time | Parallelizable? |
|---|---|---|---|---|
| DeepSeek-R1 70B | ~10,000 | 3 min avg | ~500 hours (~21 days) | Yes — can run 2 instances at Q4 on 128GB |
| Qwen 2.5 Coder 32B | ~15,000 | 40 sec avg | ~170 hours (~7 days) | Yes — runs alongside DeepSeek-R1 |
| **Effective total** (parallel) | 25,000 | | **~14-21 days** | Both teachers generating simultaneously |

This is the real constraint of a fully local stack. The generation phase takes 2-3 weeks for v1. Plan the pipeline to start generating early (Week 1) while the code infrastructure is still being built.

---

## 4. Synthetic Data Generation Plan

### 4.1 HL7 v2 Message Parsing

**What the input prompt should look like:**
A natural language request referencing specific HL7 concepts — segment names, field positions, message types, trigger events. Examples: "Extract the patient's insurance group number from an ADT^A01 message," "Parse all OBX segments from an ORU^R01 and return lab results with units and reference ranges," "Build an HL7 ACK for a received VXU^V04 message."

**What the expected output should look like:**
Working code (Python or Rhino JS) that correctly references HL7 segment positions (PID-3 for patient ID, OBX-5 for observation value, etc.), handles field repetitions and component separators, includes null-safety checks, and has meaningful comments explaining the HL7 semantics.

**How to validate before adding to training:**
- Verify all referenced segment names exist in the HL7 v2.x specification (MSH, PID, PV1, OBR, OBX, IN1, NK1, AL1, etc.)
- Verify field position numbers are within valid range for each segment (e.g., PID has 30 fields, OBX has 25)
- If output contains a raw HL7 message, validate MSH header structure and segment ordering
- Run security scan for hardcoded PHI patterns (real SSNs, real patient names)
- Reject outputs that hallucinate non-existent segments (e.g., "PAT" segment, "OBX-30" field)

**Teacher assignment:** DeepSeek-R1 70B for complex multi-segment parsing. Qwen 32B for single-segment extraction.

### 4.2 HL7 v2 to FHIR R4 Conversion

**What the input prompt should look like:**
Mapping requests that specify source HL7 segments and target FHIR resources. Examples: "Convert this ADT^A01 message's PID segment into a FHIR R4 Patient resource with US Core extensions," "Map OBX lab results to FHIR Observation resources with proper LOINC coding," "Transform an HL7 AL1 segment into a FHIR AllergyIntolerance resource."

**What the expected output should look like:**
Code or mapping logic that correctly maps HL7 fields to FHIR resource attributes. Must include proper FHIR resource structure (resourceType, id, meta), correct data type mappings (HL7 CX -> FHIR Identifier, HL7 XPN -> FHIR HumanName), appropriate coding systems (LOINC, SNOMED, ICD-10), and handling of required vs. optional elements.

**How to validate before adding to training:**
- Parse output JSON and verify `resourceType` is a valid FHIR R4 resource type
- Validate against FHIR R4 StructureDefinitions (if schema directory available)
- Verify coding systems use correct URIs (http://loinc.org, http://snomed.info/sct)
- Check that Must Support elements are present for US Core profiles
- Reject outputs that invent FHIR resource types or attributes that don't exist in R4

**Teacher assignment:** DeepSeek-R1 70B for complex multi-resource bundles and conversion logic with reasoning traces.

### 4.3 Mirth Connect Channel XML Generation

**What the input prompt should look like:**
Natural language descriptions of integration workflows. Examples: "Create a Mirth channel that listens for HL7 ADT messages on port 6661 via MLLP, transforms them to FHIR JSON, and POSTs to https://fhir.example.com/r4," "Build a channel with a database reader source that polls for new orders every 30 seconds and generates HL7 ORM messages," "Set up an HTTP listener channel that receives FHIR Bundles and routes them to different destinations based on resource type."

**What the expected output should look like:**
Valid Mirth Connect channel XML with proper structure: `<channel>` root with `<id>`, `<name>`, `<sourceConnector>` (with correct connector type, transport, properties), `<destinationConnectors>` with one or more `<connector>` elements, `<transformer>` elements with JavaScript steps. The XML must be parseable and structurally correct for import into Mirth Connect.

**How to validate before adding to training:**
- Parse as XML — must be well-formed
- Verify root element is `<channel>` with required child elements
- Check that connector types are valid Mirth types (TCP Listener, HTTP Listener, Database Reader, HTTP Sender, File Writer, etc.)
- Verify transformer steps contain syntactically valid JavaScript
- Reject channels that reference non-existent Mirth connector types or properties
- Check for reasonable defaults (enabled=true, proper protocol versions)

**Teacher assignment:** DeepSeek-R1 70B exclusively. This is the highest-value output type and warrants the strongest teacher.

### 4.4 Rhino JavaScript Transformers

**What the input prompt should look like:**
Transformer logic requests in the context of Mirth Connect's JavaScript environment. Examples: "Write a transformer that extracts PID-3 patient MRN and maps it to channelMap," "Build a filter that rejects messages where MSH-9 is not ADT^A01 or ADT^A08," "Create a JavaScript code template that converts HL7 date format (YYYYMMDD) to ISO 8601."

**What the expected output should look like:**
Rhino-compatible JavaScript that uses Mirth's built-in objects and methods correctly: `msg` (E4X XML), `channelMap`, `globalChannelMap`, `responseMap`, `connectorMessage`, `logger`, `DateUtil`, `SerializerFactory`. Must use E4X syntax for XML access in Mirth (e.g., `msg['PID']['PID.3']['PID.3.1'].toString()`), not DOM methods.

**How to validate before adding to training:**
- JavaScript bracket/brace matching (syntactic validity)
- Check for Mirth-specific objects (msg, channelMap, etc.) — outputs referencing `document.getElementById` or `fetch()` are browser JS, not Mirth JS
- Verify E4X syntax patterns for HL7 field access
- Reject code that uses Node.js or browser APIs not available in Mirth's Rhino engine
- Run ESLint with Mirth-specific config if available

**Teacher assignment:** DeepSeek-R1 70B for complex transformers with routing logic. Qwen 32B for simple field extraction and mapping.

### 4.5 EHR Vendor API Integration Workflows

**What the input prompt should look like:**
Vendor-specific integration requests. Examples: "Connect to Epic's FHIR R4 sandbox using SMART Backend Services auth and retrieve a patient's medication list," "Query Cerner's FHIR endpoint for all Observations with category=laboratory for patient 12345," "Set up OAuth2 client credentials flow for Athenahealth's Practice API," "Implement bulk FHIR export from Epic using the kick-off, poll, and download pattern."

**What the expected output should look like:**
Working integration code with correct vendor-specific details: the right base URLs (or parameterized placeholders), correct OAuth2 flow for the vendor (Epic uses SMART Backend Services with JWT, Cerner uses standard OAuth2, Athena uses client credentials), proper FHIR search parameters, and vendor-specific quirks (Epic's `_count` vs standard `_count`, Cerner's custom extensions).

**How to validate before adding to training:**
- Verify OAuth2 flow matches the actual vendor specification
- Check that FHIR search parameters are valid for the target resource type
- Verify base URL patterns are plausible (or use placeholder tokens like `YOUR_EPIC_BASE_URL`)
- Reject outputs that mix up vendor-specific details (e.g., using Epic's JWT auth pattern for Cerner)
- Security scan for hardcoded credentials, real API keys, real patient data

**Teacher assignment:** DeepSeek-R1 70B for both workflow design and code implementation. Inject vendor documentation excerpts as context in the prompt.

### 4.6 Error Handling & Validation Patterns

**What the input prompt should look like:**
Requests for robust error handling in healthcare integration contexts. Examples: "Add retry logic with exponential backoff to this FHIR API call," "Write error handling for a Mirth channel that sends email alerts on critical lab results and logs non-critical failures," "Validate an incoming HL7 message has all required segments before processing."

**What the expected output should look like:**
Code patterns showing try/catch blocks with specific exception types, meaningful error messages that include context (message control ID, patient MRN for correlation, not PHI in logs), retry strategies appropriate for healthcare (idempotency awareness, duplicate message detection), and graceful degradation.

**How to validate before adding to training:**
- Verify error handling actually catches and handles exceptions (not empty catch blocks)
- Check that log messages don't contain PHI patterns
- Verify retry logic includes backoff (not infinite tight loops)
- Reject patterns that silently swallow errors without logging

**Teacher assignment:** Qwen 32B for standard patterns. DeepSeek-R1 70B for complex error recovery workflows.

### 4.7 Security & PHI-Safe Logging

**What the input prompt should look like:**
Security-focused requests. Examples: "Modify this logging code to mask SSN and MRN before writing to the application log," "Add HIPAA-compliant audit logging to this FHIR API endpoint," "Implement PHI detection and redaction for HL7 messages before storing in a debug log."

**What the expected output should look like:**
Code that demonstrates correct PHI handling: masking patterns (last 4 of SSN, hashed MRN), audit log fields (who, what, when, from where), TLS configuration for data in transit, credential management (environment variables or secret stores, never hardcoded).

**How to validate before adding to training:**
- Run the existing security pattern scanner (SSN regex, hardcoded password detection, API key patterns)
- Verify that example code shows masking, not just discusses it
- Reject examples that demonstrate PHI exposure as "correct" behavior
- Check that TLS examples use TLS 1.2+ (not SSL 3.0 or TLS 1.0)

**Teacher assignment:** DeepSeek-R1 70B. Security-sensitive content benefits from careful step-by-step reasoning, which DeepSeek-R1's `<think>` traces naturally provide.

### 4.8 Multi-Turn Debugging Conversations

**What the input prompt should look like:**
A scenario description that the teacher expands into a full conversation. Example scenario: "User has a Mirth channel that silently drops HL7 messages. Through 4 turns, the assistant helps trace the issue to a source filter that rejects messages missing an optional NK1 segment."

**What the expected output should look like:**
A 3-6 turn conversation where:
- Turn 1 (user): Describes the problem with context (error message, channel behavior, what they've tried)
- Turn 2 (assistant): Asks diagnostic questions or proposes a hypothesis with initial debugging steps
- Turn 3 (user): Provides additional information or results of suggested debugging
- Turn 4 (assistant): Identifies the root cause and provides a fix with explanation
- Optional turns 5-6: Follow-up on edge cases, prevention, or related improvements

**How to validate before adding to training:**
- Verify turns alternate correctly (user, assistant, user, assistant...)
- Check minimum 2 turns (1 exchange), maximum 8 turns
- Verify the final assistant turn contains actionable content (code fix, configuration change, or clear explanation)
- Reject conversations where the assistant contradicts itself between turns
- Reject conversations where user turns are unrealistically long or contain assistant-style content
- Security scan all assistant turns

**Teacher assignment:** DeepSeek-R1 70B. Multi-turn generation requires the stronger reasoning model. Use explicit prompting to force conversational style: instruct the teacher to generate natural, back-and-forth dialogue rather than monolithic responses. Include 2-3 example conversations in the prompt as few-shot demonstrations.

**Prompting strategy for natural multi-turn flow:**
DeepSeek-R1 tends toward monolithic answers. To mitigate, use a two-phase generation approach:
1. Phase 1: Generate a conversation outline (turn summaries) — forces the model to plan the flow
2. Phase 2: Generate each turn individually, feeding prior turns as context — prevents collapsing into a single response

---

## 5. Validation and Filtering Rules

### 5.1 Universal Rules (Apply to All Data)

| Rule | Action | Rationale |
|---|---|---|
| Empty instruction or empty output | **Reject** | Zero training signal |
| Output < 20 characters | **Reject** | Too short to be useful |
| Instruction > 2,000 tokens | **Truncate or reject** | Exceeds what users will realistically input |
| Output > 4,000 tokens | **Flag for review** | May exceed max_seq_length; split if possible |
| Exact duplicate (SHA-256 hash of output) | **Reject duplicate** | Wastes training budget |
| Near duplicate (Jaccard similarity > 0.9 on 5-char shingles) | **Reject later copy** | Reduces diversity |
| Contains real SSN pattern (XXX-XX-XXXX) not in allowlist | **Reject** | PHI safety |
| Contains hardcoded password/API key not a placeholder | **Reject** | Security risk in training data |
| Contains SQL injection pattern | **Reject** | Must not teach insecure patterns |
| Identity noise (author attribution, copyright from scraped code) | **Reject** | Prevents identity leakage |

### 5.2 Healthcare Domain Rules

| Rule | Action | Rationale |
|---|---|---|
| HL7 message output missing MSH segment | **Reject** | Structurally invalid |
| HL7 field reference out of valid range (e.g., PID-35 when PID has 30 fields) | **Reject** | Hallucinated field position |
| FHIR JSON output missing `resourceType` | **Reject** | Invalid FHIR resource |
| FHIR `resourceType` not in R4 specification | **Reject** | Hallucinated resource type |
| Mirth XML output not well-formed | **Reject** | Cannot be imported into Mirth |
| JavaScript output with unmatched brackets/braces | **Reject** | Syntax error |
| Rhino JS using browser/Node.js-only APIs (fetch, document, require) | **Reject** | Wrong runtime environment |
| FHIR coding system URI incorrect (e.g., "loinc.org" instead of "http://loinc.org") | **Flag for review** | Common teacher hallucination |
| Vendor API using wrong OAuth flow for the specified vendor | **Reject** | Factually incorrect |

### 5.3 Multi-Turn Conversation Rules

| Rule | Action | Rationale |
|---|---|---|
| Fewer than 2 turns | **Reject** | Not a conversation |
| More than 8 turns | **Truncate to 8** | Exceeds useful context for 8B model |
| Turns don't alternate user/assistant | **Reject** | Malformed conversation |
| First turn is not from user | **Reject** | Conversations must start with user |
| Last turn is not from assistant | **Reject** | Training target must be an assistant response |
| Any assistant turn is empty | **Reject** | No training signal for that turn |
| Assistant contradicts itself between turns | **Flag for review** | Incoherent conversation |
| User turn contains assistant-style content ("As an AI...") | **Reject** | Role confusion in training data |

### 5.4 Deduplication Strategy

- **Stage 1:** Exact dedup via SHA-256 of output (single-turn) or concatenated turns (multi-turn)
- **Stage 2:** Near-dedup via Jaccard similarity of 5-character shingles, threshold 0.90, compared within same domain only (for performance)
- **Stage 3:** Instruction dedup — if two examples have >0.85 Jaccard similarity on instructions AND same domain, keep only the one with longer output (more detailed response)

---

## 6. Recommended Next-Step Execution Plan

### Phase 1: Foundation (Week 1-2) — Build Infrastructure + Start Generation

**Goal:** Working pipeline for all three data tiers. Begin generation immediately (it's the bottleneck).

**Week 1:**
- Pull and test `deepseek-r1:70b` and `qwen2.5-coder:32b` on DGX Spark via Ollama — verify both fit in memory simultaneously
- **Start generation immediately** while building infrastructure in parallel:
  - Kick off Qwen 32B generating general assistant examples (math, coding, Q&A) — fast, can produce ~2,000/day
  - Kick off DeepSeek-R1 70B generating P0 healthcare examples (Mirth XML, HL7 parsing) — slow, ~500/day
- Add `ConversationTurn` and `ConversationExample` data models to the pipeline
- Implement general-purpose data generation function with 5 categories
- Broaden the system prompt from narrow "healthcare expert" to general assistant with healthcare specialty

**Week 2:**
- Implement multi-turn conversation generation (two-phase: outline then sequential turn generation)
- Extend prompt formatter for multi-turn Llama 3 chat template
- Update data cleaner and validator for new record types
- Continue generation — target: 5,000 general + 3,000 healthcare + 500 identity by end of Week 2
- Begin generating multi-turn conversations via DeepSeek-R1 70B (slow — start early)
- Clean, validate, format whatever is ready into an MVP dataset (~8-10k examples)
- **Train and evaluate MVP model** — verify it can answer "What is 2+4?" AND generate correct Mirth XML

### Phase 2: Scale to v1 (Week 3-4) — Full Dataset + Production Training

**Goal:** 25k validated dataset. Deployable model for NL -> Mirth XML, Rhino JS, HL7 v2, basic FHIR.

**Week 3:**
- Continue generation toward v1 targets — both teachers running continuously
- Scale healthcare domain to 11,000 examples (DeepSeek-R1 for P0, Qwen 32B for P1)
- Scale general assistant data to 7,000 examples
- Scale multi-turn conversations to 4,500 examples
- Implement domain-specific validation rules (HL7 field range checking, FHIR resourceType validation, Mirth XML structure checks, Rhino JS API allowlist)

**Week 4:**
- Run full cleaning and validation pipeline on all generated data
- Analyze rejection rates — if >30% of a category is rejected, review teacher prompts and regenerate
- Assemble final v1 dataset with correct mixture ratios
- Increase max_seq_length to 4096
- **Train v1 model** with full dataset
- Evaluate on held-out test set: general capability benchmarks + domain-specific correctness tests
- Begin generating DPO pairs: run student model on 2,000 prompts, collect failures, regenerate with DeepSeek-R1 70B as chosen

### Phase 3: Harden for Production (Week 5-6) — Alignment and Edge Cases

**Goal:** DPO-aligned model with edge case coverage. Ready for internal production deployment.

**Week 5:**
- Train DPO alignment pass using 1,500 preference pairs (student rejected + DeepSeek-R1 chosen)
- Add vendor-specific API integration examples (Epic, Cerner, Athena, Meditech, Veradigm) — 200+ examples per vendor
- Add edge case examples: malformed HL7 messages, FHIR validation errors, partial messages, timeout handling
- Add security-focused examples: PHI redaction, audit logging, TLS configuration
- Generate additional multi-turn scenarios targeting the weakest interaction patterns identified in v1 evaluation

**Week 6:**
- Expand to IHE profiles and DICOM basics if needed
- Final evaluation against held-out test suite
- A/B test v1 vs. production model on real developer queries
- Document model card with known limitations and supported workflows
- **Deploy production model** to internal team via Ollama + OpenAI-compatible API
- Set up continuous evaluation pipeline for monitoring production quality

---

## 7. Final Recommendations

### Final Dataset Target for Next Training Run (v1)

| Category | Count | Source |
|---|---|---|
| General assistant (single-turn) | 7,000 | Qwen 2.5 Coder 32B (local) |
| Healthcare domain (single-turn) | 11,000 | DeepSeek-R1 70B (P0) + Qwen 32B (P1/P2) (local) |
| Multi-turn conversations | 4,500 | DeepSeek-R1 70B (local) |
| Identity & behavioral anchors | 1,000 | Hand-crafted + Qwen 32B (local) |
| DPO preference pairs | 1,500 | Student failures + DeepSeek-R1 70B chosen (local) |
| **Total** | **25,000** | **100% local, 100% free** |

### Final Teacher Model Stack

| Model | VRAM | Role | Why This Model |
|---|---|---|---|
| **DeepSeek-R1 70B** (Ollama, Q4_K_M) | ~40GB | Complex domain code, multi-turn, DPO chosen | Reasoning traces, architecture match with student, strongest local model for structured code output |
| **Qwen 2.5 Coder 32B** (Ollama, Q4_K_M) | ~18GB | Bulk generation, general data, simple domain | 3-5x faster, strong code quality, handles volume |
| **Student (8B)** | ~6GB | DPO rejected responses | Self-play: student's own failures become training signal |
| **Total VRAM** | **~64GB** | Fits comfortably on DGX Spark 128GB | Room for OS, training framework, and batch processing |

### Concrete Example Dataset Mix (v1: 25,000 examples)

| # | Category | Sub-Category | Count | Teacher |
|---|---|---|---|---|
| 1 | General | Math & arithmetic | 1,200 | Qwen 32B |
| 2 | General | General coding (Python, JS, SQL) | 1,500 | Qwen 32B |
| 3 | General | CS & technical Q&A | 1,200 | Qwen 32B |
| 4 | General | Reasoning & comparison | 1,000 | DeepSeek-R1 70B |
| 5 | General | Casual conversation | 800 | Qwen 32B |
| 6 | General | Summarization & explanation | 300 | Qwen 32B |
| 7 | Healthcare | Mirth channel XML generation | 2,000 | DeepSeek-R1 70B |
| 8 | Healthcare | Rhino JS transformers | 2,000 | DeepSeek-R1 70B + Qwen 32B |
| 9 | Healthcare | HL7 v2 parsing & transformation | 1,500 | DeepSeek-R1 70B + Qwen 32B |
| 10 | Healthcare | HL7 v2 -> FHIR R4 conversion | 1,500 | DeepSeek-R1 70B |
| 11 | Healthcare | FHIR R4 resource creation | 1,200 | DeepSeek-R1 70B |
| 12 | Healthcare | EHR vendor API integration | 1,200 | DeepSeek-R1 70B |
| 13 | Healthcare | Error handling & validation | 800 | Qwen 32B |
| 14 | Healthcare | Security & PHI-safe logging | 500 | DeepSeek-R1 70B |
| 15 | Healthcare | IHE & DICOM basics | 300 | DeepSeek-R1 70B |
| 16 | Multi-turn | Debugging conversations | 1,200 | DeepSeek-R1 70B |
| 17 | Multi-turn | Clarification dialogues | 900 | DeepSeek-R1 70B |
| 18 | Multi-turn | Iterative code building | 900 | DeepSeek-R1 70B |
| 19 | Multi-turn | Code review & improvement | 600 | DeepSeek-R1 70B |
| 20 | Multi-turn | Migration guidance | 500 | DeepSeek-R1 70B |
| 21 | Multi-turn | Step-by-step walkthroughs | 400 | DeepSeek-R1 70B |
| 22 | Identity | NexiFuse persona & behavioral anchors | 1,000 | Hand-crafted + Qwen 32B |
| 23 | DPO | Preference pairs (chosen/rejected) | 1,500 | Student failures + DeepSeek-R1 70B |
| | **Total** | | **25,000** | **All local. All free.** |

---

*This plan prioritizes correctness and deployment readiness over benchmark metrics. The first milestone is a model that simultaneously answers "What is 2 + 4?" correctly and generates valid Mirth channel XML from natural language. Everything else follows from that foundation.*

*Total API cost: $0. Total cloud dependency: none. Everything runs on DGX Spark.*
