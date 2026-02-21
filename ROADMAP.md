# NexiFuse Health — Robust Expert: Complete Project Roadmap

## Project Summary

Build a domain-specific "Robust Expert" AI model for healthcare data interoperability, trained on-premise on NVIDIA DGX Spark (GB10 Grace Blackwell, 128GB unified memory). The model powers "vibe coding" — translating natural language intent into production-grade Mirth Connect, HL7 v2, FHIR R4, and EHR API integration code. The trained model is served locally via an OpenAI-compatible API and consumed by the Integrator desktop app.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION                             │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │GitHub Scraper│  │ Doc Ingestion    │  │ Teacher-Student       │  │
│  │(repos, code) │  │  (PDFs, HTML,    │  │ Data Factory          │  │
│  │              │  │  API specs)      │  │ (Llama 70B/DeepSeek)  │  │
│  └──────┬───────┘  └────────┬─────────┘  └───────────┬───────────┘  │
│         │                   │                        │              │
│         └───────────────────┼────────────────────────┘              │
│                             ▼                                       │
│                    Raw JSONL Store                                  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                       DATA PROCESSING                               │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ Data Cleaner │→ │ Validation Engine│→ │ DPO Pair Generator    │  │
│  │ (dedup,      │  │ (JS, XML, HL7,   │  │ (pass/fail → chosen/  │  │ 
│  │  normalize)  │  │  FHIR, security) │  │  rejected pairs)      │  │
│  └──────────────┘  └──────────────────┘  └───────────────────────┘  │
│                             │                         │             │
│                    Validated JSONL              DPO JSONL           │
└─────────────────────────────┬─────────────────────┬─────────────────┘
                              │                     │
┌─────────────────────────────▼─────────────────────▼─────────────────┐
│                         TRAINING                                    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Training Pipeline (Unsloth + LoRA on DGX Spark)             │   │
│  │   Base: DeepSeek-R1-Distill-Llama-70B (4-bit NF4)            │   │
│  │   LoRA: r=16/32, alpha=2x, all linear layers                 │   │
│  │   BF16 mixed precision, cosine LR, 128k context              │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐   │
│  │ DPO Alignment Stage (optional, from preference pairs)        │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐   │
│  │ GGUF Converter (merge LoRA → F16 + Q4_K_M via llama.cpp)     │   │
│  │ Generate Ollama Modelfile with NexiFuse system prompt        │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                        DEPLOYMENT                                   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Inference Server (Ollama/vLLM on DGX Spark)                 │   │
│  │  OpenAI-compatible: /v1/chat/completions                     │   │
│  │  HIPAA-compliant logging (no prompt content)                 │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐   │
│  │  Integrator Desktop App (Tauri 2 + React)                    │   │
│  │  Agent modes: agent, plan, debug, ask                        │   │
│  │  Context engine, schema mapping, code generation             │   │
│  │  VITE_AGENT_URL → http://localhost:8080                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: DGX Spark Environment Setup

**Goal:** Establish a production-ready training environment on the DGX Spark.

### 0.1 OS & Driver Verification
- Verify DGX OS (Ubuntu-based) is current
- Confirm NVIDIA Drivers R580+ and CUDA 13.0+ for Blackwell support
- Validate GPU visibility: `nvidia-smi` should show GB10 with 128GB unified memory
- Minimize background services to free ~120GB for training jobs

### 0.2 Python Environment
- Create isolated environment: `python -m venv nexifuse_env`
- Install core dependencies:
  ```bash
  pip install unsloth torch torchvision torchaudio
  pip install transformers datasets peft accelerate bitsandbytes
  pip install triton>=3.3.1  # Required for Blackwell kernels
  ```
- Build xFormers from source for Arm (Grace CPU):
  ```bash
  export TORCH_CUDA_ARCH_LIST="12.0"
  pip install xformers --no-build-isolation
  ```

### 0.3 Supporting Tools
- Install Ollama for teacher model serving and final inference
- Clone llama.cpp for GGUF conversion (build with CUDA support for Blackwell)
- Install ESLint + xmllint for validation engine
- Install Docker + NVIDIA Container Toolkit for reproducible environments

### 0.4 Directory Structure
```
nexifuse_project/
├── nexifuse/                  # Python package (all pipeline code)
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── models.py              # Shared data models
│   ├── scraper.py             # GitHub corpus scraper
│   ├── doc_ingester.py        # Documentation PDF/HTML ingestion
│   ├── data_factory.py        # Teacher-student synthetic data factory
│   ├── data_cleaner.py        # Cleaning, dedup, normalization
│   ├── validator.py           # Multi-format validation engine
│   ├── dpo_generator.py       # DPO preference pair generation
│   ├── training_pipeline.py   # Unsloth fine-tuning pipeline
│   ├── gguf_converter.py      # LoRA merge + GGUF conversion
│   ├── inference_server.py    # FastAPI OpenAI-compatible server
│   ├── prompt_formatter.py    # ChatML/Llama prompt templates
│   └── cli.py                 # CLI entry points
├── docs/                      # Raw documentation corpus by domain
│   ├── ehr_api/               # Epic, Cerner, Athena, Meditech, Veradigm
│   ├── hl7v2/                 # v2.3.1, v2.5.1, v2.8 specs + IGs
│   ├── fhir_r4/               # US Core, Blue Button, TEFCA
│   ├── mirth/                 # Mirth User Guide, channel examples
│   ├── ihe/                   # XDS.b, ITI profiles
│   └── dicom/                 # Conformance statements
├── data/                      # Generated training data
│   ├── raw/                   # Raw scraped + generated JSONL
│   ├── cleaned/               # Post-cleaning JSONL
│   ├── validated/             # Post-validation JSONL
│   └── dpo/                   # DPO preference pairs
├── schemas/                   # FHIR R4 JSON schemas for validation
├── config.yaml                # Pipeline configuration
├── tests/                     # Property-based + unit tests
├── integrator/                # Integrator desktop app (Tauri 2)
└── outputs/                   # Training outputs, adapters, GGUF files
```

---

## Phase 1: Documentation Corpus Assembly

**Goal:** Gather and organize the raw documentation that feeds the teacher model's context window.

### 1.1 EHR API Documentation
| EHR Platform | Sources | Key Patterns to Capture |
|---|---|---|
| Epic | open.epic.com sandbox APIs, Data Sharing Playbooks, Bulk Data Access tutorials | FHIR Bulk export (kick-off → poll → download NDJSON), SOAP web services, App Orchard auth |
| Oracle Cerner | fhir.cerner.com CapabilityStatements, Ignite specs | Tenant ID in URLs, mandatory search params (_id, patient) |
| AthenaHealth | docs.athenahealth.com OpenAPI/Swagger (800+ endpoints) | Certified Workflows (scheduling, check-in), Base64 document upload, rate limits |
| Meditech | home.meditech.com REST API specs, Patient Access guides | client_id/secret auth, EMPI query patterns |
| Veradigm | developer.veradigm.com Unity Spec, FHIR Endpoint Directory | Professional vs TouchWorks platform distinction, proprietary Unity JSON |

### 1.2 Healthcare Standards
- **HL7 v2.x:** v2.3.1, v2.5.1, v2.8 base standards + segment definitions (MSH, PID, PV1, OBX, ORC) + trigger events (ADT^A01, ORU^R01, SIU^S12)
- **HL7 v2 IGs:** ELR (state-level: Texas DSHS), Immunization (CDC VXU^V04), Lab Results (LRI), vendor specs (LabCorp, Quest)
- **FHIR R4:** US Core IG (v3.1.1 through v7.0.0), Must Support flags, cardinality constraints
- **FHIR Exchange:** CARIN Blue Button (ExplanationOfBenefit), TEFCA QTF, Facilitated FHIR IG
- **IHE:** XDS.b (ITI-41 Provide & Register, ITI-43 Retrieve), metadata requirements
- **DICOM:** Vendor conformance statements (GE, Canon, Ambu), Modality Worklist tag mappings

### 1.3 Mirth Connect
- Channel XML structures: `<channel>`, `<sourceConnector>`, `<destinationConnectors>`, `<properties>`
- JavaScript transformers: E4X parsing (`msg.toString()`), DatabaseConnectionFactory, Java class invocation
- Code template libraries: HL7 escaping, date conversion, PDF generation
- GitHub repos: nextgenhealthcare/connect-examples, SagaHealthcareIT/mirthsync, koratech/mirthconnect_channels-examples, nextgenhealthcare/fhir-example-channels

### 1.4 Documentation Ingestion Pipeline (`doc_ingester.py`)
- PDF text extraction (OCR for scanned docs) → structured text
- HTML scraping → strip tags, extract content
- OpenAPI/Swagger JSON → endpoint summaries
- Output: organized text files in `docs/` by domain, ready for context injection

---

## Phase 2: Data Generation (Target: 25,000+ cleaned examples)

**Goal:** Produce a massive, diverse training corpus using GitHub scraping and teacher-student synthesis.

### 2.1 GitHub Corpus Scraping
- Scrape configured repos (connect-examples, mirthsync, community channels)
- Extract .js, .xml, .json files matching healthcare integration patterns
- PHI/credential filtering (SSN, API keys, passwords → skip + log)
- Teacher model synthesizes instruction for each code file
- Output: raw JSONL with instruction, input, output, source metadata

### 2.2 Teacher-Student Synthetic Data Factory
- **Teacher models:** DeepSeek-R1-Distill-Llama-70B or Llama-3-70B via Ollama/vLLM
- **3-stage pipeline:**
  1. **Instruction generation:** Inject domain docs into teacher context → generate realistic user "vibes"
  2. **Code synthesis:** Teacher generates Mirth JS / Channel XML / FHIR mappings from instructions
  3. **Validation filtering:** Syntax check outputs, reject hallucinated APIs
- **CoT traces:** When using DeepSeek-R1, capture Chain-of-Thought reasoning ("We map PID.18 to Patient Account Number because...")
- **Domain targets:**

| Domain | Example "Vibe" Input | Expected Output |
|---|---|---|
| HL7 Parsing | "Extract patient insurance from this ADT message" | Mirth JS accessing IN1 segments |
| FHIR Conversion | "Turn this HL7 v2 PID into a FHIR Patient resource" | JS mapping PID-3→identifier, PID-5→name |
| EHR API | "Connect to Cerner R4 sandbox and pull allergies" | HTTP Sender config + OAuth2 + JSON parsing |
| Error Handling | "Send email if lab result is critical" | Conditional logic + SMTP sender |
| Security | "Log access but mask the MRN" | MRN masking + logger.info() |
| Mirth Channel | "Create a channel that receives HL7 ADT and writes to PostgreSQL" | Full channel XML with DB writer destination |

### 2.3 Prompt Formatting (`prompt_formatter.py`)
- Format all examples into ChatML/Llama template:
  ```
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  You are a healthcare integration expert specializing in Mirth Connect,
  HL7 v2, FHIR R4, and EHR API connectivity...
  <|eot_id|><|start_header_id|>user<|end_header_id|>
  {instruction}
  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  {output}
  <|eot_id|>
  ```
- Support 128k context windows for in-context learning (full vendor API reference in prompt)

---

## Phase 3: Data Processing

**Goal:** Clean, validate, and prepare the corpus for training.

### 3.1 Data Cleaning
- Normalize field types (non-string → string)
- MinHash near-duplicate detection on output field (threshold 0.9)
- Discard empty instruction+output pairs
- Skip malformed JSON lines with counter
- Report: input rows, output rows, duplicates removed, malformed skipped

### 3.2 Multi-Format Validation
- **JavaScript:** ESLint with Mirth-compatible (Rhino engine) ruleset
- **XML:** xmllint well-formedness (fallback: Python xml.etree)
- **HL7 v2:** MSH header, segment structure, field separators, encoding chars, required fields per message type
- **FHIR R4:** JSON schema validation against resource definitions (resourceType, required fields)
- **Security scan:** Hardcoded credentials, unmasked PHI (SSN/MRN patterns), SQL injection

### 3.3 DPO Preference Pair Generation
- Match pass/fail validation results for same instruction
- Passing output → chosen, failing output → rejected
- Target: 1,000+ preference pairs
- Output: JSONL with prompt, chosen, rejected fields

---

## Phase 4: Model Training on DGX Spark

**Goal:** Fine-tune a 70B-class model with domain expertise.

### 4.1 Base Model Configuration
- **Primary:** DeepSeek-R1-Distill-Llama-70B (reasoning capabilities for complex integration logic)
- **Fallback:** Llama-3-70B-Instruct
- **Quantization:** 4-bit NF4 via Unsloth (~40GB weights, leaving ~88GB for gradients/activations/context)

### 4.2 LoRA Configuration
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (all linear layers)
- **Rank:** r=16 or r=32
- **Alpha:** 2x rank (32 or 64)
- **Dropout:** 0 (standard for LoRA fine-tuning)

### 4.3 Training Hyperparameters
- **Micro-batch size:** 1 or 2
- **Gradient accumulation:** 16 steps (effective batch: 16-32)
- **Learning rate:** 2e-4 with cosine decay
- **Warmup:** 5-10% of total steps
- **Precision:** BF16 mixed precision (native Blackwell support)
- **Max sequence length:** 128k tokens
- **Epochs:** 3

### 4.4 Memory Management
- Monitor via `nvidia-smi` — ensure no NVMe swap thrashing
- OOM handling: catch CUDA errors, log batch/seq details, suggest reduced params
- Checkpoint at regular intervals for resumability

### 4.5 DPO Alignment (Optional Second Stage)
- After SFT, run DPO training using preference pairs
- Aligns model to prefer validated, correct outputs over broken ones

---

## Phase 5: Model Export & Conversion

**Goal:** Convert trained model to deployable format.

### 5.1 LoRA Merge
- Merge adapter into base weights → HuggingFace safetensors
- Sequential layer processing for 70B (memory-efficient)

### 5.2 GGUF Conversion
- Convert merged model via llama.cpp
- Produce F16 (full precision) and Q4_K_M (quantized) GGUF files

### 5.3 Ollama Modelfile
- Generate Modelfile with:
  - FROM directive → GGUF path
  - SYSTEM → NexiFuse healthcare integration expert prompt
  - PARAMETER temperature 0.1
  - PARAMETER stop tokens (architecture-appropriate)

---

## Phase 6: Deployment & Integration

**Goal:** Serve the model and connect it to the Integrator app.

### 6.1 Inference Server
- Serve GGUF model via Ollama or vLLM on DGX Spark
- Expose OpenAI-compatible endpoint: `http://localhost:8080/v1/chat/completions`
- Streaming + non-streaming response modes
- HIPAA-compliant logging (timestamp, model, token count — no prompt content)
- Error handling: 404 (unknown model), 422 (malformed request), 502 (backend down), 503 (loading), 504 (timeout)

### 6.2 Integrator App Connection
- Set `VITE_AGENT_URL=http://localhost:8080` in Integrator `.env`
- Update default model in `integrator/src/lib/settings.ts` to `nexifuse-robust-expert`
- The Integrator's agent adapter (`src/agent.ts`) already calls `/v1/chat/completions` — no code changes needed
- Context engine injects documentation into agent prompts for RAG-style augmentation

### 6.3 Functional Validation
- Deploy generated channels to headless Mirth Connect (Docker)
- Feed Synthea-generated HL7 messages as test data
- Verify correct transformation at destinations
- Use results to generate additional DPO pairs for iterative improvement

---

## Phase 7: Iterative Improvement Loop

**Goal:** Continuously improve model quality.

### 7.1 Feedback Collection
- Integrator app tracks agent response quality (user accepts/rejects generated code)
- Validation engine scores model outputs on new prompts
- Failed generations feed back into DPO preference dataset

### 7.2 Corpus Expansion
- Add new EHR vendor docs as they become available
- Scrape new community Mirth repos
- Generate examples for edge cases identified during production use

### 7.3 Retraining Cycle
- Merge new data into corpus → clean → validate → retrain
- Compare metrics against previous model version
- A/B test in Integrator app

---

## Execution Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Phase 0 | DGX Spark environment ready, all deps installed |
| 1-2 | Phase 1 | Documentation corpus organized in `docs/` |
| 2-3 | Phase 2 | 25,000+ raw training examples in JSONL |
| 3-4 | Phase 3 | Cleaned, validated corpus + DPO pairs |
| 4-5 | Phase 4 | Fine-tuned 70B model (LoRA adapter) |
| 5 | Phase 5 | GGUF files + Ollama Modelfile |
| 5-6 | Phase 6 | Inference server running, Integrator connected |
| 6+ | Phase 7 | Iterative improvement cycle |

---

## Key Technical Decisions

1. **DeepSeek-R1-Distill-Llama-70B over Llama-3-70B:** CoT reasoning traces improve code generation quality and reduce hallucinations for complex integration logic
2. **4-bit NF4 quantization:** Fits 70B model in 128GB unified memory with headroom for 128k context
3. **All-linear-layer LoRA:** Maximizes domain adaptation without full fine-tuning memory cost
4. **JSONL as interchange format:** Every pipeline stage reads/writes JSONL — resumable, debuggable, version-controllable
5. **Ollama for inference:** Simpler deployment than vLLM for single-model on-premise use; vLLM available as fallback for multi-model serving
6. **On-premise only:** Zero data leaves the DGX Spark — satisfies HIPAA and data sovereignty requirements

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| 70B model doesn't fit in memory during training | Fall back to 4-bit MXFP4, reduce LoRA rank to 8, or use 30B model |
| Teacher model generates hallucinated APIs | Validation engine cross-references against source documentation |
| Insufficient training data quality | MinHash dedup + multi-format validation + security scanning |
| DGX Spark Arm CPU incompatible with xFormers | Build from source with TORCH_CUDA_ARCH_LIST="12.0" |
| Model generates PHI in outputs | Security scanner in validation engine + inference-time output filtering |
| Mirth Rhino JS dialect not standard | Custom ESLint config for Rhino engine + E4X support |
