# NexiFuse Health — Robust Expert: Complete Project Roadmap

## Project Summary

Build a domain-specific "Robust Expert" AI model for healthcare data interoperability, trained on-premise on NVIDIA hardware. The model powers "vibe coding" — translating natural language intent into production-grade Mirth Connect, HL7 v2, FHIR R4, and EHR API integration code, while retaining general assistant capabilities. The trained model is served locally via an OpenAI-compatible API and consumed by the Integrator desktop app.

**Key Insight:** A well-curated 25k-example dataset with the right mixture (healthcare + general + multi-turn + identity) outperforms a 100k-example dataset with the wrong mixture on an 8B model. Data quality, mixture ratio, and validation rigor are the moat — not volume.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION                             │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │GitHub Scraper│  │ Doc Ingestion    │  │ Teacher-Student       │  │
│  │(repos, code) │  │  (PDFs, HTML,    │  │ Data Factory          │  │
│  │              │  │  API specs)      │  │ (DeepSeek-R1 70B +    │  │
│  │              │  │                  │  │  Qwen 2.5 Coder 32B)  │  │
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
│  │  Training Pipeline (Unsloth + LoRA, multi-GPU DDP)           │   │
│  │   Base: DeepSeek-R1-Distill-Llama-8B (4-bit NF4)            │   │
│  │   LoRA: r=32, alpha=64, all linear layers                    │   │
│  │   BF16 mixed precision, cosine LR, max_seq=4096              │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐   │
│  │ DPO Alignment Stage (from preference pairs)                  │   │
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
│  │  Inference Server (Ollama on-premise)                        │   │
│  │  OpenAI-compatible: /v1/chat/completions                     │   │
│  │  HIPAA-compliant logging (no prompt content)                 │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐   │
│  │  Integrator Desktop App (Tauri 2 + React)                    │   │
│  │  Agent modes: agent, plan, debug, ask                        │   │
│  │  Context engine, schema mapping, code generation             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Progress Summary

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 0: Environment Setup | **COMPLETE** | 100% |
| Phase 1: Documentation Corpus | **COMPLETE** | 100% |
| Phase 2: Data Generation | **COMPLETE** | 100% — 22,124 raw examples (healthcare + general + multi-turn + scraped) |
| Phase 3: Data Processing | **COMPLETE** | 100% — 18,055 cleaned → 17,661 validated (97.8%) → 35,394 formatted |
| Phase 4: Model Training (MVP) | **COMPLETE** | 100% — 8B model trained on 9.3k, loss 0.2256, deployed |
| Phase 5: Model Export (MVP) | **COMPLETE** | 100% — GGUF Q4_K_M (4.6 GB) |
| Phase 6: Deployment | **COMPLETE** | 100% — Ollama + FastAPI + Integrator |
| Phase 7: v1 Training & Improvement | **IN PROGRESS** | 60% — v1 data ready, training at step 3,000/13,275 (22.6%) |
| Phase 8: MedGemma v2 Upgrade | **PLANNED** | 0% — after v1 completion |

---

## Phase 0: Environment Setup — COMPLETE

**Goal:** Establish a production-ready training environment.

### Completed
- [x] Ubuntu environment with NVIDIA drivers and CUDA
- [x] Python 3.12 venv with Unsloth, PyTorch 2.10, Transformers, Accelerate, bitsandbytes
- [x] Ollama installed for teacher model serving and inference
- [x] llama.cpp built for GGUF conversion
- [x] 8x NVIDIA L4 GPUs (22 GB each) verified and tested with DDP
- [x] Directory structure established

---

## Phase 1: Documentation Corpus Assembly — COMPLETE

**Goal:** Gather and organize raw documentation for teacher model context.

### Completed
- [x] EHR API documentation (Epic, Cerner, Athena, Meditech, Veradigm)
- [x] HL7 v2 specs (v2.3.1, v2.5.1, v2.8) + trigger events + segment definitions
- [x] FHIR R4 US Core IG + resource definitions
- [x] Mirth Connect user guide + channel examples
- [x] IHE profiles (XDS.b, PIX, PDQ) + DICOM conformance
- [x] Documentation ingestion pipeline (`doc_ingester.py`) operational

---

## Phase 2: Data Generation — COMPLETE

**Goal:** Produce 25,000+ cleaned examples with balanced mixture.

### Target vs Actual Dataset Composition (v1)

| Category | Target | Actual | Teacher | Status |
|----------|--------|--------|---------|--------|
| Healthcare domain (single-turn) | 11,000 | 12,961 | Llama 3 70B | **COMPLETE** |
| General assistant (single-turn) | 7,000 | 7,500 | Llama 3 70B + 8B | **COMPLETE** |
| Multi-turn conversations | 4,500 | 1,116 | Llama 3 70B | **COMPLETE** (lower count, sufficient for v1) |
| GitHub scraped code | — | 547 | — | **COMPLETE** |
| Identity & behavioral anchors | 1,000 | 72 | Hand-crafted | **Partial** (mixed into formatted data) |
| DPO preference pairs | 1,500 | 0 | Student failures + teacher chosen | **PENDING** (after v1 SFT) |
| **Total raw** | **25,000** | **22,124** | **All local, all free** | **COMPLETE** |

### Healthcare Sub-Categories

| Sub-Category | Target | Teacher | Priority |
|-------------|--------|---------|----------|
| Mirth Connect channel XML generation | 2,000 | DeepSeek-R1 70B | P0 |
| Rhino JavaScript transformers | 2,000 | DeepSeek-R1 70B + Qwen 32B | P0 |
| HL7 v2 message parsing & transformation | 1,500 | DeepSeek-R1 70B + Qwen 32B | P0 |
| HL7 v2 to FHIR R4 conversion | 1,500 | DeepSeek-R1 70B | P0 |
| FHIR R4 resource creation & bundles | 1,200 | DeepSeek-R1 70B | P1 |
| EHR vendor API integration | 1,200 | DeepSeek-R1 70B | P1 |
| Error handling & validation patterns | 800 | Qwen 32B | P1 |
| Security, PHI-safe logging | 500 | DeepSeek-R1 70B | P2 |
| IHE & DICOM basics | 300 | DeepSeek-R1 70B | P2 |

### General Assistant Categories (NEW — prevents catastrophic forgetting)

| Category | Count | Teacher |
|----------|-------|---------|
| Math & arithmetic | 1,200 | Qwen 32B |
| General coding (Python, JS, SQL) | 1,500 | Qwen 32B |
| CS & technical Q&A | 1,200 | Qwen 32B |
| Reasoning & comparison | 1,000 | DeepSeek-R1 70B |
| Casual conversation | 800 | Qwen 32B |
| Summarization & explanation | 300 | Qwen 32B |

### Multi-Turn Conversation Scenarios (NEW)

| Scenario | Count | Teacher |
|----------|-------|---------|
| Debugging conversations | 1,200 | DeepSeek-R1 70B |
| Clarification dialogues | 900 | DeepSeek-R1 70B |
| Iterative code building | 900 | DeepSeek-R1 70B |
| Code review & improvement | 600 | DeepSeek-R1 70B |
| Migration guidance | 500 | DeepSeek-R1 70B |
| Step-by-step walkthroughs | 400 | DeepSeek-R1 70B |

### Teacher Model Stack

| Teacher | VRAM | Role | Speed |
|---------|------|------|-------|
| **DeepSeek-R1 70B** (Q4_K_M) | ~40 GB | Complex domain, multi-turn, DPO chosen, reasoning traces | 2-5 min/example |
| **Qwen 2.5 Coder 32B** (Q4_K_M) | ~18 GB | Bulk generation, general data, simple domain tasks | 20-60 sec/example |
| **Student (8B)** | ~6 GB | DPO rejected responses (self-play) | Very fast |

**Actual setup:** Using `llama3:70b` as primary teacher and `llama3:8b` for faster bulk generation, both via Ollama.

### Completed
- [x] GitHub scraper operational (3 repos configured)
- [x] Synthetic data factory with resume support and domain-aware generation
- [x] Generated 12,961 healthcare domain examples (synthetic_run1.jsonl + synthetic.jsonl)
- [x] Generated 7,500 general assistant examples (5 categories × 1,500)
- [x] Generated 1,116 multi-turn conversation examples (6 scenarios × 6 domains)
- [x] Scraped 547 code examples from GitHub repos
- [x] Auto-detection of all `data/raw/*.jsonl` files
- [x] `generate-general` and `generate-conversations` CLI commands with `--model` override
- [x] 72 conversational/identity examples
- [x] Pulled and tested `llama3:70b` teacher model

### Remaining (for Production scale)
- [ ] Expand multi-turn conversations to 4,500 (currently 1,116)
- [ ] Expand identity/behavioral anchors to 1,000 (currently 72)
- [ ] Pull and test `qwen2.5-coder:32b` for faster bulk generation
- [ ] Scale total raw to 50k+ for production milestone

---

## Phase 3: Data Processing — COMPLETE

**Goal:** Clean, validate, and prepare the corpus for training.

### Completed
- [x] Data cleaner with 4-stage pipeline: normalize → exact dedup → near-dedup (Jaccard 0.9) → write
- [x] Identity/attribution leakage filtering
- [x] Multi-format validation engine (JavaScript, XML, HL7 v2, FHIR R4)
- [x] Security scanner with context-aware allowlist (SQL injection, SSN, passwords, bearer tokens)
- [x] DPO preference pair generator
- [x] Prompt formatter with Llama 3 and ChatML templates
- [x] Progress logging with per-file and per-domain statistics

### Current Pipeline Results (v1 dataset)
| Stage | Count |
|-------|-------|
| Raw input | 22,124 (5 JSONL files auto-detected) |
| After cleaning | 18,055 (dedup + normalization + identity filtering) |
| Validation passed | 17,661 (97.8% pass rate) |
| Validation failed | 394 (syntax errors) |
| Formatted (+ identity + conversations) | 35,394 |

### Completed
- [x] Multi-turn conversation validation rules (turn alternation, min/max turns)
- [x] Multi-turn conversation cleaning and formatting support
- [x] Extended prompt formatter for multi-turn Llama 3 chat template

### Remaining (for Production)
- [ ] Add domain-specific validation (HL7 field range checking, FHIR resourceType validation)
- [ ] Add instruction dedup (Jaccard 0.85 on instruction + same domain)

---

## Phase 4: Model Training — MVP COMPLETE

**Goal:** Fine-tune the model with domain expertise and general capabilities.

### MVP Training (COMPLETE)
- [x] DeepSeek-R1-Distill-Llama-8B base model loaded in 4-bit NF4
- [x] LoRA applied: r=32, alpha=64, all linear layers (83.9M trainable / 8.1B total = 1.03%)
- [x] SafeSFTTrainer with Unsloth int-loss fix
- [x] Multi-GPU DDP training (8x NVIDIA L4, 2 hours)
- [x] Final loss: 0.2256 over 5 epochs
- [x] LoRA adapter saved to `nexifuse_model_adapter/`

### MVP Results
| Parameter | Value |
|-----------|-------|
| Dataset | 9,302 examples (healthcare + 72 conversational) |
| GPUs | 8x NVIDIA L4 (22 GB each) |
| Effective batch size | 32 (1 x 4 x 8) |
| Training time | 2 hours |
| Final loss | 0.2256 |
| max_seq_length | 2048 |

### v1 Training (IN PROGRESS)

| Parameter | Value |
|-----------|-------|
| Dataset | 35,394 examples (balanced: healthcare + general + multi-turn + identity) |
| Max Seq Length | 4096 |
| Effective Batch Size | 4 (1 × 4 grad_accum × 1 GPU) |
| Epochs | 3 |
| Total Steps | 13,275 |
| Current Progress | Step 3,000/13,275 (22.6%), loss 0.205 |

- [x] Increase `max_seq_length` to 4096
- [x] Assemble balanced v1 dataset (22k raw → 18k cleaned → 35k formatted)
- [ ] Complete v1 SFT training (remaining ~10k steps)
- [ ] Re-export GGUF Q4_K_M from v1 adapter
- [ ] Evaluate on held-out test set: general capability + domain correctness
- [ ] Run DPO alignment pass with preference pairs

---

## Phase 5: Model Export & Conversion — COMPLETE

**Goal:** Convert trained model to deployable format.

### Completed
- [x] LoRA merge into base weights (16-bit safetensors)
- [x] GGUF F16 conversion via llama.cpp `convert_hf_to_gguf.py` (15 GB)
- [x] Q4_K_M quantization via `llama-quantize` (4.6 GB)
- [x] Ollama Modelfile generated with Llama 3 chat template + NexiFuse system prompt

---

## Phase 6: Deployment & Integration — COMPLETE

**Goal:** Serve the model and connect it to the Integrator app.

### Completed
- [x] Model registered with Ollama as `nexifuse-robust-expert`
- [x] FastAPI inference server on port 8080
  - `/health` — health check + model status
  - `/v1/models` — list models
  - `/v1/chat/completions` — chat completions (streaming + non-streaming)
  - CORS enabled for Integrator desktop app
- [x] Integrator desktop app (Tauri 2 + React) built and connecting
- [x] Model responds with production-quality healthcare integration code

### Configuration
| Setting | Value |
|---------|-------|
| Server URL | `http://<server-ip>:8080` |
| Model name | `nexifuse-robust-expert` |
| Backend | Ollama |
| GGUF size | 4.6 GB (Q4_K_M) |

---

## Phase 7: v1 Training & Iterative Improvement — IN PROGRESS

**Goal:** Complete v1 training on balanced dataset, align with DPO, harden for production.

Following the [Upgrade Plan](Upgrade_Plan_2026_3_11.md).

### Execution Plan

#### Phase 7.1: Foundation — COMPLETE
- [x] Pull and test `llama3:70b` on GCP 8x L4 cluster
- [x] Implement general-purpose data generation function (5 categories)
- [x] Implement multi-turn conversation generation (6 scenarios × 6 domains)
- [x] Add `generate-general` and `generate-conversations` CLI commands with `--model` override
- [x] Extend prompt formatter for multi-turn Llama 3 chat template
- [x] Update validator and cleaner for multi-turn and general data types
- [x] Generate 7,500 general assistant examples
- [x] Generate 1,116 multi-turn conversation examples
- [x] Scale healthcare domain to 12,961 examples
- [x] Run full clean → validate → format pipeline on 22k raw examples

#### Phase 7.2: v1 Training — IN PROGRESS
- [x] Increase max_seq_length to 4096
- [x] Assemble balanced v1 dataset (35,394 formatted examples)
- [ ] **Complete v1 SFT training** — currently at step 3,000/13,275 (22.6%), loss 0.205
- [ ] Re-export GGUF Q4_K_M from v1 adapter
- [ ] Deploy v1 model and benchmark vs MVP
- [ ] Begin generating DPO pairs (student failures + Llama 3 70B chosen)

#### Phase 7.3: Harden for Production — PENDING
- [ ] Train DPO alignment pass (1,500 preference pairs)
- [ ] Add vendor-specific API examples (Epic, Cerner, Athena — 200+ per vendor)
- [ ] Add edge case examples (malformed HL7, FHIR errors, timeouts)
- [ ] Expand multi-turn conversations to 4,500
- [ ] Expand identity anchors to 1,000
- [ ] Final evaluation against held-out test suite
- [ ] A/B test MVP vs. v1 model on real developer queries
- [ ] **Deploy production model**

---

## Phase 8: v2 — MedGemma Base Model Upgrade — PLANNED

**Goal:** Replace the DeepSeek-R1-Distill-Llama-8B base with Google's MedGemma 27B for significantly stronger clinical and FHIR capabilities out-of-the-box, reducing training data requirements for healthcare tasks.

### Why MedGemma?

| Advantage | Detail |
|-----------|--------|
| **Built-in FHIR comprehension** | MedGemma 27B was pre-trained on FHIR-based EHR data — understands Patient, Observation, Bundle resources natively |
| **Clinical knowledge** | Pre-trained on medical text, Q&A, radiology, pathology — generates clinically accurate HL7/FHIR messages (correct LOINC, ICD-10 codes) |
| **Open & free** | Available on Hugging Face for research and commercial use, no API dependency |
| **LoRA fine-tuning supported** | Same LoRA approach we use today; Google provides official fine-tuning notebooks |
| **On-premise deployment** | Runs locally on our 8x L4 GPUs — HIPAA compliant |

### Model Comparison

| Factor | v1: DeepSeek-R1-Distill-Llama-8B | v2: MedGemma 27B |
|--------|-----------------------------------|-------------------|
| Parameters | 8B | 27B |
| Medical knowledge | None (learned from training data) | Pre-trained on medical corpus + FHIR EHR data |
| FHIR understanding | Learned from ~2,000 synthetic examples | **Native** — trained on FHIR resources |
| HL7 v2 / Mirth | Learned from training data | Gap — needs fine-tuning (our data) |
| General chat | Good (Llama base) | Weaker — needs general data tier |
| GPU requirement (inference) | 1x L4 | 2-3x L4 (Q4 quantized) |
| GPU requirement (training) | 8x L4 with LoRA | 8x L4 with LoRA (QLoRA for memory) |

### Training Data Strategy (v2)

The key insight: MedGemma already knows FHIR/clinical — we can **reduce healthcare FHIR data** and **focus training on Mirth/HL7 v2** (MedGemma's gap).

| Tier | v1 Count | v2 Count | Change | Reason |
|------|----------|----------|--------|--------|
| Mirth Connect + HL7 v2 | ~6,000 | ~8,000 | +33% | MedGemma's gap — double down |
| FHIR R4 | ~3,000 | ~1,000 | -67% | MedGemma already knows FHIR natively |
| EHR API / IHE / DICOM | ~3,000 | ~2,000 | -33% | MedGemma has partial clinical knowledge |
| General assistant | ~7,500 | ~7,500 | Same | Still needed to prevent forgetting |
| Multi-turn conversations | ~2,520 | ~3,000 | +19% | Leverage MedGemma's clinical reasoning |
| Identity anchors | ~1,000 | ~1,000 | Same | |
| DPO preference pairs | ~1,500 | ~2,000 | +33% | Align MedGemma's verbose medical style |
| **Total** | **~25,000** | **~24,500** | ~Same | Less data needed, higher quality base |

### Implementation Plan

#### Phase 8.1: Evaluation & Setup
- [ ] Download MedGemma 27B from Hugging Face (`google/medgemma-27b-text-it`)
- [ ] Benchmark on NexiFuse test suite: FHIR generation, HL7 parsing, Mirth JS, general chat
- [ ] Compare zero-shot MedGemma 27B vs fine-tuned DeepSeek-R1 8B on domain tasks
- [ ] Test QLoRA training on 8x L4 (27B + 4-bit + LoRA r=16)
- [ ] Adapt prompt formatter for Gemma 3 chat template

#### Phase 8.2: Fine-Tuning
- [ ] Generate Mirth/HL7-focused training data (expand from v1)
- [ ] Fine-tune MedGemma 27B with QLoRA on balanced dataset
- [ ] Evaluate: domain accuracy + general capability + clinical correctness
- [ ] DPO alignment pass (medical style calibration)

#### Phase 8.3: Deployment
- [ ] Convert to GGUF (Q4_K_M, ~15 GB estimated)
- [ ] Update Ollama Modelfile with Gemma 3 chat template
- [ ] A/B test v1 (8B) vs v2 (27B MedGemma) on real developer queries
- [ ] Deploy as `nexifuse-robust-expert-v2`

#### Phase 8.4: Multimodal (Future)
- [ ] Evaluate MedGemma 4B multimodal for medical image → FHIR DiagnosticReport
- [ ] Add radiology/pathology image interpretation to Integrator app
- [ ] CT/MRI volume support via MedGemma 1.5

### Expected Outcomes

| Metric | v1 (8B DeepSeek) | v2 (27B MedGemma) | Improvement |
|--------|-------------------|-------------------|-------------|
| FHIR resource generation accuracy | ~85% | ~95%+ | Native FHIR knowledge |
| Clinical terminology correctness | ~70% | ~90%+ | Pre-trained on medical data |
| Mirth/HL7 code quality | ~85% | ~90% | More training data focused here |
| General chat capability | ~80% | ~75-80% | Similar (both need general data) |
| Inference speed (tokens/sec) | ~12 t/s | ~5-7 t/s | Slower (3x larger model) |

### References
- [MedGemma: Google's open models for health AI](https://research.google/blog/medgemma-our-most-capable-open-models-for-health-ai-development/)
- [MedGemma Developer Docs](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [MedGemma on Hugging Face](https://huggingface.co/google/medgemma-27b-text-it)
- [MedGemma GitHub](https://github.com/Google-Health/medgemma)

---

## Key Technical Decisions

1. **DeepSeek-R1-Distill-Llama-8B as student:** Architecture match with DeepSeek-R1 70B teacher enables efficient knowledge transfer. CoT reasoning traces transfer directly.
2. **Dual teacher stack (DeepSeek-R1 70B + Qwen 32B):** Optimal quality/speed tradeoff. Complex tasks use 70B, bulk generation uses 32B. Both run locally.
3. **25-30% general data is structural:** Research consistently shows fine-tuning on pure domain data destroys 40-60% of base capability within 1-2 epochs. General data acts as a regularizer.
4. **4-bit NF4 quantization:** Fits 8B model on single L4 GPU with headroom for training.
5. **max_seq_length 4096:** Increased from 2048 to support multi-turn conversations and full Mirth channel XML outputs.
6. **JSONL as interchange format:** Every pipeline stage reads/writes JSONL — resumable, debuggable, version-controllable.
7. **Ollama for inference:** Simple on-premise deployment. GGUF Q4_K_M serves at ~6.5 tokens/sec.
8. **100% on-premise:** Zero API costs. Zero data leaves the premises. Satisfies HIPAA and data sovereignty.

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Model catastrophically overfits to healthcare code | 25-30% general assistant data as regularizer |
| Teacher model generates hallucinated APIs | Validation engine cross-references against source documentation |
| Insufficient training data quality | MinHash dedup + multi-format validation + security scanning |
| CUDA OOM during training | Gradient offloading, single-GPU fallback, batch size=1 |
| max_seq_length too short for XML outputs | Increased to 4096; consider 8192 for production |
| 7B teacher too weak for student | Upgrade to 32B Qwen + 70B DeepSeek-R1 |
| Model loses identity/persona | 3-5% identity anchors + behavioral examples |
| DPO pairs not available | Self-play: student failures + DeepSeek-R1 chosen responses |

---

## Generation Time Budget (v1: 25,000 examples)

| Teacher | Examples | Time/Example | Total Time |
|---------|----------|-------------|------------|
| DeepSeek-R1 70B | ~10,000 | 3 min avg | ~500 hours (~21 days) |
| Qwen 2.5 Coder 32B | ~15,000 | 40 sec avg | ~170 hours (~7 days) |
| **Effective total** (parallel) | 25,000 | | **~14-21 days** |

Both teachers run simultaneously on DGX Spark (128 GB unified memory: ~40 GB + ~18 GB = ~58 GB). The generation phase is the primary bottleneck — plan to start early.
