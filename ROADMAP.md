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
| Phase 2: Data Generation | **IN PROGRESS** | 40% — MVP healthcare data done (9.3k); general + multi-turn pending |
| Phase 3: Data Processing | **COMPLETE** | 100% — pipeline works end-to-end |
| Phase 4: Model Training (MVP) | **COMPLETE** | 100% — 8B model trained, loss 0.2256 |
| Phase 5: Model Export | **COMPLETE** | 100% — GGUF Q4_K_M (4.6 GB) |
| Phase 6: Deployment | **COMPLETE** | 100% — Ollama + FastAPI + Integrator |
| Phase 7: Iterative Improvement | **IN PROGRESS** | 20% — v1 dataset expansion underway |

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

## Phase 2: Data Generation — IN PROGRESS

**Goal:** Produce 25,000+ cleaned examples with balanced mixture.

### Target Dataset Composition (v1: 25,000 examples)

| Category | % | Count | Teacher | Status |
|----------|---|-------|---------|--------|
| Healthcare domain (single-turn) | 40-45% | 11,000 | DeepSeek-R1 70B + Qwen 32B | **9,000 done** (6 domains x 1,500) |
| General assistant (single-turn) | 25-30% | 7,000 | Qwen 2.5 Coder 32B | **PLANNED** |
| Multi-turn conversations | 15-20% | 4,500 | DeepSeek-R1 70B | **PLANNED** |
| Identity & behavioral anchors | 3-5% | 1,000 | Hand-crafted + Qwen 32B | **72 done** (conversational.jsonl) |
| DPO preference pairs | 5% | 1,500 | Student failures + DeepSeek-R1 70B | **PLANNED** |
| **Total** | **100%** | **25,000** | **All local, all free** | **~9,072 done** |

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

**Note:** Current config uses `qwen2.5-coder:7b`. Upgrading to 32B is recommended — the 7B teacher is too close in capacity to the 8B student for effective knowledge transfer.

### Completed
- [x] GitHub scraper operational (3 repos configured)
- [x] Synthetic data factory with resume support and domain-aware generation
- [x] Generated 9,000 healthcare domain examples (1,500 x 6 domains)
- [x] Auto-detection of all `data/raw/*.jsonl` files
- [x] 72 conversational/identity examples

### Remaining
- [ ] Pull `deepseek-r1:70b` and `qwen2.5-coder:32b` on DGX Spark
- [ ] Implement general-purpose data generation (5 categories, 7,000 examples)
- [ ] Implement multi-turn conversation generation (6 scenarios, 4,500 examples)
- [ ] Expand identity/behavioral anchors to 1,000 examples
- [ ] Scale healthcare domain to 11,000 with higher-quality teacher models

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

### Current Pipeline Results
| Stage | Count |
|-------|-------|
| Raw input | 9,565 |
| After cleaning | 9,504 (50 dupes, 11 identity filtered) |
| Validation passed | 9,230 (97.1% pass rate) |
| Validation failed | 274 (syntax errors) |
| Formatted (+ conversational) | 9,302 |

### Remaining
- [ ] Add multi-turn conversation validation rules (turn alternation, min/max turns)
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

### v1 Training Plan (NEXT)
- [ ] Increase `max_seq_length` to 4096 (already in config)
- [ ] Train on balanced 25k dataset (healthcare + general + multi-turn + identity)
- [ ] Evaluate on held-out test set: general capability + domain correctness
- [ ] Run DPO alignment pass with 1,500 preference pairs

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

## Phase 7: Iterative Improvement — IN PROGRESS

**Goal:** Expand dataset, retrain with balanced mixture, harden for production.

### Execution Plan

#### Phase 7.1: Foundation (Week 1-2) — Build Infrastructure + Start Generation
- [ ] Pull and test `deepseek-r1:70b` + `qwen2.5-coder:32b` on DGX Spark
- [ ] Start Qwen 32B generating general assistant examples (~2,000/day)
- [ ] Start DeepSeek-R1 70B generating P0 healthcare examples (~500/day)
- [ ] Implement general-purpose data generation function (5 categories)
- [ ] Implement multi-turn conversation generation (two-phase: outline → turns)
- [ ] Extend prompt formatter for multi-turn Llama 3 chat template
- [ ] Update validator for multi-turn and general data types
- [ ] Assemble MVP balanced dataset (~8-10k examples)
- [ ] **Train and evaluate MVP balanced model** — verify general + domain capability

#### Phase 7.2: Scale to v1 (Week 3-4) — Full Dataset + Production Training
- [ ] Scale healthcare domain to 11,000 examples
- [ ] Scale general assistant to 7,000 examples
- [ ] Scale multi-turn conversations to 4,500 examples
- [ ] Run full cleaning and validation pipeline
- [ ] Increase max_seq_length to 4096
- [ ] **Train v1 model** with full 25k dataset
- [ ] Begin generating DPO pairs (student failures + DeepSeek-R1 70B chosen)

#### Phase 7.3: Harden for Production (Week 5-6) — Alignment and Edge Cases
- [ ] Train DPO alignment pass (1,500 preference pairs)
- [ ] Add vendor-specific API examples (Epic, Cerner, Athena — 200+ per vendor)
- [ ] Add edge case examples (malformed HL7, FHIR errors, timeouts)
- [ ] Final evaluation against held-out test suite
- [ ] A/B test MVP vs. v1 model on real developer queries
- [ ] **Deploy production model**

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
