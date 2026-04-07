# NexiFuse Health: Robust Expert

A domain-specific AI model for healthcare data interoperability, fine-tuned on **DeepSeek-R1-Distill-Llama-8B** using LoRA (Unsloth). Translates natural language into production-grade **Mirth Connect**, **HL7 v2**, **FHIR R4**, and **EHR API** integration code while retaining general assistant capabilities (math, reasoning, casual conversation, general coding).

Designed for fully on-premise deployment. Zero API costs. Zero data leaves the premises. The trained model is quantized to GGUF Q4_K_M (4.6 GB), served locally via Ollama with an OpenAI-compatible API, and consumed by the **Integrator** desktop app (Tauri 2 + React).

## Highlights

- **V3 deployed** with 100% pass rate across 8,901 benchmark prompts and zero failures
- **70k+ raw examples** spanning healthcare (57%), general (31%), conceptual (5%), multi-turn (3%), identity (3%), scraped (1%)
- **79,647 formatted training examples** including 16,921 V3 identity anchors (explicit, negative, injected)
- **Identity recall: 80.2%** (up from 27.4% in V2.5) with enhanced system prompt and post-processing
- **8-GPU distributed training** via torchrun DDP on 8x NVIDIA L4, completed in 5h 11m
- **Final training loss: 0.276** with strong domain alignment across all 10 categories
- **25+ CLI commands** covering data ingestion through alignment training in one tool
- **OpenAI-compatible API** as a drop-in replacement for any OpenAI client
- **Inference-time identity enforcement** with post-processing that blocks hallucinated identities
- **100% local, 100% free** with all training, generation, and inference on-premise

## Current Status

V3 deployed and benchmarked. See [ROADMAP.md](ROADMAP.md) for the full project roadmap.

### Milestones

| Milestone | Status | Details |
|-----------|--------|---------|
| **MVP** (8-12k examples) | COMPLETE | 9.3k healthcare-only dataset. Proved pipeline works end-to-end. |
| **V1** (20-30k examples) | COMPLETE | 21.2k raw, 8-GPU DDP, loss 0.175, grade B+. |
| **V2** (45k examples) | COMPLETE | 45.3k raw, 41k formatted, loss 0.27, grade A- (981/981 benchmark). |
| **V2.5** (58k examples) | COMPLETE | 58k raw, 51.8k formatted, loss 0.303, grade A- (8,901/8,901 benchmark). |
| **V3** (80k examples) | COMPLETE | 70k raw, 79.6k formatted, loss 0.276, identity recall 80.2%, 8,901/8,901 benchmark. |
| **V4** | NEXT | ORPO alignment, MedGemma 27B base, 100k+ dataset, full vendor coverage. |

### V3 Summary (Current)

| Phase | Status | Details |
|-------|--------|---------|
| V3 Identity Data | COMPLETE | 16,921 examples (5,000 explicit + 1,000 negative + 10,921 injected) |
| Data Processing | COMPLETE | Clean, validate, format with V3 identity data merged |
| SFT Training | COMPLETE | 8-GPU DDP, 2 epochs, 5h 11m, loss 0.276 |
| Inference Post-Processing | COMPLETE | Identity enforcement, hallucination blocking, short response retry |
| Export and Deploy | COMPLETE | Merged LoRA to F16 to Q4_K_M (4.6 GB), Ollama registered |
| 10K Benchmark | COMPLETE | 8,901/8,901 passed, zero errors |

### V3 Key Improvements over V2.5

| Metric | V2.5 | V3 | Change |
|--------|------|-----|--------|
| Identity NexiFuse mention | 27.4% | **80.2%** | +52.8pp |
| Identity hallucination | 177/1,000 | **22/1,000** | -87.6% |
| Training data (formatted) | 51,818 | **79,647** | +54% |
| Identity data percentage | 0.97% | **21.2%** | +20pp |
| Final training loss | 0.303 | **0.276** | Improved |

### Dataset Composition (V3)

| Source | Count | % of Raw |
|--------|-------|----------|
| Healthcare domain (HL7v2, FHIR R4, Mirth, EHR API, IHE, DICOM) | 33,000 | 47% |
| General assistant (math, coding, reasoning, casual, knowledge) | 18,000 | 26% |
| V3 Identity (explicit + negative + injected) | 16,921 | 24% |
| Conceptual explanations | 3,120 | 4% |
| Multi-turn conversations | 1,920 | 3% |
| Raw HL7 messages | 1,000 | 1% |
| GitHub scraped code | 547 | <1% |
| **Total formatted** | **79,647** | |

### Training Runs

| Metric | MVP | V1 | V2 | V2.5 | V3 |
|--------|-----|----|----|------|----|
| Raw examples | 9,302 | 21,763 | 45,346 | 58,087 | 70,338 |
| Formatted | 9,302 | 35,394 | 41,019 | 51,818 | 79,647 |
| LoRA rank | 32 | 32 | 64 | 64 | 64 |
| Epochs | 5 | 3 | 2 | 2 | 2 |
| Training time | ~2h | 2h 20m | 3h 10m | 3h 17m | 5h 11m |
| Final loss | 0.226 | 0.175 | 0.27 | 0.303 | 0.276 |
| Identity recall | ~10% | ~10% | 17% | 27.4% | 80.2% |
| Benchmark grade | N/A | B+ | A- | A- | A- |
| GGUF size | 4.6 GB | 4.6 GB | 4.6 GB | 4.6 GB | 4.6 GB |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA ACQUISITION                        │
│                                                             │
│   GitHub Scraper    Doc Ingestion     Teacher-Student        │
│   (repos, code)     (PDFs, HTML)      Data Factory          │
│         │                │            (DeepSeek-R1 70B +    │
│         │                │             Qwen 2.5 Coder 32B)  │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│                   Raw JSONL Store                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    DATA PROCESSING                           │
│                                                             │
│   Data Cleaner  →  Validator  →  DPO Generator              │
│   (dedup, norm,    (JS, XML,     (pass/fail →               │
│    identity)        HL7, FHIR,    preference                │
│                     security)     pairs)                    │
│         │                                                   │
│         ▼                                                   │
│   Prompt Formatter (Llama 3 chat template)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      TRAINING                                │
│                                                             │
│   SFT Fine-Tuning (Unsloth + LoRA, multi-GPU DDP)          │
│         │                                                   │
│   Optional DPO Alignment                                    │
│         │                                                   │
│   Merge LoRA → GGUF Q4_K_M (4.6 GB)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     DEPLOYMENT                               │
│                                                             │
│   Ollama (GGUF) → FastAPI Server → Integrator Desktop App   │
│                    (port 8080)      (Tauri 2 + React)       │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
├── nexifuse/                  # Core Python package
│   ├── cli.py                 # CLI entry points (20+ commands)
│   ├── config.py              # Configuration management
│   ├── scraper.py             # GitHub corpus scraper
│   ├── doc_ingester.py        # Documentation ingestion (PDF/HTML)
│   ├── data_factory.py        # Teacher-student synthetic data generation
│   ├── data_cleaner.py        # Dedup, normalization, identity filtering
│   ├── validator.py           # Multi-format validation + security scanning
│   ├── dpo_generator.py       # DPO preference pair generation
│   ├── prompt_formatter.py    # Llama 3 / ChatML prompt templates
│   ├── training_pipeline.py   # Unsloth SFT + multi-GPU DDP
│   ├── gguf_converter.py      # LoRA merge + GGUF conversion
│   └── inference_server.py    # FastAPI OpenAI-compatible server
├── integrator/                # Desktop app (Tauri 2 + React)
├── docs/                      # Raw documentation corpus by domain
├── data/                      # Training data (all pipeline stages)
│   ├── raw/                   # Scraped + synthetic JSONL
│   ├── cleaned/               # Post-cleaning JSONL
│   ├── validated/             # Post-validation (passed/failed)
│   ├── formatted/             # Chat-template formatted for training
│   ├── dpo/                   # DPO preference pairs
│   ├── identity/              # Conversational/identity examples
│   └── docs_processed/        # Processed documentation text
├── nexifuse_model_adapter/    # Trained LoRA adapter weights
├── outputs/                   # GGUF files, checkpoints, Modelfile
├── tests/                     # Test suite
├── config.yaml                # Pipeline configuration
├── Upgrade_Plan_2026_3_11.md  # Dataset strategy & teacher model plan
└── ROADMAP.md                 # 7-phase project roadmap
```

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- [Ollama](https://ollama.com/) for teacher model and inference serving

### Installation

```bash
# Clone the repository
git clone https://github.com/aleriado/nexifuse-robust-expert.git
cd nexifuse-robust-expert

# Create virtual environment
python -m venv nexifuse_env
source nexifuse_env/bin/activate

# Install dependencies
pip install unsloth torch torchvision torchaudio
pip install transformers datasets peft accelerate bitsandbytes
pip install trl fastapi uvicorn httpx pydantic
```

### Run the Full Pipeline

```bash
# 1. Pull teacher models (recommended: both for optimal quality)
ollama pull deepseek-r1:70b        # Complex reasoning, multi-turn
ollama pull qwen2.5-coder:32b     # Bulk generation, general data

# 2. Generate training data
python -m nexifuse pipeline --num-per-domain 1500

# 3. Train on all available GPUs
python -m nexifuse train-multigpu

# 4. Export and deploy
python -m nexifuse convert
python -m nexifuse modelfile
python -m nexifuse register
python -m nexifuse serve
```

The inference server is now live at `http://localhost:8080` with an OpenAI-compatible API.

### Test the Model

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nexifuse-robust-expert",
    "messages": [{"role": "user", "content": "Write a Mirth Connect transformer that extracts patient name from HL7 ADT PID segment"}],
    "temperature": 0.1
  }'
```

## Dataset Strategy

The core thesis: **a well-curated 25k-example dataset with the right mixture will outperform a 100k-example dataset with the wrong mixture on an 8B model.**

### Target Composition (v1: 25,000 examples)

| Category | % of Dataset | Count | Purpose |
|----------|-------------|-------|---------|
| **Healthcare domain** (single-turn) | 40-45% | 11,000 | Core value: Mirth XML, HL7, FHIR, EHR APIs |
| **General assistant** (single-turn) | 25-30% | 7,000 | Prevents catastrophic forgetting (math, coding, reasoning) |
| **Multi-turn conversations** | 15-20% | 4,500 | Debugging, clarification, iterative building |
| **Identity & behavioral anchors** | 3-5% | 1,000 | NexiFuse persona, safety boundaries |
| **DPO preference pairs** | 5% | 1,500 | Alignment via chosen/rejected pairs |

### Teacher Model Stack (100% Local)

| Teacher | VRAM | Role | Speed |
|---------|------|------|-------|
| **DeepSeek-R1 70B** (Q4_K_M) | ~40 GB | Complex healthcare code, multi-turn, DPO chosen | 2-5 min/example |
| **Qwen 2.5 Coder 32B** (Q4_K_M) | ~18 GB | Bulk generation, general data, simple domain tasks | 20-60 sec/example |
| **Student (8B)** | ~6 GB | DPO rejected responses (self-play) | Very fast |

Both teachers run simultaneously on DGX Spark (128 GB) via Ollama. Total cost: $0.

### Healthcare Domain Breakdown

| Sub-Category | Count | Priority |
|-------------|-------|----------|
| Mirth Connect channel XML generation | 2,000 | P0 |
| Rhino JavaScript transformers | 2,000 | P0 |
| HL7 v2 message parsing & transformation | 1,500 | P0 |
| HL7 v2 to FHIR R4 conversion | 1,500 | P0 |
| FHIR R4 resource creation & bundles | 1,200 | P1 |
| EHR vendor API integration (Epic, Cerner, Athena) | 1,200 | P1 |
| Error handling & validation patterns | 800 | P1 |
| Security, PHI-safe logging, compliance | 500 | P2 |
| IHE profiles & DICOM | 300 | P2 |

### General Assistant Categories

| Category | Count |
|----------|-------|
| Math & arithmetic | 1,200 |
| General coding (Python, JS, SQL) | 1,500 |
| CS & technical Q&A | 1,200 |
| Reasoning & comparison | 1,000 |
| Casual conversation | 800 |
| Summarization & explanation | 300 |

### Multi-Turn Conversation Scenarios

| Scenario | Examples |
|----------|----------|
| Debugging conversations | 1,200 |
| Clarification dialogues | 900 |
| Iterative code building | 900 |
| Code review & improvement | 600 |
| Migration guidance | 500 |
| Step-by-step walkthroughs | 400 |

## Configuration

All pipeline settings are in `config.yaml`:

```yaml
training:
  base_model: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  lora_rank: 32
  lora_alpha: 64
  lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  batch_size: 1
  gradient_accumulation: 4
  learning_rate: 0.0002
  lr_scheduler: "cosine"
  num_epochs: 3
  max_seq_length: 2048
  quantization: "nf4"

data_factory:
  model_name: "llama3:70b"           # Primary teacher model
  domains: [hl7v2, fhir_r4, mirth, ehr_api, ihe, dicom]
  general_categories: [math, general_knowledge, casual, general_coding, reasoning]
  num_per_general_category: 1500
  conversation_scenarios: [debugging, clarification, iterative, code_review, migration, walkthrough]
  num_per_scenario_domain: 70

inference:
  model_name: "nexifuse-robust-expert"
  backend: "ollama"
  port: 8080
```

## Data Pipeline

The pipeline processes data through 6 stages, with auto-detection of all raw JSONL files:

| Stage | Command | Description | Current Count |
|-------|---------|-------------|---------------|
| Ingest | `nexifuse ingest` | Extract text from PDFs, HTML, API specs | — |
| Scrape | `nexifuse scrape` | Clone GitHub repos, extract code examples | 547 |
| Generate | `nexifuse generate` | Healthcare domain examples via teacher model | 12,961 |
| Generate | `nexifuse generate-general` | General assistant examples (5 categories) | 7,500 |
| Generate | `nexifuse generate-conversations` | Multi-turn conversations (6 scenarios) | 1,116 |
| Clean | `nexifuse clean` | Dedup, normalize, filter identity leakage | 18,055 |
| Validate | `nexifuse validate` | JS/XML/HL7/FHIR syntax + security scan | 17,661 passed |
| Format | `nexifuse format` | Chat-template with system prompt + identity | 35,394 |

### Validation Engine

The validator checks training example outputs against multiple format-specific rules:

- **JavaScript**: Bracket/brace matching (ESLint when configured)
- **XML**: Well-formedness via `xml.etree`
- **HL7 v2**: MSH header, required segments per message type (ADT, ORU, ORM, SIU, VXU)
- **FHIR R4**: JSON structure, `resourceType` field, optional JSON Schema validation
- **Security**: SQL injection detection, context-aware allowlist for placeholder credentials/PHI

## Training

### Single GPU

```bash
python -m nexifuse train
```

### Multi-GPU (Recommended)

Uses Hugging Face Accelerate with DDP for distributed training across all visible GPUs:

```bash
python -m nexifuse train-multigpu
```

### Training Results

#### MVP (Complete, deployed)

| Parameter | Value |
|-----------|-------|
| Dataset | 9,302 examples (healthcare domain only) |
| Max Seq Length | 2048 |
| Effective Batch Size | 32 (1 × 4 grad_accum × 8 GPUs) |
| Epochs | 5 |
| Training Time | ~2 hours (8x NVIDIA L4) |
| Final Loss | 0.2256 |
| GGUF Export | Q4_K_M, 4.6 GB |
| Status | **Deployed**, serving via Ollama on port 8080 |

#### v1 (Complete, balanced dataset)

| Parameter | Value |
|-----------|-------|
| Dataset | 35,394 examples (healthcare + general + multi-turn + identity) |
| Max Seq Length | 2048 |
| Effective Batch Size | 32 (1 × 4 grad_accum × 8 GPUs) |
| Epochs | 3 |
| GGUF Export | Q4_K_M, 4.6 GB |
| Status | **Complete**, trained, exported, and deployed |

#### Common Training Parameters

| Parameter | Value |
|-----------|-------|
| Base Model | DeepSeek-R1-Distill-Llama-8B |
| Quantization | 4-bit NF4 |
| LoRA Rank / Alpha | 32 / 64 |
| Target Modules | q, k, v, o, gate, up, down proj |
| Learning Rate | 2e-4 (cosine decay) |
| Trainable Parameters | 83.9M / 8.1B (1.03%) |

### Next Steps (V4)

1. ORPO preference alignment with 5,000 rejection sampling pairs
2. Error handling enforcement with 5,000 new examples where every code block has try/catch
3. Debug with code: 3,000 examples with diagnostic commands and fix code
4. Scale to 100k+ examples with full vendor coverage (Epic, Cerner, Athena, MEDITECH, Allscripts)
5. MedGemma 27B base model upgrade (built-in FHIR comprehension)
6. Extend context window to 8192+ tokens
7. RAG integration for real-time documentation grounding

## Model Export & Deployment

```bash
# Convert LoRA adapter to GGUF (Q4_K_M quantization)
python -m nexifuse convert

# Generate Ollama Modelfile with Llama 3 chat template
python -m nexifuse modelfile

# Register with Ollama
python -m nexifuse register --name nexifuse-robust-expert

# Start OpenAI-compatible inference server
python -m nexifuse serve
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |

## Integrator Desktop App

The Integrator is a Tauri 2 + React desktop app that connects to the inference server for "vibe coding" healthcare integrations.

```bash
cd integrator
cp .env.example .env    # Set VITE_AGENT_URL=http://localhost:8080
npm install
npm run tauri dev       # With display
npm run dev:headless    # Headless (xvfb)
```

### Settings

| Field | Value |
|-------|-------|
| URL | `http://<server-ip>:8080` |
| Model | `nexifuse-robust-expert` |
| API key | *(leave empty)* |
| Timeout | `60000` |

## Pipeline Commands Reference

All commands are run with `python -m nexifuse <command>`. Global flags: `-v` (verbose), `-c <path>` (config file, default: `config.yaml`).

### Step 1: Data Acquisition

#### `ingest`: Extract text from documentation

```bash
python -m nexifuse ingest --docs-dir docs --output-dir data/docs_processed
```

Reads PDFs, HTML, and text files from `docs/` (organized by domain: hl7v2, fhir_r4, mirth, ehr_api, ihe, dicom). Outputs cleaned text files to `data/docs_processed/` for use as context during synthetic data generation.

#### `scrape`: Clone GitHub repos and extract code examples

```bash
python -m nexifuse scrape -o data/raw/scraped.jsonl --repos-dir data/repos
python -m nexifuse scrape --no-teacher   # Skip teacher model instruction synthesis
```

Clones repos defined in `config.yaml` → `scraper.repos` (e.g., Mirth Connect examples), extracts code matching `file_patterns` (*.js, *.xml, *.json), scrubs PHI via regex, and optionally uses the teacher model to generate instruction-output pairs. Output: `data/raw/scraped.jsonl`.

#### `generate`: Synthetic healthcare domain examples

```bash
python -m nexifuse generate --num-per-domain 1500 -w 8 -o data/raw/synthetic.jsonl
```

Uses the teacher model (configured in `config.yaml` → `data_factory.model_name`) to generate instruction-output pairs for each domain (hl7v2, fhir_r4, mirth, ehr_api, ihe, dicom). Supports resume. If the output file exists, it counts existing examples per domain and continues from where it left off. Output: `data/raw/synthetic.jsonl`.

| Flag | Default | Description |
|------|---------|-------------|
| `--num-per-domain` | 500 | Examples per domain (6 domains × N) |
| `-w, --num-workers` | 8 | Parallel generation threads |
| `-o, --output` | `data/raw/synthetic.jsonl` | Output path |

#### `generate-general`: General assistant examples

```bash
python -m nexifuse generate-general --num-per-category 1500 -w 8 -o data/raw/general.jsonl
```

Generates examples across 5 categories defined in `config.yaml` → `general_categories`: math, general_knowledge, casual, general_coding, reasoning. Prevents catastrophic forgetting of base model capabilities. Output: `data/raw/general.jsonl`.

#### `generate-conversations`: Multi-turn conversation examples

```bash
python -m nexifuse generate-conversations --num-per-scenario-domain 70 -w 8 -o data/raw/conversations.jsonl
```

Generates multi-turn conversations (3-8 turns each) across 6 scenarios × 6 domains defined in `config.yaml` → `conversation_scenarios`: debugging, clarification, iterative, code_review, migration, walkthrough. Output: `data/raw/conversations.jsonl`.

### Step 2: Data Processing

#### `clean`: Deduplicate, normalize, and filter

```bash
python -m nexifuse clean                                    # Auto-detect all data/raw/*.jsonl
python -m nexifuse clean -i data/raw/synthetic.jsonl data/raw/general.jsonl  # Specific files
python -m nexifuse clean --threshold 0.85                   # Adjust dedup similarity threshold
```

Auto-detects all `data/raw/*.jsonl` files (or accepts explicit `-i` paths). Runs 4 stages: dedup by cosine similarity, normalization, identity/persona filtering, and output writing. Output: `data/cleaned/cleaned.jsonl`.

#### `validate`: Multi-format syntax + security validation

```bash
python -m nexifuse validate -i data/cleaned/cleaned.jsonl
```

Validates each example's output against format-specific rules (JavaScript bracket matching, XML well-formedness, HL7 v2 segment structure, FHIR R4 JSON schema, SQL injection detection). Splits into passed and failed sets. Output: `data/validated/passed.jsonl` + `data/validated/failed.jsonl`.

#### `dpo`: Generate DPO preference pairs

```bash
python -m nexifuse dpo --passed data/validated/passed.jsonl --failed data/validated/failed.jsonl -o data/dpo/dpo_pairs.jsonl
```

Creates chosen/rejected preference pairs from validated pass/fail splits for Direct Preference Optimization alignment training. Output: `data/dpo/dpo_pairs.jsonl`.

#### `format`: Apply chat template for training

```bash
python -m nexifuse format -i data/validated/passed.jsonl -o data/formatted/train.jsonl --template llama
python -m nexifuse format --identity data/identity/conversational.jsonl --conversations data/raw/conversations.jsonl
```

Wraps each example in Llama 3 (or ChatML) chat template with system prompt and NexiFuse identity anchors. Merges single-turn, multi-turn conversations, and identity examples into one training file. Output: `data/formatted/train.jsonl`.

### Step 3: Training

#### `train`: Single-GPU SFT fine-tuning

```bash
python -m nexifuse train -i data/formatted/train.jsonl
```

Runs LoRA SFT fine-tuning with Unsloth on one GPU. Uses settings from `config.yaml` → `training` (base model, LoRA rank/alpha, learning rate, epochs, etc.). Output: LoRA adapter in `nexifuse_model_adapter/`.

#### `train-multigpu`: Multi-GPU distributed training (recommended)

```bash
python -m nexifuse train-multigpu -i data/formatted/train.jsonl
python -m nexifuse train-multigpu -n 4   # Use only 4 GPUs
```

Launches training via Hugging Face Accelerate DDP across all visible GPUs. Automatically detects GPU count (override with `-n`). Effective batch size = `batch_size × gradient_accumulation × num_gpus`. Output: LoRA adapter in `nexifuse_model_adapter/`.

#### `train-dpo`: DPO alignment (after SFT)

```bash
python -m nexifuse train-dpo -i data/dpo/dpo_pairs.jsonl --adapter nexifuse_model_adapter
```

Runs Direct Preference Optimization on DPO pairs using the SFT adapter as starting point. Output: Updated adapter in `nexifuse_model_adapter/`.

### Step 4: Export & Deployment

#### `merge`: Merge LoRA adapter into base model

```bash
python -m nexifuse merge --adapter nexifuse_model_adapter -o outputs/merged_model
```

Merges the LoRA adapter weights into the full base model. Required if using llama.cpp for manual GGUF conversion. Output: `outputs/merged_model/`.

#### `convert`: LoRA → GGUF conversion

```bash
python -m nexifuse convert --adapter nexifuse_model_adapter -o outputs --quant q4_k_m
```

Converts the LoRA adapter directly to GGUF format via Unsloth (or falls back to llama.cpp). Quantization options: `q4_k_m` (4.6 GB, recommended), `q5_k_m`, `q8_0`, `f16`. Output: `outputs/nexifuse-q4km.gguf`.

#### `modelfile`: Generate Ollama Modelfile

```bash
python -m nexifuse modelfile --gguf outputs/nexifuse-q4km.gguf -o outputs/Modelfile
```

Generates an Ollama Modelfile with the Llama 3 chat template, system prompt, and inference parameters. Output: `outputs/Modelfile`.

#### `register`: Register model with Ollama

```bash
python -m nexifuse register --modelfile outputs/Modelfile --name nexifuse-robust-expert
```

Runs `ollama create` to register the GGUF model. After this, `ollama list` will show `nexifuse-robust-expert`.

#### `serve`: Start OpenAI-compatible inference server

```bash
python -m nexifuse serve
```

Starts a FastAPI server (default `0.0.0.0:8080`) that proxies to Ollama with an OpenAI-compatible API. Endpoints: `/health`, `/v1/models`, `/v1/chat/completions` (streaming supported). Configure host/port in `config.yaml` → `inference`.

### Shortcuts

#### `pipeline`: Run full data pipeline in one command

```bash
python -m nexifuse pipeline --num-per-domain 1500 -w 8
```

Runs all 6 stages sequentially: ingest → scrape → generate (domain + general + conversations) → clean → validate → format. The recommended way to build the full dataset from scratch.

#### `pipeline-20k`: Target 20k+ cleaned examples

```bash
python -m nexifuse pipeline-20k -w 8
```

Same as `pipeline` but with `--num-per-domain 6000` and `--no-teacher` for scraping (faster). Targets 20k+ cleaned examples after dedup and validation.

### Data Flow Summary

```
docs/                  →  ingest   →  data/docs_processed/
GitHub repos           →  scrape   →  data/raw/scraped.jsonl
Teacher model (domain) →  generate →  data/raw/synthetic.jsonl
Teacher model (general)→  generate-general      →  data/raw/general.jsonl
Teacher model (conv.)  →  generate-conversations →  data/raw/conversations.jsonl

data/raw/*.jsonl       →  clean    →  data/cleaned/cleaned.jsonl
data/cleaned/          →  validate →  data/validated/{passed,failed}.jsonl
data/validated/        →  dpo      →  data/dpo/dpo_pairs.jsonl
data/validated/passed  →  format   →  data/formatted/train.jsonl

data/formatted/train   →  train / train-multigpu →  nexifuse_model_adapter/
nexifuse_model_adapter →  convert  →  outputs/nexifuse-q4km.gguf
outputs/*.gguf         →  modelfile → outputs/Modelfile
outputs/Modelfile      →  register →  Ollama model registry
Ollama                 →  serve    →  http://0.0.0.0:8080
```

## Tests

```bash
pytest tests/ -v
```

## License

This project is proprietary. See [ROADMAP.md](ROADMAP.md) for the full technical roadmap.
