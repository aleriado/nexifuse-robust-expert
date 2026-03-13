# NexiFuse Health — Robust Expert

A domain-specific AI model for healthcare data interoperability, fine-tuned on **DeepSeek-R1-Distill-Llama-8B** using LoRA (Unsloth). Translates natural language into production-grade **Mirth Connect**, **HL7 v2**, **FHIR R4**, and **EHR API** integration code — while retaining general assistant capabilities (math, reasoning, casual conversation, general coding).

Designed for fully on-premise deployment. Zero API costs. Zero data leaves the premises. The trained model is quantized to GGUF Q4_K_M (4.6 GB), served locally via Ollama with an OpenAI-compatible API, and consumed by the **Integrator** desktop app (Tauri 2 + React).

## Highlights

- **MVP deployed, v1 training in progress** — following the [Upgrade Plan](Upgrade_Plan_2026_3_11.md) to build a balanced Robust Expert
- **22k+ raw examples generated** — 57% healthcare, 34% general assistant, 5% multi-turn, 2.5% scraped code
- **Dual teacher model stack** — Llama 3 70B (complex reasoning) + Llama 3 8B (bulk generation), both running locally via Ollama
- **Multi-GPU distributed training** via Accelerate DDP — 8x NVIDIA L4 cluster
- **97.8% validation pass rate** with multi-format validation (JavaScript, XML, HL7 v2, FHIR R4, security scanning)
- **End-to-end CLI pipeline** — from data ingestion to model serving in one tool
- **OpenAI-compatible API** — drop-in replacement for any OpenAI client
- **100% local, 100% free** — all training, generation, and inference on-premise

## Current Status

Following the [Upgrade Plan](Upgrade_Plan_2026_3_11.md) to evolve from MVP to production-ready v1.

### Milestones

| Milestone | Status | Details |
|-----------|--------|---------|
| **MVP** (8-12k examples) | COMPLETE | 9.3k healthcare-only dataset. Model deployed via Ollama. Proved pipeline works end-to-end. |
| **v1** (20-30k examples) | IN PROGRESS | 22.1k raw generated → 18k cleaned → 17.6k validated. Training on balanced dataset underway. |
| **Production** (50-80k examples) | PLANNED | Full vendor coverage, DPO alignment, edge-case hardening. |

### v1 Progress (per [Upgrade Plan](Upgrade_Plan_2026_3_11.md))

| Phase | Status | Details |
|-------|--------|---------|
| Data Generation | COMPLETE | 22,124 raw examples across all categories |
| Data Cleaning | COMPLETE | 18,055 examples after dedup + normalization |
| Validation | COMPLETE | 17,661 passed (97.8% pass rate) |
| Formatting | COMPLETE | 35,394 training examples (with identity anchors) |
| v1 Training | IN PROGRESS | Step 3,000/13,275 (22.6%), loss 0.205 |
| v1 Export | PENDING | Re-export GGUF Q4_K_M after training completes |
| DPO Alignment | PENDING | Generate preference pairs, then train-dpo |

### Dataset Composition (v1)

| Source | File | Count | % of Raw |
|--------|------|-------|----------|
| Healthcare domain (synthetic) | `synthetic_run1.jsonl` | 12,600 | 57% |
| General assistant (5 categories) | `general.jsonl` | 7,500 | 34% |
| Multi-turn conversations (6 scenarios) | `conversations.jsonl` | 1,116 | 5% |
| GitHub scraped code | `scraped.jsonl` | 547 | 2.5% |
| Domain synthetic (early run) | `synthetic.jsonl` | 361 | 1.5% |
| **Total raw** | | **22,124** | |

| Processing Stage | File | Count |
|-----------------|------|-------|
| After cleaning | `cleaned.jsonl` | 18,055 |
| After validation (passed) | `passed.jsonl` | 17,661 |
| After validation (failed) | `failed.jsonl` | 394 |
| After formatting (with identity + conversations) | `train.jsonl` | 35,394 |

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
  num_epochs: 5
  max_seq_length: 4096          # Increased from 2048 for multi-turn + full XML outputs
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

- **JavaScript** — Bracket/brace matching (ESLint when configured)
- **XML** — Well-formedness via `xml.etree`
- **HL7 v2** — MSH header, required segments per message type (ADT, ORU, ORM, SIU, VXU)
- **FHIR R4** — JSON structure, `resourceType` field, optional JSON Schema validation
- **Security** — SQL injection detection, context-aware allowlist for placeholder credentials/PHI

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

#### MVP (Complete — deployed)

| Parameter | Value |
|-----------|-------|
| Dataset | 9,302 examples (healthcare domain only) |
| Max Seq Length | 2048 |
| Effective Batch Size | 32 (1 × 4 grad_accum × 8 GPUs) |
| Epochs | 5 |
| Training Time | ~2 hours (8x NVIDIA L4) |
| Final Loss | 0.2256 |
| GGUF Export | Q4_K_M — 4.6 GB |
| Status | **Deployed** — serving via Ollama on port 8080 |

#### v1 (In progress — balanced dataset)

| Parameter | Value |
|-----------|-------|
| Dataset | 35,394 examples (healthcare + general + multi-turn + identity) |
| Max Seq Length | 4096 |
| Effective Batch Size | 4 (1 × 4 grad_accum × 1 GPU) |
| Epochs | 3 |
| Total Steps | 13,275 |
| Current Progress | Step 3,000 (22.6%), loss 0.205 |
| Status | **In progress** — to be resumed |

#### Common Training Parameters

| Parameter | Value |
|-----------|-------|
| Base Model | DeepSeek-R1-Distill-Llama-8B |
| Quantization | 4-bit NF4 |
| LoRA Rank / Alpha | 32 / 64 |
| Target Modules | q, k, v, o, gate, up, down proj |
| Learning Rate | 2e-4 (cosine decay) |
| Trainable Parameters | 83.9M / 8.1B (1.03%) |

### Remaining Steps (v1 → Production)

1. Resume v1 training to completion (remaining ~10k steps)
2. Re-export GGUF Q4_K_M from v1 adapter
3. Deploy v1 model and benchmark vs MVP (general capability + healthcare correctness)
4. Generate DPO preference pairs (student failures + DeepSeek-R1 70B chosen)
5. Run DPO alignment training
6. Scale toward production (50-80k examples) per [Upgrade Plan](Upgrade_Plan_2026_3_11.md)

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

#### `ingest` — Extract text from documentation

```bash
python -m nexifuse ingest --docs-dir docs --output-dir data/docs_processed
```

Reads PDFs, HTML, and text files from `docs/` (organized by domain: hl7v2, fhir_r4, mirth, ehr_api, ihe, dicom). Outputs cleaned text files to `data/docs_processed/` for use as context during synthetic data generation.

#### `scrape` — Clone GitHub repos and extract code examples

```bash
python -m nexifuse scrape -o data/raw/scraped.jsonl --repos-dir data/repos
python -m nexifuse scrape --no-teacher   # Skip teacher model instruction synthesis
```

Clones repos defined in `config.yaml` → `scraper.repos` (e.g., Mirth Connect examples), extracts code matching `file_patterns` (*.js, *.xml, *.json), scrubs PHI via regex, and optionally uses the teacher model to generate instruction-output pairs. Output: `data/raw/scraped.jsonl`.

#### `generate` — Synthetic healthcare domain examples

```bash
python -m nexifuse generate --num-per-domain 1500 -w 8 -o data/raw/synthetic.jsonl
```

Uses the teacher model (configured in `config.yaml` → `data_factory.model_name`) to generate instruction-output pairs for each domain (hl7v2, fhir_r4, mirth, ehr_api, ihe, dicom). Supports resume — if the output file exists, it counts existing examples per domain and continues from where it left off. Output: `data/raw/synthetic.jsonl`.

| Flag | Default | Description |
|------|---------|-------------|
| `--num-per-domain` | 500 | Examples per domain (6 domains × N) |
| `-w, --num-workers` | 8 | Parallel generation threads |
| `-o, --output` | `data/raw/synthetic.jsonl` | Output path |

#### `generate-general` — General assistant examples

```bash
python -m nexifuse generate-general --num-per-category 1500 -w 8 -o data/raw/general.jsonl
```

Generates examples across 5 categories defined in `config.yaml` → `general_categories`: math, general_knowledge, casual, general_coding, reasoning. Prevents catastrophic forgetting of base model capabilities. Output: `data/raw/general.jsonl`.

#### `generate-conversations` — Multi-turn conversation examples

```bash
python -m nexifuse generate-conversations --num-per-scenario-domain 70 -w 8 -o data/raw/conversations.jsonl
```

Generates multi-turn conversations (3-8 turns each) across 6 scenarios × 6 domains defined in `config.yaml` → `conversation_scenarios`: debugging, clarification, iterative, code_review, migration, walkthrough. Output: `data/raw/conversations.jsonl`.

### Step 2: Data Processing

#### `clean` — Deduplicate, normalize, and filter

```bash
python -m nexifuse clean                                    # Auto-detect all data/raw/*.jsonl
python -m nexifuse clean -i data/raw/synthetic.jsonl data/raw/general.jsonl  # Specific files
python -m nexifuse clean --threshold 0.85                   # Adjust dedup similarity threshold
```

Auto-detects all `data/raw/*.jsonl` files (or accepts explicit `-i` paths). Runs 4 stages: dedup by cosine similarity, normalization, identity/persona filtering, and output writing. Output: `data/cleaned/cleaned.jsonl`.

#### `validate` — Multi-format syntax + security validation

```bash
python -m nexifuse validate -i data/cleaned/cleaned.jsonl
```

Validates each example's output against format-specific rules (JavaScript bracket matching, XML well-formedness, HL7 v2 segment structure, FHIR R4 JSON schema, SQL injection detection). Splits into passed and failed sets. Output: `data/validated/passed.jsonl` + `data/validated/failed.jsonl`.

#### `dpo` — Generate DPO preference pairs

```bash
python -m nexifuse dpo --passed data/validated/passed.jsonl --failed data/validated/failed.jsonl -o data/dpo/dpo_pairs.jsonl
```

Creates chosen/rejected preference pairs from validated pass/fail splits for Direct Preference Optimization alignment training. Output: `data/dpo/dpo_pairs.jsonl`.

#### `format` — Apply chat template for training

```bash
python -m nexifuse format -i data/validated/passed.jsonl -o data/formatted/train.jsonl --template llama
python -m nexifuse format --identity data/identity/conversational.jsonl --conversations data/raw/conversations.jsonl
```

Wraps each example in Llama 3 (or ChatML) chat template with system prompt and NexiFuse identity anchors. Merges single-turn, multi-turn conversations, and identity examples into one training file. Output: `data/formatted/train.jsonl`.

### Step 3: Training

#### `train` — Single-GPU SFT fine-tuning

```bash
python -m nexifuse train -i data/formatted/train.jsonl
```

Runs LoRA SFT fine-tuning with Unsloth on one GPU. Uses settings from `config.yaml` → `training` (base model, LoRA rank/alpha, learning rate, epochs, etc.). Output: LoRA adapter in `nexifuse_model_adapter/`.

#### `train-multigpu` — Multi-GPU distributed training (recommended)

```bash
python -m nexifuse train-multigpu -i data/formatted/train.jsonl
python -m nexifuse train-multigpu -n 4   # Use only 4 GPUs
```

Launches training via Hugging Face Accelerate DDP across all visible GPUs. Automatically detects GPU count (override with `-n`). Effective batch size = `batch_size × gradient_accumulation × num_gpus`. Output: LoRA adapter in `nexifuse_model_adapter/`.

#### `train-dpo` — DPO alignment (after SFT)

```bash
python -m nexifuse train-dpo -i data/dpo/dpo_pairs.jsonl --adapter nexifuse_model_adapter
```

Runs Direct Preference Optimization on DPO pairs using the SFT adapter as starting point. Output: Updated adapter in `nexifuse_model_adapter/`.

### Step 4: Export & Deployment

#### `merge` — Merge LoRA adapter into base model

```bash
python -m nexifuse merge --adapter nexifuse_model_adapter -o outputs/merged_model
```

Merges the LoRA adapter weights into the full base model. Required if using llama.cpp for manual GGUF conversion. Output: `outputs/merged_model/`.

#### `convert` — LoRA → GGUF conversion

```bash
python -m nexifuse convert --adapter nexifuse_model_adapter -o outputs --quant q4_k_m
```

Converts the LoRA adapter directly to GGUF format via Unsloth (or falls back to llama.cpp). Quantization options: `q4_k_m` (4.6 GB, recommended), `q5_k_m`, `q8_0`, `f16`. Output: `outputs/nexifuse-q4km.gguf`.

#### `modelfile` — Generate Ollama Modelfile

```bash
python -m nexifuse modelfile --gguf outputs/nexifuse-q4km.gguf -o outputs/Modelfile
```

Generates an Ollama Modelfile with the Llama 3 chat template, system prompt, and inference parameters. Output: `outputs/Modelfile`.

#### `register` — Register model with Ollama

```bash
python -m nexifuse register --modelfile outputs/Modelfile --name nexifuse-robust-expert
```

Runs `ollama create` to register the GGUF model. After this, `ollama list` will show `nexifuse-robust-expert`.

#### `serve` — Start OpenAI-compatible inference server

```bash
python -m nexifuse serve
```

Starts a FastAPI server (default `0.0.0.0:8080`) that proxies to Ollama with an OpenAI-compatible API. Endpoints: `/health`, `/v1/models`, `/v1/chat/completions` (streaming supported). Configure host/port in `config.yaml` → `inference`.

### Shortcuts

#### `pipeline` — Run full data pipeline in one command

```bash
python -m nexifuse pipeline --num-per-domain 1500 -w 8
```

Runs all 6 stages sequentially: ingest → scrape → generate (domain + general + conversations) → clean → validate → format. The recommended way to build the full dataset from scratch.

#### `pipeline-20k` — Target 20k+ cleaned examples

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

This project is proprietary. See [ROADMAP.md](ROADMAP.md) for the full technical roadmap and [Upgrade_Plan_2026_3_11.md](Upgrade_Plan_2026_3_11.md) for the detailed dataset strategy.
