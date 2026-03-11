# NexiFuse Health — Robust Expert

A domain-specific AI model for healthcare data interoperability, fine-tuned on **DeepSeek-R1-Distill-Llama-8B** using LoRA (Unsloth). Translates natural language into production-grade **Mirth Connect**, **HL7 v2**, **FHIR R4**, and **EHR API** integration code — while retaining general assistant capabilities (math, reasoning, casual conversation, general coding).

Designed for fully on-premise deployment. Zero API costs. Zero data leaves the premises. The trained model is quantized to GGUF Q4_K_M (4.6 GB), served locally via Ollama with an OpenAI-compatible API, and consumed by the **Integrator** desktop app (Tauri 2 + React).

## Highlights

- **25k target dataset** with balanced mixture: 40-45% healthcare domain, 25-30% general assistant, 15-20% multi-turn conversations, 3-5% identity anchors
- **Dual teacher model stack** — DeepSeek-R1 70B (complex reasoning) + Qwen 2.5 Coder 32B (bulk generation), both running locally via Ollama
- **Multi-GPU distributed training** via Accelerate DDP — 8x NVIDIA L4 cluster
- **97% validation pass rate** with multi-format validation (JavaScript, XML, HL7 v2, FHIR R4, security scanning)
- **End-to-end CLI pipeline** — from data ingestion to model serving in one tool
- **OpenAI-compatible API** — drop-in replacement for any OpenAI client
- **100% local, 100% free** — all training, generation, and inference on-premise

## Current Status

| Phase | Status | Details |
|-------|--------|---------|
| Phase 0: Environment Setup | COMPLETE | DGX Spark / GCP L4 cluster ready |
| Phase 1: Documentation Corpus | COMPLETE | docs/ organized by domain |
| Phase 2: Data Generation | IN PROGRESS | 9.3k healthcare examples done; general + multi-turn data planned (v1 target: 25k) |
| Phase 3: Data Processing | COMPLETE | Pipeline works: clean → validate → format |
| Phase 4: Model Training (MVP) | COMPLETE | 8B model trained on 9.3k examples, loss 0.2256 |
| Phase 5: Model Export | COMPLETE | GGUF Q4_K_M (4.6 GB) exported |
| Phase 6: Deployment | COMPLETE | Ollama + FastAPI server + Integrator app |
| Phase 7: Iterative Improvement | IN PROGRESS | Expanding dataset with general + multi-turn data |

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
  model_name: "qwen2.5-coder:7b"    # Upgrade to qwen2.5-coder:32b recommended
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
| Scrape | `nexifuse scrape` | Clone GitHub repos, extract code examples | ~500 |
| Generate | `nexifuse generate` | Synthetic examples via teacher model | 9,000 |
| Clean | `nexifuse clean` | Dedup, normalize, filter identity leakage | 9,504 |
| Validate | `nexifuse validate` | JS/XML/HL7/FHIR syntax + security scan | 9,230 passed |
| Format | `nexifuse format` | Chat-template with system prompt + identity | 9,302 |

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

### MVP Training Results (9.3k healthcare-only dataset)

| Parameter | Value |
|-----------|-------|
| Base Model | DeepSeek-R1-Distill-Llama-8B |
| Quantization | 4-bit NF4 |
| LoRA Rank / Alpha | 32 / 64 |
| Target Modules | q, k, v, o, gate, up, down proj |
| Learning Rate | 2e-4 (cosine decay) |
| Effective Batch Size | 32 (1 x 4 grad_accum x 8 GPUs) |
| Epochs | 5 |
| Training Time | ~2 hours (8x NVIDIA L4) |
| Final Loss | 0.2256 |
| Trainable Parameters | 83.9M / 8.1B (1.03%) |
| GGUF Export | Q4_K_M — 4.6 GB |

### Next Training Run (v1: 25k balanced dataset)

- Increase `max_seq_length` to 4096 (already updated in config)
- Add general assistant data (7k examples) to prevent catastrophic forgetting
- Add multi-turn conversations (4.5k examples) for interactive debugging
- Upgrade teacher model from Qwen 7B to Qwen 32B + DeepSeek-R1 70B
- Add DPO alignment pass after SFT

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

## CLI Reference

```
nexifuse <command> [options]

Data Pipeline:
  ingest           Ingest documentation from docs/
  scrape           Scrape GitHub repos for training data
  generate         Generate synthetic examples via teacher model
  clean            Clean and deduplicate raw data
  validate         Run multi-format validation
  dpo              Generate DPO preference pairs
  format           Format data into chat templates

Training:
  train            SFT fine-tuning (single GPU)
  train-multigpu   SFT on all GPUs via Accelerate DDP
  train-dpo        DPO alignment training

Export & Deploy:
  merge            Merge LoRA adapter into full model
  convert          Convert LoRA to GGUF format
  quantize         Quantize model (alias for convert)
  modelfile        Generate Ollama Modelfile
  register         Register model with Ollama
  serve            Start inference server

Shortcuts:
  pipeline         Run full data pipeline (ingest → format)
  pipeline-20k     Run pipeline targeting 20k+ examples

Global Options:
  -v, --verbose    Enable debug logging
  -c, --config     Config file path (default: config.yaml)
```

## Tests

```bash
pytest tests/ -v
```

## License

This project is proprietary. See [ROADMAP.md](ROADMAP.md) for the full technical roadmap and [Upgrade_Plan_2026_3_11.md](Upgrade_Plan_2026_3_11.md) for the detailed dataset strategy.
