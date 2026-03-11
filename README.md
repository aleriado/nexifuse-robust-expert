# NexiFuse Health — Robust Expert

A domain-specific AI model for healthcare data interoperability, fine-tuned on **DeepSeek-R1-Distill-Llama-8B** using LoRA (Unsloth). Translates natural language into production-grade **Mirth Connect**, **HL7 v2**, **FHIR R4**, and **EHR API** integration code.

Designed for on-premise deployment on NVIDIA hardware. The trained model is quantized to GGUF Q4_K_M (4.6 GB), served locally via Ollama with an OpenAI-compatible API, and consumed by the **Integrator** desktop app (Tauri 2 + React).

## Highlights

- **9,302 training examples** across 6 healthcare domains — synthetic + scraped + conversational
- **8-GPU distributed training** on NVIDIA L4 cluster via Accelerate DDP — 2 hours to convergence
- **97% validation pass rate** with multi-format validation (JavaScript, XML, HL7 v2, FHIR R4, security)
- **End-to-end CLI pipeline** — from data ingestion to model serving in one tool
- **OpenAI-compatible API** — drop-in replacement for any OpenAI client

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA ACQUISITION                        │
│                                                             │
│   GitHub Scraper    Doc Ingestion     Teacher-Student       │
│   (repos, code)     (PDFs, HTML)      Data Factory          │
│         │                │            (Ollama teacher)      │
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
# 1. Generate training data (requires Ollama with teacher model)
ollama pull qwen2.5-coder:7b
python -m nexifuse pipeline --num-per-domain 1500

# 2. Train on all available GPUs
python -m nexifuse train-multigpu

# 3. Export and deploy
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
  max_seq_length: 2048
  quantization: "nf4"

data_factory:
  model_name: "qwen2.5-coder:7b"    # Teacher model via Ollama
  domains: [hl7v2, fhir_r4, mirth, ehr_api, ihe, dicom]

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

### Domains

| Domain | Description | Examples |
|--------|-------------|----------|
| `hl7v2` | HL7 v2.x message parsing, segment manipulation, ADT/ORU/ORM | 1,500 |
| `fhir_r4` | FHIR R4 resource creation, Bundle operations, search params | 1,500 |
| `mirth` | Mirth Connect channels, transformers, filters, deployment | 1,500 |
| `ehr_api` | Epic FHIR, Cerner, Allscripts API integrations | 1,500 |
| `ihe` | IHE profiles (XDS, PIX, PDQ), cross-enterprise workflows | 1,500 |
| `dicom` | DICOM networking, C-STORE/C-FIND, worklist management | 1,500 |

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
npm run tauri dev
```

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

This project is proprietary. See [ROADMAP.md](ROADMAP.md) for the full technical roadmap.
