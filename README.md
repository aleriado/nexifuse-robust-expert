# NexiFuse Health — Robust Expert

A domain-specific AI model for healthcare data interoperability, fine-tuned on **DeepSeek-R1-Distill-Llama-8B** using LoRA. Translates natural language into production-grade **Mirth Connect**, **HL7 v2**, **FHIR R4**, and **EHR API** integration code.

Designed for on-premise deployment on NVIDIA DGX Spark (GB10 Grace Blackwell, 128GB unified memory). The trained model is served locally via an OpenAI-compatible API and consumed by the **Integrator** desktop app (Tauri 2 + React).

## Architecture

```
Documentation + GitHub Repos + Teacher Model
                │
                ▼
     Raw JSONL (scraped + synthetic)
                │
     Clean → Validate → DPO Pairs
                │
                ▼
     SFT Fine-Tuning (Unsloth + LoRA)
                │
     Optional DPO Alignment
                │
     Merge LoRA → GGUF Conversion
                │
                ▼
     Ollama / vLLM Inference Server
                │
                ▼
     Integrator Desktop App
```

## Project Structure

```
├── nexifuse/                  # Core Python package (all pipeline modules)
│   ├── cli.py                 # CLI entry points
│   ├── config.py              # Configuration management
│   ├── scraper.py             # GitHub corpus scraper
│   ├── doc_ingester.py        # Documentation ingestion (PDF/HTML)
│   ├── data_factory.py        # Teacher-student synthetic data generation
│   ├── data_cleaner.py        # Dedup, normalization, cleaning
│   ├── validator.py           # Multi-format validation (JS, XML, HL7, FHIR)
│   ├── dpo_generator.py       # DPO preference pair generation
│   ├── prompt_formatter.py    # ChatML/Llama prompt templates
│   ├── training_pipeline.py   # Unsloth SFT fine-tuning
│   ├── gguf_converter.py      # LoRA merge + GGUF conversion
│   └── inference_server.py    # FastAPI OpenAI-compatible server
├── integrator/                # Integrator desktop app (Tauri 2 + React)
├── docs/                      # Raw documentation corpus by domain
├── data/                      # Training data (all stages)
│   ├── raw/                   # Scraped + synthetic JSONL
│   ├── cleaned/               # Post-cleaning JSONL
│   ├── validated/             # Post-validation (passed/failed)
│   ├── formatted/             # Chat-template formatted for training
│   ├── dpo/                   # DPO preference pairs
│   └── docs_processed/        # Processed documentation text
├── tests/                     # Tests
├── config.yaml                # Pipeline configuration
└── ROADMAP.md                 # Detailed project roadmap
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on DGX Spark GB10)
- [Ollama](https://ollama.com/) (for teacher model serving and final inference)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (for GGUF conversion, build with CUDA)
- ESLint + xmllint (for validation engine)

## Setup

```bash
# Create and activate virtual environment
python -m venv nexifuse_env
source nexifuse_env/bin/activate

# Install dependencies
pip install unsloth torch torchvision torchaudio
pip install transformers datasets peft accelerate bitsandbytes
pip install triton>=3.3.1  # Required for Blackwell kernels

# For Arm-based systems (DGX Spark Grace CPU)
export TORCH_CUDA_ARCH_LIST="12.0"
pip install xformers --no-build-isolation
```

## Configuration

All pipeline settings are in `config.yaml`:

```yaml
training:
  base_model: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  lora_rank: 16
  lora_alpha: 32
  batch_size: 1
  gradient_accumulation: 4
  learning_rate: 0.0002
  num_epochs: 3
  max_seq_length: 2048
  quantization: "nf4"

data_factory:
  model_name: "qwen2.5-coder:7b"       # Teacher model via Ollama
  endpoint: "http://localhost:11434/api/generate"
```

## Usage

The entire pipeline is driven through the `nexifuse` CLI.

### Full Data Pipeline (one command)

```bash
python -m nexifuse pipeline --num-per-domain 100
```

This runs: ingest → scrape → generate → clean → validate → format.

### Step-by-Step Pipeline

```bash
# 1. Ingest documentation into processed text
python -m nexifuse ingest

# 2. Scrape GitHub repos for training examples
python -m nexifuse scrape

# 3. Generate synthetic data via teacher model
python -m nexifuse generate --num-per-domain 100

# 4. Clean and deduplicate
python -m nexifuse clean

# 5. Validate outputs (JS, XML, HL7, FHIR, security)
python -m nexifuse validate

# 6. Generate DPO preference pairs from pass/fail results
python -m nexifuse dpo

# 7. Format into chat templates for training
python -m nexifuse format
```

### Training

```bash
# SFT fine-tuning with LoRA
python -m nexifuse train

# Optional: DPO alignment using preference pairs
python -m nexifuse train-dpo
```

### Model Export & Deployment

```bash
# Convert LoRA adapter to GGUF (default: q4_k_m quantization)
python -m nexifuse convert

# Generate Ollama Modelfile
python -m nexifuse modelfile

# Register with Ollama
python -m nexifuse register --name nexifuse-robust-expert

# Start inference server (OpenAI-compatible at :8080)
python -m nexifuse serve
```

The inference server exposes `POST /v1/chat/completions` — compatible with any OpenAI client.

### Integrator App

The Integrator desktop app connects to the inference server:

```bash
cd integrator
cp .env.example .env
# Set VITE_AGENT_URL=http://localhost:8080 in .env
npm install
npm run tauri dev
```

## Training Details

| Parameter | Value |
|---|---|
| Base Model | DeepSeek-R1-Distill-Llama-8B |
| Quantization | 4-bit NF4 |
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Target Modules | q, k, v, o, gate, up, down proj |
| Learning Rate | 2e-4 (cosine decay) |
| Batch Size | 1 (×4 gradient accumulation) |
| Epochs | 5 |
| Precision | BF16 mixed |

## Data Pipeline Overview

Target: **10k–20k cleaned + validated examples** for better model behavior. Use `--num-per-domain 500` (default) or `2000` for larger runs.

| Stage | Input | Output | Records |
|---|---|---|---|
| Scrape | GitHub repos | `data/raw/scraped.jsonl` | ~6,400 |
| Generate | Teacher model (e.g. llama3:70b) + docs | `data/raw/synthetic.jsonl` | growing |
| Clean | Raw JSONL | `data/cleaned/cleaned.jsonl` | identity filtered, deduped |
| Validate | Cleaned JSONL | `data/validated/passed.jsonl` | validated |
| Format | Passed + `data/identity/conversational.jsonl` | `data/formatted/train.jsonl` | identity + code |
| DPO | Pass/fail pairs | `data/dpo/dpo_pairs.jsonl` | preference pairs |

## CLI Reference

```
nexifuse <command> [options]

Commands:
  ingest       Ingest documentation from docs/ into processed text
  scrape       Scrape configured GitHub repos for training data
  generate     Generate synthetic examples via teacher model
  clean        Clean and deduplicate raw data
  validate     Run multi-format validation on cleaned data
  dpo          Generate DPO preference pairs from validation results
  format       Format validated data (+ optional identity examples) into chat templates
  train        Run SFT fine-tuning with LoRA
  train-dpo    Run DPO alignment training
  merge        Merge LoRA adapter into full model weights
  convert      Convert LoRA adapter to GGUF format
  quantize     Quantize model to GGUF (alias for convert)
  modelfile    Generate Ollama Modelfile
  register     Register GGUF model with Ollama
  serve        Start OpenAI-compatible inference server
  pipeline     Run full data pipeline (ingest → format)

Global options:
  -v, --verbose    Enable debug logging
  -c, --config     Config file path (default: config.yaml)
```

## Tests

```bash
pytest tests/ -v
```

## License

This project is proprietary. See ROADMAP.md for full technical details.
