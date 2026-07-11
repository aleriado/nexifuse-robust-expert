# NexiFuse-Integrator - Project Overview

Internal briefing. Everything below was read out of the code in this repo. Items marked
"Unverified" are flagged rather than guessed.

---

## 1. What NexiFuse-Integrator is

Two halves of one product:

| Half | What it is | Lives in |
|---|---|---|
| The model pipeline | Scrapes and synthesises healthcare-integration training data, fine-tunes an LLM, exports it to GGUF, and serves it behind an OpenAI-compatible API. | `nexifuse/` (Python) |
| The Integrator | A Tauri desktop app - an API client (think Postman) with the NexiFuse model wired in as a pair-programmer for healthcare integrations. | `integrator/` (React + Rust) |

The domain is deliberately narrow: Mirth Connect, HL7 v2, FHIR R4, CDA, DICOM, IHE, and EHR
vendor APIs (Epic, Cerner, Athena). The system prompt hard-codes that identity and six
behavioural rules, including "never claim to be ChatGPT/GPT-4/Gemini" and "include error
handling in all code examples" (`nexifuse/prompt_formatter.py:17-29`).

---

## 2. Architecture

```
+---------------------------------------------------------+
|  Integrator desktop app  (Tauri v2)                      |
|                                                          |
|  React UI --invoke()--> Rust core (14 commands)          |
|     |                     * HTTP client (no CORS)        |
|     |                     * context / doc store          |
|     |                     * audit log, stress test       |
|     |                                                    |
|     +--fetch()--> POST /v1/chat/completions              |
+--------------------------+-------------------------------+
                           |  OpenAI-compatible, SSE streaming
                  +--------v---------+
                  | inference_server |  FastAPI + uvicorn
                  |  * PHI scanner   |
                  |  * CORS          |
                  +--------+---------+
                           |  /api/generate  (raw=true)
                  +--------v---------+
                  |     Ollama       |  :11434
                  | nexifuse-robust- |
                  | expert (Q4_K_M)  |
                  +------------------+
```

The client speaks the OpenAI API, so anything that already talks to OpenAI can point at
NexiFuse-Integrator unchanged.

---

## 3. The model pipeline (`nexifuse/`)

### Modules

| Module | Purpose |
|---|---|
| `doc_ingester.py` | Ingest reference docs from `docs/` |
| `scraper.py` | Scrape GitHub repos for real integration code |
| `data_factory.py` | Synthesise training data via a teacher model |
| `dpo_generator.py` | Build DPO preference pairs |
| `data_cleaner.py` | Clean and deduplicate raw data |
| `validator.py` | Validate training examples |
| `prompt_formatter.py` | Render examples into the Llama-3 chat template |
| `training_pipeline.py` | SFT / GRPO / SimPO / DPO training |
| `gguf_converter.py` | Merge LoRA -> GGUF -> Ollama Modelfile -> register |
| `inference_server.py` | OpenAI-compatible FastAPI server |
| `config.py` | Dataclass config, overridden by `config.yaml` |
| `cli.py` | `python -m nexifuse <command>` |

### CLI surface

Data: `ingest`, `scrape`, `generate`, `generate-general`, `generate-conversations`,
`generate-conceptual`, `generate-raw-hl7`, `clean`, `validate`, `dpo`, `format`

Training: `train`, `train-multigpu` (Accelerate DDP), `train-grpo`, `train-simpo`, `train-dpo`

Ship: `merge`, `convert`, `modelfile`, `register`, `serve`

Orchestration: `pipeline`, `pipeline-v2` (generate -> clean -> validate -> format -> SFT ->
GRPO -> SimPO), `pipeline-20k`

### Training setup

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (`config.py:15`)
- Method: LoRA, rank 16 / alpha 32, on all seven attention and MLP projections (`config.py:16-23`)
- Quantisation: `nf4` during training; Q4_K_M for the shipped GGUF
- Alignment: three preference methods are implemented - GRPO (verifiable rewards), SimPO, DPO
- Teacher models: `deepseek-v3.1` for reasoning data, `qwen3-coder:30b-a3b` for bulk (`config.py:40-41`)

Unverified, needs a decision: `config.py:15` names the 70B distill, but the shipped GGUF
reports `architecture llama, parameters 8.0B`. The two disagree. Someone should confirm which
checkpoint is actually in production.

Also: `outputs/nexifuse-v3-q4km.gguf` and `outputs/nexifuse-v35-r1-q4km.gguf` are
byte-identical - Ollama dedupes them to the same blob digest. They are the same model under
two names.

---

## 4. The Integrator app (`integrator/`)

### Rust core - 14 Tauri commands (`src-tauri/src/lib.rs:613-628`)

| Group | Commands |
|---|---|
| HTTP | `execute_request`, `get_traffic`, `discover_endpoints`, `stress_test` |
| Context / RAG | `context_ingest`, `context_ingest_url`, `context_search`, `context_list`, `context_remove` |
| Compliance | `audit_record`, `audit_list` |
| Workspace | `project_read_file`, `project_write_file`, `open_in_terminal` |

Requests go through Rust, not the browser, so the app is not subject to CORS and can hit
internal hospital endpoints directly. That is a real advantage over web-based API clients.

### Agent modes (`src/agent.ts:22-32`)

| Mode | Behaviour |
|---|---|
| Agent | Full power. May rewrite the current request (`ACTION: {...}`) and emit code files. |
| Plan | Architecture and sequencing only. No code, no actions. |
| Debug | Forced structure: Problem Analysis -> Diagnostic Commands -> Fix Code -> Verification Steps. |
| Ask | Short factual answers. No code, no actions. |

Agent mode can drive the UI: it emits one `ACTION:` line and the app applies it - `set_url`,
`set_method`, `set_auth_bearer`, `set_body`, `set_header`. Generated code lands in the Code
Generation tab.

Context: `@`-mention ingested docs and they are retrieved and injected as a `[DOCUMENTATION]`
block; recent traffic (last 3 requests) is injected as a `[CURRENT REQUEST]` block.

### Compliance features - the differentiator

- PHI response scanner (`inference_server.py:23-55`): seven regexes catch model output that
  would log patient identifiers (`console.log(patient.ssn)`, `logger.info(patient.mrn)`, and
  so on). A violation appends a visible PHI Safety Warning to the answer.
- Secret masking (`src/lib/security.ts:6-18`): `Bearer` tokens, `api_key`, `token`,
  `password`, and cookie headers are redacted before anything is logged or sent to the model.
- Audit log (`security.ts:20-30`): every agent prompt is recorded, capped at 500 entries
  client-side and 1000 server-side, mirrored into Rust via `audit_record`.
- PhiBadge on rendered agent messages.

This is the part that makes it sellable into hospitals. It should lead the demo.

---

## 6. Known issues - be honest about these

1. DeepSeek-R1 template mismatch (fixed, but the root cause remains).
   The GGUF carries DeepSeek-R1's Jinja chat template in its metadata, inherited from the base
   model. Ollama's `/api/chat` prefers that template over our Modelfile `TEMPLATE`. It ends
   with `<|Assistant|><think>`, forcing reasoning mode, so Ollama routed the whole reply into a
   `thinking` field and returned `content: ""`. The app rendered empty bubbles.
   Fix applied: `inference_server.py` now renders the Llama-3 prompt itself and calls
   `/api/generate` with `raw: true`, bypassing both the Jinja renderer and the thinking parser.
   This matches how the model was actually trained (`prompt_formatter.py:32`).
   Proper fix: re-export the GGUF with the Llama-3 chat template in metadata, then `/api/chat`
   works normally.

2. No authentication. `/v1/chat/completions` is unauthenticated and currently bound to
   `0.0.0.0`. The client already sends `Authorization: Bearer <apiKey>` when configured
   (`agent.ts:81`); the server just ignores it. A ten-line FastAPI dependency closes this.

3. `npm run build` fails typecheck. Three pre-existing errors: unused `path`
   (`search-palette.tsx:24`), unused `Command` (`top-bar.tsx:1`), and `"Zoom"` is not a valid
   Tauri `PredefinedMenuItem` (`app-menu.ts:104`). `tauri dev` never runs `tsc`, which is why
   these went unnoticed. The web build currently uses `vite build` to skip typecheck.

4. Browser mode is degraded. Every `invoke()` site is guarded by `isTauriRuntime()`, so in a
   browser the app silently loses doc ingestion, project file writes, `open_in_terminal`, and
   context search. Chat and the request panel still work. Good design, but the gap should be
   visible in the UI rather than silent.

5. Config precedence is a footgun. `config.yaml` overrides the dataclass defaults in
   `config.py`. Editing `config.py` alone does nothing. This has already cost debugging time.
