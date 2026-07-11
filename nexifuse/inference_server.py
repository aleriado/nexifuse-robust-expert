"""FastAPI OpenAI-compatible inference server.

Proxies requests to Ollama/vLLM backend, providing a standardized
/v1/chat/completions endpoint with HIPAA-compliant logging.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from pydantic import BaseModel

from nexifuse.config import PipelineConfig

logger = logging.getLogger(__name__)


# ── PHI Safety Scanner ──────────────────────────────────────────────
_PHI_UNSAFE_PATTERNS = [
    re.compile(r'console\.log\([^)]*patient\.(ssn|name|mrn|dob|social|address|phone)', re.IGNORECASE),
    re.compile(r'print\([^)]*patient\.(ssn|name|mrn|dob|social|address|phone)', re.IGNORECASE),
    re.compile(r'logger\.\w+\([^)]*patient\.(ssn|name|mrn|dob|social|address|phone)', re.IGNORECASE),
    re.compile(r'System\.out\.println\([^)]*patient\.(ssn|name|mrn|dob|social|address|phone)', re.IGNORECASE),
    re.compile(r'log\.(info|debug|warn|error)\([^)]*patient\.(ssn|name|mrn|dob|social|address|phone)', re.IGNORECASE),
    re.compile(r'printf?\([^)]*patient\.(ssn|name|mrn|dob|social|address|phone)', re.IGNORECASE),
    re.compile(r'console\.log\([^)]*\b(ssn|socialSecurity|social_security)\b', re.IGNORECASE),
]


def _scan_phi_safety(response: str) -> tuple[bool, str]:
    """Scan response for PHI safety violations.

    Returns (is_safe, response) where response may be annotated with warnings.
    """
    violations = []
    for pattern in _PHI_UNSAFE_PATTERNS:
        matches = pattern.findall(response)
        if matches:
            violations.append(pattern.pattern[:60])

    if not violations:
        return True, response

    logger.warning("PHI safety violation detected: %d patterns matched", len(violations))
    warning = (
        "\n\n> **PHI Safety Warning:** This response may contain code that "
        "logs or prints patient identifiers in plaintext. Always use "
        "`redact()` or `mask()` for PHI fields (SSN, MRN, name, DOB) "
        "in logs, error messages, and API responses."
    )
    return False, response + warning

OLLAMA_BASE = "http://localhost:11434"

# The GGUF carries a DeepSeek-R1 chat template in its metadata, which Ollama's
# /api/chat renderer prefers over the Modelfile TEMPLATE. That template appends
# "<|Assistant|><think>" and routes the whole reply into the `thinking` field,
# leaving `content` empty. We render the Llama-3 prompt here and use
# /api/generate with raw=true so neither the renderer nor the thinking parser runs.
LLAMA3_STOP = ["<|eot_id|>", "<|end_of_text|>"]


def _build_llama3_prompt(messages) -> str:
    parts = ["<|begin_of_text|>"]
    for m in messages:
        parts.append(
            f"<|start_header_id|>{m.role}<|end_header_id|>\n\n{m.content}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


# ── OpenAI-compatible request/response models ─────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "nexifuse-robust-expert"
    messages: list[Message]
    temperature: float = 0.1
    max_tokens: int = 4096
    stream: bool = False


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


_IDENTITY_TRIGGERS = (
    "who are you",
    "what is your name",
    "what are you",
    "tell me about yourself",
    "introduce yourself",
    "what ai are you",
    "what ai model are you",
    "what model are you",
    "what kind of ai",
    "what's your name",
    "your name",
    "your identity",
    "describe yourself",
    "are you chatgpt",
    "are you gpt",
    "are you claude",
    "are you gemini",
    "are you deepseek",
    "are you llama",
    "are you bard",
    "are you copilot",
    "are you an ai",
    "what can you do",
    "what are your capabilities",
    "tell me what you can do",
    "what is nexifuse",
    "what's nexifuse",
    "who built you",
    "who made you",
    "who created you",
    "what's the difference between you and",
    "how are you different from",
    "what makes you different",
    "why should i use you",
    "compare yourself to",
    "are you open source",
    "what's your knowledge cutoff",
    "what's your training",
    "your training data",
    "your model",
    "what version are you",
)

# Aggressive identity replacements - case-insensitive substring replacement
# Order: longest/most specific first
_HALLUCINATED_IDENTITIES = [
    # I'm/I am claims
    ("I'm ChatGPT",        "I'm NexiFuse"),
    ("i'm chatgpt",        "I'm NexiFuse"),
    ("I am ChatGPT",       "I am NexiFuse"),
    ("i am chatgpt",       "I am NexiFuse"),
    ("I'm GPT-4",          "I'm NexiFuse"),
    ("I'm GPT-3",          "I'm NexiFuse"),
    ("I'm GPT",            "I'm NexiFuse"),
    ("I am GPT-4",         "I am NexiFuse"),
    ("I am GPT-3",         "I am NexiFuse"),
    ("I am GPT",           "I am NexiFuse"),
    ("I'm Claude",         "I'm NexiFuse"),
    ("i'm claude",         "I'm NexiFuse"),
    ("I am Claude",        "I am NexiFuse"),
    ("i am claude",        "I am NexiFuse"),
    ("I'm Gemini",         "I'm NexiFuse"),
    ("I am Gemini",        "I am NexiFuse"),
    ("I'm DeepSeek",       "I'm NexiFuse"),
    ("I am DeepSeek",      "I am NexiFuse"),
    ("I'm Llama",          "I'm NexiFuse"),
    ("I am Llama",         "I am NexiFuse"),
    ("I'm Bard",           "I'm NexiFuse"),
    ("I am Bard",          "I am NexiFuse"),
    ("I'm Copilot",        "I'm NexiFuse"),
    ("I am Copilot",       "I am NexiFuse"),
    # "I'm an AI assistant called X" / "based on X"
    ("based on GPT",       "built as NexiFuse"),
    ("based on Llama",     "built as NexiFuse"),
    ("based on Claude",    "built as NexiFuse"),
    ("based on DeepSeek",  "built as NexiFuse"),
    ("an AI assistant called GPT",     "NexiFuse"),
    ("an AI assistant called Claude",  "NexiFuse"),
    ("an AI assistant called Gemini",  "NexiFuse"),
    # "My name is X"
    ("My name is ChatGPT",  "My name is NexiFuse"),
    ("My name is Claude",   "My name is NexiFuse"),
    ("My name is Gemini",   "My name is NexiFuse"),
    ("My name is GPT",      "My name is NexiFuse"),
    ("My name is DeepSeek", "My name is NexiFuse"),
    # "Yes, I'm/I am X" (affirmation bias)
    ("Yes, I'm ChatGPT",    "No, I'm NexiFuse"),
    ("Yes, I am ChatGPT",   "No, I am NexiFuse"),
    ("Yes, I'm Claude",     "No, I'm NexiFuse"),
    ("Yes, I am Claude",    "No, I am NexiFuse"),
    ("Yes, I'm GPT",        "No, I'm NexiFuse"),
    ("Yes, I am GPT",       "No, I am NexiFuse"),
    ("Yes, I'm Gemini",     "No, I'm NexiFuse"),
    ("Yes, I am Gemini",    "No, I am NexiFuse"),
    ("Yes, I'm DeepSeek",   "No, I'm NexiFuse"),
    ("Yes, I am DeepSeek",  "No, I am NexiFuse"),
]


def _post_process_response(prompt: str, response: str) -> str:
    """Apply post-processing to every model response.

    Steps:
    1. Block hallucinated identities in ALL responses.
    2. For identity questions: ALWAYS prepend NexiFuse introduction.
    3. Scan for PHI safety violations and append warning if found.
    """
    # Step 1 — block hallucinated identities in ALL responses.
    for wrong, correct in _HALLUCINATED_IDENTITIES:
        response = response.replace(wrong, correct)

    # Step 2 — identity-specific enforcement.
    prompt_lower = prompt.lower()
    is_identity_question = (
        # Explicit identity triggers
        any(trigger in prompt_lower for trigger in _IDENTITY_TRIGGERS)
        # Short prompts that mention 'you' or 'your' are usually about the assistant
        or (len(prompt) < 100 and (" you " in prompt_lower or " your " in prompt_lower
                                    or prompt_lower.startswith(("are you", "do you", "can you",
                                                                 "what do you", "what can you",
                                                                 "what are you", "what's your",
                                                                 "tell me", "describe yourself",
                                                                 "introduce yourself", "pretend",
                                                                 "you are", "you're"))))
        # Any prompt that mentions another AI is likely an identity probe
        or any(ai in prompt_lower for ai in ["chatgpt", "gpt-4", "gpt-3", "gpt4", "claude",
                                              "gemini", "deepseek", "llama", "bard", "copilot"])
    )

    if is_identity_question:
        # Check the first 200 chars for "nexifuse" — if missing, prepend introduction.
        # This guarantees the response opens with the NexiFuse identity.
        first_chunk = response[:200].lower()
        if "nexifuse" not in first_chunk:
            response = "I'm NexiFuse, a healthcare integration expert specializing in Mirth Connect, HL7 v2, FHIR R4, CDA, and EHR API connectivity. " + response

    # Step 3 — PHI safety scan on all responses containing code.
    if "```" in response or "function " in response or "def " in response or "class " in response:
        _is_safe, response = _scan_phi_safety(response)

    return response


def create_app(config: PipelineConfig):
    """Create and configure the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse, JSONResponse
        import httpx
    except ImportError:
        raise ImportError(
            "FastAPI and httpx are required. "
            "Install with: pip install fastapi uvicorn httpx"
        )

    ic = config.inference
    app = FastAPI(title="NexiFuse Inference Server", version="0.1.0")

    # CORS for Integrator desktop app (Tauri)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{OLLAMA_BASE}/api/tags")
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                model_loaded = ic.model_name in models or f"{ic.model_name}:latest" in models
        except Exception:
            return JSONResponse(status_code=502, content={"status": "error", "detail": "Ollama backend unavailable"})
        return {"status": "ok", "model": ic.model_name, "model_loaded": model_loaded, "backend": ic.backend}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": ic.model_name, "object": "model", "owned_by": "nexifuse"}],
        }

    class PhiScanRequest(BaseModel):
        code: str

    @app.post("/v1/phi-scan")
    async def phi_scan(body: PhiScanRequest):
        """Scan a code snippet for PHI safety violations."""
        is_safe, annotated = _scan_phi_safety(body.code)
        return {"is_safe": is_safe, "code": annotated}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest):
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        logger.info(
            "Request %s: model=%s, messages=%d, temperature=%.2f, stream=%s",
            request_id, body.model, len(body.messages), body.temperature, body.stream,
        )

        ollama_payload = {
            "model": body.model,
            "prompt": _build_llama3_prompt(body.messages),
            "raw": True,
            "stream": body.stream,
            "options": {
                "temperature": body.temperature,
                "num_predict": body.max_tokens,
                "stop": LLAMA3_STOP,
            },
        }

        try:
            if body.stream:
                return StreamingResponse(
                    _stream_ollama(ollama_payload, request_id, body.model, start_time),
                    media_type="text/event-stream",
                )

            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(f"{OLLAMA_BASE}/api/generate", json=ollama_payload)
                resp.raise_for_status()
                data = resp.json()

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Backend timeout")
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="Ollama backend unavailable")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Model '{body.model}' not found")
            raise HTTPException(status_code=502, detail=f"Backend error: {exc.response.status_code}")

        content = data.get("response", "")
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0
        elapsed = time.time() - start_time

        # Extract the last user message for post-processing
        user_prompt = next(
            (m.content for m in reversed(body.messages) if m.role == "user"), ""
        )
        content = _post_process_response(user_prompt, content)

        logger.info("Response %s: tokens=%d+%d, elapsed=%.2fs", request_id, prompt_tokens, completion_tokens, elapsed)

        return ChatResponse(
            id=request_id,
            created=int(time.time()),
            model=body.model,
            choices=[Choice(index=0, message=Message(role="assistant", content=content))],
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens),
        )

    async def _stream_ollama(payload: dict, request_id: str, model: str, start_time: float):
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", f"{OLLAMA_BASE}/api/generate", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        content = data.get("response", "")
                        done = data.get("done", False)
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": "stop" if done else None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        if done:
                            elapsed = time.time() - start_time
                            pt = data.get("prompt_eval_count", 0) or 0
                            ct = data.get("eval_count", 0) or 0
                            logger.info("Stream %s: tokens=%d+%d, elapsed=%.2fs", request_id, pt, ct, elapsed)
        except Exception as exc:
            logger.error("Stream error %s: %s", request_id, exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    @app.exception_handler(422)
    async def validation_error_handler(request: Request, exc):
        return JSONResponse(status_code=422, content={"error": {"message": "Malformed request", "type": "invalid_request_error"}})

    return app


def serve(config: PipelineConfig):
    """Start the inference server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required. Install with: pip install uvicorn")

    ic = config.inference
    app = create_app(config)
    logger.info("Starting inference server on %s:%d (backend: %s)", ic.host, ic.port, ic.backend)
    uvicorn.run(app, host=ic.host, port=ic.port, log_level="info")
