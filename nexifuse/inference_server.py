"""FastAPI OpenAI-compatible inference server.

Proxies requests to Ollama/vLLM backend, providing a standardized
/v1/chat/completions endpoint with HIPAA-compliant logging.
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from pydantic import BaseModel

from nexifuse.config import PipelineConfig

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"


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

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest):
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        logger.info(
            "Request %s: model=%s, messages=%d, temperature=%.2f, stream=%s",
            request_id, body.model, len(body.messages), body.temperature, body.stream,
        )

        ollama_messages = [{"role": m.role, "content": m.content} for m in body.messages]
        ollama_payload = {
            "model": body.model,
            "messages": ollama_messages,
            "stream": body.stream,
            "options": {"temperature": body.temperature, "num_predict": body.max_tokens},
        }

        try:
            if body.stream:
                return StreamingResponse(
                    _stream_ollama(ollama_payload, request_id, body.model, start_time),
                    media_type="text/event-stream",
                )

            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=ollama_payload)
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

        content = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0
        elapsed = time.time() - start_time

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
                async with client.stream("POST", f"{OLLAMA_BASE}/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
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
