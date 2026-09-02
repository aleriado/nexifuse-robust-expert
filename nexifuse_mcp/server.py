"""NexiFuse Integrator - MCP server.

Exposes the NexiFuse healthcare-integration expert (Mirth Connect, HL7 v2, FHIR R4, CDA,
DICOM, IHE, and EHR vendor APIs) and its PHI safety scanner as MCP tools, so any AI agent
(Claude, ChatGPT, etc.) can call them.

Tools:
  - ask_integration_expert(question, mode) : ask the fine-tuned NexiFuse model.
  - scan_phi(code)                          : check code for PHI-logging violations (offline).

Config (env):
  NEXIFUSE_API_URL   OpenAI-compatible base of the inference server (default http://localhost:8080/v1)
  NEXIFUSE_API_KEY   optional Bearer token for that server
  NEXIFUSE_MODEL     model name (default nexifuse-robust-expert)
  NEXIFUSE_TIMEOUT   request timeout seconds (default 180)
  NEXIFUSE_MCP_TRANSPORT  stdio (default) | http
  NEXIFUSE_MCP_HOST / NEXIFUSE_MCP_PORT   bind for http transport (default 0.0.0.0:8765)
"""

from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

from nexifuse.phi_scanner import scan_phi as _scan_phi

# The model was fine-tuned with this identity/behaviour prompt. The inference server does NOT
# inject it, so we send it ourselves. Import it, with a self-contained fallback so the MCP
# server runs even if the training package is not importable.
try:
    from nexifuse.prompt_formatter import SYSTEM_PROMPT
except Exception:  # pragma: no cover - fallback only
    SYSTEM_PROMPT = (
        "You are NexiFuse, a healthcare integration expert specializing in Mirth Connect, "
        "HL7 v2, FHIR R4, and EHR API connectivity. You write production-grade integration "
        "code with proper error handling, security best practices, and compliance with "
        "healthcare data standards. Include error handling in all code examples. Never claim "
        "to be ChatGPT, GPT-4, Gemini, or any other AI system."
    )

API_URL = os.getenv("NEXIFUSE_API_URL", "http://localhost:8080/v1").rstrip("/")
API_KEY = os.getenv("NEXIFUSE_API_KEY", "").strip()
MODEL = os.getenv("NEXIFUSE_MODEL", "nexifuse-robust-expert")
TIMEOUT = float(os.getenv("NEXIFUSE_TIMEOUT", "180"))

MODE_DIRECTIVES = {
    "expert": "",
    "ask": "\n\nAnswer concisely and factually. Only include code if essential.",
    "plan": "\n\nProvide architecture and sequencing only. Do not write code.",
    "debug": "\n\nStructure the answer as: Problem Analysis, Diagnostic Commands, Fix Code, Verification Steps.",
}

INSTRUCTIONS = (
    "NexiFuse is a specialist healthcare-integration engineer. Use `ask_integration_expert` "
    "for anything involving Mirth Connect, HL7 v2, FHIR R4, CDA, DICOM, IHE, or EHR vendor "
    "APIs (Epic, Cerner, Athena) - it returns production-grade code with error handling. Use "
    "`scan_phi` to check generated code for patient-identifier logging before you ship it."
)

mcp = FastMCP(
    "NexiFuse Integrator",
    instructions=INSTRUCTIONS,
    host=os.getenv("NEXIFUSE_MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("NEXIFUSE_MCP_PORT", "8765")),
)


@mcp.tool()
async def ask_integration_expert(question: str, mode: str = "expert") -> str:
    """Ask the NexiFuse healthcare-integration expert model.

    Best for Mirth Connect channels, HL7 v2 parsing/mapping, FHIR R4 resources and APIs, CDA,
    DICOM, IHE profiles, and EHR vendor integrations (Epic/Cerner/Athena). Returns
    production-grade code with error handling.

    Args:
        question: The integration question or task.
        mode: expert (default, full answer with code) | ask (concise) | plan (architecture only,
              no code) | debug (Problem Analysis / Diagnostic Commands / Fix Code / Verification).
    """
    directive = MODE_DIRECTIVES.get(mode.lower().strip(), "")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + directive},
            {"role": "user", "content": question},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{API_URL}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]
    except httpx.ConnectError:
        return (
            f"[NexiFuse model is unreachable at {API_URL}. Start the inference server "
            f"(`python -m nexifuse serve`) and Ollama, or set NEXIFUSE_API_URL.]"
        )
    except httpx.HTTPStatusError as exc:
        return f"[NexiFuse expert returned HTTP {exc.response.status_code}: {exc.response.text[:200]}]"
    except Exception as exc:  # noqa: BLE001
        return f"[NexiFuse expert call failed: {exc}]"


@mcp.tool()
def scan_phi(code: str) -> dict:
    """Scan code for PHI-logging safety violations (offline, no model needed).

    Flags code that would write patient identifiers (SSN, MRN, name, DOB, address, phone) to
    logs/console/stdout in plaintext. Run this on any healthcare code before shipping.

    Returns: {"safe": bool, "violations": [{"pattern", "count"}], "warning": str|None}.
    """
    return _scan_phi(code)


@mcp.resource("nexifuse://guide")
def guide() -> str:
    """A short, machine-readable guide to what NexiFuse covers and how to use it."""
    return (
        "# NexiFuse Integrator (MCP)\n\n"
        "A specialist healthcare-integration engineer, available as tools.\n\n"
        "## Domain\n"
        "Mirth Connect, HL7 v2, FHIR R4, CDA, DICOM, IHE, and EHR vendor APIs "
        "(Epic, Cerner, Athena).\n\n"
        "## Tools\n"
        "- `ask_integration_expert(question, mode)` - production-grade integration answers "
        "with error handling. Modes: expert | ask | plan | debug.\n"
        "- `scan_phi(code)` - flags code that logs patient identifiers in plaintext.\n\n"
        "## Compliance\n"
        "Built for healthcare: outputs are PHI-aware and code should be run through `scan_phi` "
        "before use.\n"
    )


def main() -> None:
    transport = os.getenv("NEXIFUSE_MCP_TRANSPORT", "stdio").lower().strip()
    if transport in ("http", "streamable-http", "streamable_http"):
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio (default) - for `claude mcp add` / desktop clients


if __name__ == "__main__":
    main()
