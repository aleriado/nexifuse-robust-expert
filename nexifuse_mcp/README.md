# NexiFuse Integrator - MCP server

Exposes the NexiFuse **healthcare-integration expert** and its **PHI safety scanner** as MCP
tools, so any AI agent (Claude, ChatGPT, etc.) can use them.

## Tools

| Tool | What it does |
|---|---|
| `ask_integration_expert(question, mode)` | Ask the fine-tuned NexiFuse model. Domain: Mirth Connect, HL7 v2, FHIR R4, CDA, DICOM, IHE, EHR vendor APIs (Epic/Cerner/Athena). Returns production-grade code with error handling. Modes: `expert` (default) / `ask` / `plan` / `debug`. |
| `scan_phi(code)` | Flags code that would log/print patient identifiers (SSN, MRN, name, DOB, ...) in plaintext. Runs offline - no model needed. |

Resource `nexifuse://guide` returns a short machine-readable description of the domain and tools.

## Install

```bash
pip install -r nexifuse_mcp/requirements.txt
```

`ask_integration_expert` calls the OpenAI-compatible inference server (default
`http://localhost:8080/v1`), so start it for that tool to work:

```bash
python -m nexifuse serve        # + Ollama running with the nexifuse-robust-expert model
```

`scan_phi` needs nothing running.

## Run

```bash
# stdio (default) - for desktop/CLI MCP clients
python -m nexifuse_mcp

# HTTP (for remote / agent access)
NEXIFUSE_MCP_TRANSPORT=http NEXIFUSE_MCP_PORT=8765 python -m nexifuse_mcp
```

## Add to Claude Code

```bash
claude mcp add nexifuse -- python -m nexifuse_mcp
```

Or in `.mcp.json`:

```json
{
  "mcpServers": {
    "nexifuse": {
      "command": "python",
      "args": ["-m", "nexifuse_mcp"],
      "env": {
        "NEXIFUSE_API_URL": "http://localhost:8080/v1",
        "NEXIFUSE_MODEL": "nexifuse-robust-expert"
      }
    }
  }
}
```

