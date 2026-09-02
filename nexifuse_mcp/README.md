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

## Pay-per-call gate (x402)

`nexifuse_mcp.gateway` serves the same tools over HTTP but behind an **x402 pay-per-call gate**:
a `tools/call` costs credits; the MCP handshake, `tools/list`, resource reads and `ping` are
free. Credits are bought with an x402 payment and spent via an API key.

```bash
python -m nexifuse_mcp.gateway        # instead of `python -m nexifuse_mcp` (http)
```

Flow:

1. **Discover the price** - `GET /x402/quote` returns the price and x402 `accepts` requirements.
2. **Buy credits** - `POST /x402/topup` with an `X-PAYMENT` header. The server verifies +
   settles the payment, credits a wallet, and returns an **API key**. No payment -> `402`.
3. **Call tools** - connect to `/mcp` with `Authorization: Bearer <api_key>`. Each `tools/call`
   decrements the balance (`X-Nexifuse-Balance` header shows the remainder). Out of credit -> `402`.
4. **Check balance** - `GET /x402/wallet` with the bearer key.

Two verifier backends:

- **`facilitator`** (production) - set `X402_FACILITATOR_URL` and real x402 verify/settle is
  delegated to that facilitator (e.g. Coinbase's). Set `X402_PAY_TO` to your receiving address.
- **`dev`** (default, testing only) - set `X402_DEV_SECRET` to accept locally-signed payments so
  the whole flow runs with no chain. Never enable in production.

Config (env): `X402_PRICE_ATOMIC` (default 10000 = 0.01 USDC), `X402_PAY_TO`, `X402_NETWORK`
(default base-sepolia), `X402_ASSET`, `X402_FACILITATOR_URL`, `X402_DEV_SECRET`, `X402_LEDGER_DB`.

Test it end to end (dev mode):

```bash
export X402_DEV_SECRET=some-secret X402_PAY_TO=0xYourAddress
python -m nexifuse_mcp.gateway &                                   # start the gated server
X402_DEV_SECRET=some-secret python -m nexifuse_mcp.buy --calls 5   # prints an API key
python -m nexifuse_mcp.x402_selftest                              # 21 offline checks
```

Then point any MCP client at `/mcp` with `Authorization: Bearer <api_key>`.

> The plain `python -m nexifuse_mcp` http transport stays **unauthenticated** - use the gateway
> when exposing the service. Still open: a ChatGPT connector, a fuller agent-discovery guide, and
> ops hardening (process manager + HTTPS in front).

