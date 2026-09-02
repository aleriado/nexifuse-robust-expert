"""Paid HTTP entrypoint for the NexiFuse Integrator MCP.

Serves the same MCP tools as ``python -m nexifuse_mcp`` (http transport), but behind the x402
pay-per-call gate: a ``tools/call`` costs credits, bought via ``POST /x402/topup``.

Run:
    python -m nexifuse_mcp.gateway

Key env (see nexifuse_mcp/x402.py for the full list):
    X402_PRICE_ATOMIC     price of one tool call, atomic units (default 10000 = 0.01 USDC)
    X402_PAY_TO           receiving address (required for real payments)
    X402_FACILITATOR_URL  x402 facilitator base URL -> real verify/settle (production)
    X402_DEV_SECRET       enables locally-signed 'dev' payments for testing (never in prod)
    NEXIFUSE_MCP_HOST / NEXIFUSE_MCP_PORT   bind (default 0.0.0.0:8765)

Endpoints:
    /mcp           the MCP streamable-http transport (tools/call is metered)
    /x402/quote    GET/POST - price + x402 payment requirements (free, for discovery)
    /x402/topup    POST with X-PAYMENT - buy credits, returns an API key
    /x402/wallet   GET with Authorization: Bearer <api_key> - remaining balance
"""

from __future__ import annotations

import os

import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from nexifuse_mcp.server import mcp
from nexifuse_mcp.x402 import (
    MCP_PATH,
    Ledger,
    Verifier,
    X402Config,
    X402Middleware,
    build_requirements,
    encode_payment_response,
    payment_required_body,
)

cfg = X402Config()
ledger = Ledger(cfg.ledger_db)
verifier = Verifier(cfg, ledger)


def _bearer(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else ""


async def quote(request: Request) -> JSONResponse:
    resource = str(request.url).split("?")[0].rsplit("/x402/quote", 1)[0] + MCP_PATH
    return JSONResponse({
        "service": "NexiFuse Integrator MCP",
        "mode": cfg.mode,
        "unit_price_atomic": cfg.price_atomic,
        "unit_price": cfg.human_price(),
        "asset": cfg.asset_name,
        "asset_decimals": cfg.asset_decimals,
        "network": cfg.network,
        "pay_to": cfg.pay_to or None,
        "topup": "POST /x402/topup with an X-PAYMENT header to receive an API key with credits",
        "accepts": [build_requirements(cfg, resource, "NexiFuse Integrator MCP - per-call access")],
    })


async def topup(request: Request) -> JSONResponse:
    resource = str(request.url).split("?")[0]
    pay = request.headers.get("x-payment")
    if not pay:
        return JSONResponse(
            payment_required_body(cfg, resource, "Top up NexiFuse MCP credits", "X-PAYMENT header required"),
            status_code=402,
        )
    reqs = build_requirements(cfg, resource, "Top up NexiFuse MCP credits")
    s = await verifier.verify_and_settle(pay, reqs)
    if not s.ok:
        return JSONResponse({"error": s.reason, "x402Version": 1}, status_code=402)

    balance = ledger.credit(s.wallet, s.amount)
    key = ledger.issue_key(s.wallet)
    resp = JSONResponse({
        "api_key": key,
        "wallet": s.wallet,
        "credited_atomic": s.amount,
        "balance_atomic": balance,
        "unit_price_atomic": cfg.price_atomic,
        "calls_available": balance // cfg.price_atomic if cfg.price_atomic else 0,
        "asset": cfg.asset_name,
        "network": cfg.network,
        "usage": "Send this key as 'Authorization: Bearer <api_key>' to the /mcp endpoint. "
                 "Each tools/call costs one unit.",
    })
    if s.receipt:
        resp.headers["X-Payment-Response"] = encode_payment_response(s.receipt)
    return resp


async def wallet(request: Request) -> JSONResponse:
    addr = ledger.address_for_key(_bearer(request))
    if not addr:
        return JSONResponse({"error": "unknown or missing API key"}, status_code=401)
    bal = ledger.balance(addr)
    return JSONResponse({
        "wallet": addr,
        "balance_atomic": bal,
        "unit_price_atomic": cfg.price_atomic,
        "calls_available": bal // cfg.price_atomic if cfg.price_atomic else 0,
    })


def build_app():
    app = mcp.streamable_http_app()
    # Add the x402 support routes alongside /mcp.
    app.router.routes.insert(0, Route("/x402/quote", quote, methods=["GET", "POST"]))
    app.router.routes.insert(0, Route("/x402/topup", topup, methods=["POST"]))
    app.router.routes.insert(0, Route("/x402/wallet", wallet, methods=["GET"]))
    # Wrap everything in the pay-per-call gate.
    return X402Middleware(app, cfg, ledger, verifier)


def main() -> None:
    host = os.getenv("NEXIFUSE_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("NEXIFUSE_MCP_PORT", "8765"))
    print("=" * 74)
    print(" NexiFuse Integrator MCP  -  x402 pay-per-call gate")
    print(f"   endpoint     http://{host}:{port}{MCP_PATH}")
    print(f"   price        {cfg.price_atomic} atomic ({cfg.human_price()} {cfg.asset_name}) / tools_call")
    print(f"   verify mode  {cfg.mode}" + ("  [facilitator: %s]" % cfg.facilitator_url if cfg.facilitator_url else ""))
    print(f"   pay_to       {cfg.pay_to or '(unset)'}")
    print(f"   ledger       {cfg.ledger_db}")
    if cfg.mode == "dev":
        print("   WARNING: dev verifier active (locally-signed payments). Do NOT use in production.")
    if cfg.mode == "facilitator" and not cfg.pay_to:
        print("   WARNING: X402_PAY_TO is unset - real payments have nowhere to settle to.")
    print("=" * 74)
    uvicorn.run(build_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
