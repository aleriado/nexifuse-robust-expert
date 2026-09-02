"""Buy MCP credits (dev helper).

In dev mode (X402_DEV_SECRET set) this mints a locally-signed payment, tops up a wallet, and
prints an API key you can hand to any MCP client. In production (a facilitator is configured)
minting a real payment is the job of an x402 wallet/client, so this just prints the quote.

Usage:
    python -m nexifuse_mcp.buy [--url URL] [--calls N] [--wallet 0x..]

Then use the printed key:
    Authorization: Bearer <api_key>   on the /mcp endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys

import httpx

from nexifuse_mcp.x402 import make_dev_payment


def main() -> int:
    ap = argparse.ArgumentParser(description="Buy NexiFuse MCP credits (dev helper).")
    ap.add_argument("--url", default=os.getenv("NEXIFUSE_MCP_URL", "http://localhost:8765"))
    ap.add_argument("--calls", type=int, default=5, help="how many tool calls to buy")
    ap.add_argument("--wallet", default="0x" + secrets.token_hex(20))
    args = ap.parse_args()
    base = args.url.rstrip("/")

    try:
        quote = httpx.get(f"{base}/x402/quote", timeout=10).json()
    except Exception as exc:  # noqa: BLE001
        print(f"Could not reach {base}/x402/quote: {exc}", file=sys.stderr)
        return 1

    price = int(quote["unit_price_atomic"])
    amount = price * max(1, args.calls)
    print(f"Service : {quote['service']}  (mode={quote['mode']})")
    print(f"Price   : {quote['unit_price']} {quote['asset']} / call  ({price} atomic)")
    print(f"Buying  : {args.calls} calls = {amount} atomic for wallet {args.wallet}")

    secret = os.getenv("X402_DEV_SECRET", "").strip()
    if quote["mode"] != "dev" or not secret:
        print("\nThis server is not in dev mode (or X402_DEV_SECRET is unset here), so this helper "
              "cannot mint a payment. Pay with an x402 client using these requirements:")
        print(json.dumps(quote["accepts"][0], indent=2))
        return 2

    payment = make_dev_payment(args.wallet, amount, secret)
    r = httpx.post(f"{base}/x402/topup", headers={"X-PAYMENT": payment}, timeout=30)
    if r.status_code != 200:
        print(f"\nTop-up failed: HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
        return 1
    body = r.json()
    print("\nPaid. Your API key (send as 'Authorization: Bearer <key>' to /mcp):\n")
    print("   " + body["api_key"])
    print(f"\nBalance: {body['balance_atomic']} atomic  |  calls available: {body['calls_available']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
