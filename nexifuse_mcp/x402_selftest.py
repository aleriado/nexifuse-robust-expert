"""Offline self-test for the x402 gate. Run: python -m nexifuse_mcp.x402_selftest

No server, no chain, no model. Exercises the ledger, the dev verifier (signature / expiry /
replay), the requirements builder, and the paid-method detector.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

from nexifuse_mcp.x402 import (
    Ledger,
    Verifier,
    X402Config,
    _is_paid_body,
    build_requirements,
    make_dev_payment,
)

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail and not cond else ""))


def main() -> int:
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "ledger.db")
    cfg = X402Config()
    cfg.ledger_db = db
    cfg.dev_secret = "test-secret-please-change"
    cfg.price_atomic = 10000
    cfg.facilitator_url = ""  # force dev mode
    ledger = Ledger(db)
    verifier = Verifier(cfg, ledger)
    W = "0xAAaaBBbbCCccDDddEEeeFFff0011223344556677"

    print("Ledger")
    check("credit returns new balance", ledger.credit(W, 30000) == 30000)
    check("balance reads back", ledger.balance(W) == 30000)
    ok, rem = ledger.debit(W, 10000, "tools/call")
    check("debit ok when funded", ok and rem == 20000, f"ok={ok} rem={rem}")
    ok2, rem2 = ledger.debit(W, 999999, "tools/call")
    check("debit refused when short", (not ok2) and rem2 == 20000, f"ok={ok2} rem={rem2}")
    ledger.refund(W, 10000)
    check("refund restores balance", ledger.balance(W) == 30000)

    print("API keys")
    key = ledger.issue_key(W)
    check("issued key resolves to wallet", ledger.address_for_key(key) == W)
    check("unknown key resolves to None", ledger.address_for_key("sk_nf_bogus") is None)

    print("Payment replay ledger")
    check("record nonce once", ledger.record_payment("nonce-1", W, 10000, "dev", "nonce-1") is True)
    check("same nonce rejected", ledger.record_payment("nonce-1", W, 10000, "dev", "nonce-1") is False)

    print("Dev verifier")
    reqs = build_requirements(cfg, "http://x/mcp", "test")
    pay = make_dev_payment(W, 50000, cfg.dev_secret)
    s = asyncio.run(verifier.verify_and_settle(pay, reqs))
    check("valid dev payment settles", s.ok and s.wallet == W and s.amount == 50000, s.reason)
    s_replay = asyncio.run(verifier.verify_and_settle(pay, reqs))
    check("dev payment cannot be replayed", not s_replay.ok, s_replay.reason)
    bad = make_dev_payment(W, 50000, "wrong-secret")
    check("bad signature rejected", not asyncio.run(verifier.verify_and_settle(bad, reqs)).ok)
    expired = make_dev_payment(W, 50000, cfg.dev_secret, ttl=-10)
    check("expired payment rejected", not asyncio.run(verifier.verify_and_settle(expired, reqs)).ok)
    check("garbage payment rejected", not asyncio.run(verifier.verify_and_settle("!!notb64!!", reqs)).ok)

    print("Requirements shape")
    check("scheme exact", reqs["scheme"] == "exact")
    check("has payTo/asset/network/maxAmountRequired",
          all(k in reqs for k in ("payTo", "asset", "network", "maxAmountRequired")))

    print("Paid-method detection")
    call = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "scan_phi"}}).encode()
    init = json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize"}).encode()
    lst = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).encode()
    batch = json.dumps([{"method": "tools/list"}, {"method": "tools/call"}]).encode()
    check("tools/call is paid", _is_paid_body(call) is True)
    check("initialize is free", _is_paid_body(init) is False)
    check("tools/list is free", _is_paid_body(lst) is False)
    check("batch with a call is paid", _is_paid_body(batch) is True)
    check("empty body is free", _is_paid_body(b"") is False)

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("-" * 60)
    print(f"{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
