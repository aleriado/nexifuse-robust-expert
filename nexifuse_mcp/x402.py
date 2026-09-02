"""x402 pay-per-call gate for the NexiFuse Integrator MCP.

Wraps the MCP HTTP (streamable-http) transport so that calling a tool costs money, following
the x402 protocol (HTTP 402 Payment Required + an ``X-PAYMENT`` header). Design:

  * A caller tops up a *wallet* by presenting a payment to ``POST /x402/topup``. The server
    verifies + settles it and returns an **API key** carrying a credit balance.
  * The caller then talks to the MCP with ``Authorization: Bearer <api_key>``. Each
    ``tools/call`` decrements the balance. The MCP handshake (``initialize``), ``tools/list``,
    resource reads, ``ping`` and notifications are free.
  * When there is no key / no credit on a ``tools/call``, the gate answers ``402`` with the
    x402 ``accepts`` payment requirements, pointing at ``/x402/topup``.

Two verifier backends (no heavy deps, no chain required to test):

  * ``facilitator`` - if ``X402_FACILITATOR_URL`` is set, real x402 verify/settle is delegated
    to that facilitator (e.g. Coinbase's). Production path.
  * ``dev`` (default) - a locally HMAC-signed payment payload is accepted, so the whole
    flow is exercisable end to end right now without a wallet or testnet. Never enable in prod.

Everything here uses only the stdlib plus httpx (already a dependency) and starlette (ships
with mcp). The ledger is SQLite.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from starlette.datastructures import MutableHeaders

# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------

MCP_PATH = os.getenv("NEXIFUSE_MCP_PATH", "/mcp")
_DEBUG = os.getenv("X402_DEBUG", "").strip() not in ("", "0", "false", "no")


@dataclass
class X402Config:
    pay_to: str = field(default_factory=lambda: os.getenv("X402_PAY_TO", "").strip())
    network: str = field(default_factory=lambda: os.getenv("X402_NETWORK", "base-sepolia"))
    asset: str = field(default_factory=lambda: os.getenv(
        "X402_ASSET", "0x036CbD53842c5426634e7929541eC2318f3dCF7e"))  # USDC base-sepolia
    asset_name: str = field(default_factory=lambda: os.getenv("X402_ASSET_NAME", "USDC"))
    asset_decimals: int = field(default_factory=lambda: int(os.getenv("X402_ASSET_DECIMALS", "6")))
    # price of one tools/call, in atomic units of the asset (USDC 6dp: 10000 = 0.01 USDC)
    price_atomic: int = field(default_factory=lambda: int(os.getenv("X402_PRICE_ATOMIC", "10000")))
    facilitator_url: str = field(default_factory=lambda: os.getenv("X402_FACILITATOR_URL", "").strip())
    dev_secret: str = field(default_factory=lambda: os.getenv("X402_DEV_SECRET", "").strip())
    ledger_db: str = field(default_factory=lambda: os.getenv(
        "X402_LEDGER_DB", os.path.join(os.path.dirname(__file__), "x402_ledger.db")))
    max_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("X402_MAX_TIMEOUT", "60")))

    @property
    def mode(self) -> str:
        return "facilitator" if self.facilitator_url else "dev"

    def human_price(self) -> str:
        return f"{self.price_atomic / (10 ** self.asset_decimals):.{self.asset_decimals}f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------------------------
# Ledger (SQLite)
# --------------------------------------------------------------------------------------------

class Ledger:
    """Tiny SQLite credit ledger. One connection guarded by a lock (low volume, simple + safe)."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS wallets (
                    address TEXT PRIMARY KEY,
                    balance INTEGER NOT NULL DEFAULT 0,
                    created REAL, updated REAL
                );
                CREATE TABLE IF NOT EXISTS apikeys (
                    key_hash TEXT PRIMARY KEY,
                    address  TEXT NOT NULL,
                    created  REAL
                );
                CREATE TABLE IF NOT EXISTS payments (
                    nonce   TEXT PRIMARY KEY,
                    address TEXT, amount INTEGER, scheme TEXT, ref TEXT, ts REAL
                );
                CREATE TABLE IF NOT EXISTS usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    address TEXT, method TEXT, cost INTEGER, ts REAL
                );
                """
            )
            self._conn.commit()

    # -- wallets ------------------------------------------------------------------------------
    def credit(self, address: str, amount: int) -> int:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO wallets(address,balance,created,updated) VALUES(?,?,?,?) "
                "ON CONFLICT(address) DO UPDATE SET balance=balance+excluded.balance, updated=excluded.updated",
                (address, amount, now, now),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT balance FROM wallets WHERE address=?", (address,)).fetchone()
            return int(row["balance"]) if row else 0

    def balance(self, address: str) -> int:
        with self._lock:
            row = self._conn.execute("SELECT balance FROM wallets WHERE address=?", (address,)).fetchone()
            return int(row["balance"]) if row else 0

    def debit(self, address: str, cost: int, method: str) -> tuple[bool, int]:
        """Atomic check-and-decrement. Returns (ok, remaining_balance)."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE wallets SET balance=balance-?, updated=? WHERE address=? AND balance>=?",
                (cost, now, address, cost),
            )
            if cur.rowcount == 0:
                row = self._conn.execute("SELECT balance FROM wallets WHERE address=?", (address,)).fetchone()
                self._conn.commit()
                return False, int(row["balance"]) if row else 0
            self._conn.execute(
                "INSERT INTO usage(address,method,cost,ts) VALUES(?,?,?,?)", (address, method, cost, now)
            )
            row = self._conn.execute("SELECT balance FROM wallets WHERE address=?", (address,)).fetchone()
            self._conn.commit()
            return True, int(row["balance"])

    def refund(self, address: str, amount: int) -> None:
        self.credit(address, amount)

    # -- api keys -----------------------------------------------------------------------------
    def issue_key(self, address: str) -> str:
        key = "sk_nf_" + secrets.token_urlsafe(24)
        with self._lock:
            self._conn.execute(
                "INSERT INTO apikeys(key_hash,address,created) VALUES(?,?,?)",
                (_hash_key(key), address, time.time()),
            )
            self._conn.commit()
        return key

    def address_for_key(self, key: str) -> Optional[str]:
        if not key:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT address FROM apikeys WHERE key_hash=?", (_hash_key(key),)
            ).fetchone()
            return row["address"] if row else None

    # -- payments (replay protection) ---------------------------------------------------------
    def record_payment(self, nonce: str, address: str, amount: int, scheme: str, ref: str) -> bool:
        """Returns True if recorded, False if the nonce was already used (replay)."""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO payments(nonce,address,amount,scheme,ref,ts) VALUES(?,?,?,?,?,?)",
                    (nonce, address, amount, scheme, ref, time.time()),
                )
                self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# --------------------------------------------------------------------------------------------
# Payment requirements + verification
# --------------------------------------------------------------------------------------------

def build_requirements(cfg: X402Config, resource: str, description: str) -> dict:
    """One x402 PaymentRequirements object (the 'exact' scheme)."""
    return {
        "scheme": "exact",
        "network": cfg.network,
        "maxAmountRequired": str(cfg.price_atomic),
        "resource": resource,
        "description": description,
        "mimeType": "application/json",
        "payTo": cfg.pay_to or "0x0000000000000000000000000000000000000000",
        "maxTimeoutSeconds": cfg.max_timeout_seconds,
        "asset": cfg.asset,
        "extra": {"name": cfg.asset_name, "version": "2", "decimals": cfg.asset_decimals},
    }


def payment_required_body(cfg: X402Config, resource: str, description: str, error: str) -> dict:
    return {
        "x402Version": 1,
        "error": error,
        "accepts": [build_requirements(cfg, resource, description)],
    }


@dataclass
class Settlement:
    ok: bool
    wallet: str = ""
    amount: int = 0
    scheme: str = ""
    ref: str = ""
    reason: str = ""
    receipt: dict = field(default_factory=dict)


def make_dev_payment(wallet: str, amount: int, secret: str, ttl: int = 300) -> str:
    """Build a base64 dev X-PAYMENT payload (for tests / non-chain clients)."""
    nonce = secrets.token_hex(16)
    exp = int(time.time()) + ttl
    sig = _dev_sig(wallet, amount, nonce, exp, secret)
    payload = {"x402Version": 1, "scheme": "dev", "wallet": wallet,
               "amount": str(amount), "nonce": nonce, "exp": exp, "sig": sig}
    return base64.b64encode(json.dumps(payload).encode()).decode()


def _dev_sig(wallet: str, amount: int, nonce: str, exp: int, secret: str) -> str:
    msg = f"{wallet}|{amount}|{nonce}|{exp}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


class Verifier:
    def __init__(self, cfg: X402Config, ledger: Ledger):
        self.cfg = cfg
        self.ledger = ledger

    async def verify_and_settle(self, payment_b64: str, requirements: dict) -> Settlement:
        try:
            raw = base64.b64decode(payment_b64)
            payload = json.loads(raw)
        except Exception:  # noqa: BLE001
            return Settlement(False, reason="malformed X-PAYMENT (expected base64 JSON)")

        scheme = str(payload.get("scheme", "")).lower()
        if scheme == "dev":
            return self._settle_dev(payload)
        if self.cfg.facilitator_url:
            return await self._settle_facilitator(payload, requirements)
        return Settlement(False, reason=f"unsupported scheme '{scheme}' (no facilitator configured)")

    def _settle_dev(self, payload: dict) -> Settlement:
        if not self.cfg.dev_secret:
            return Settlement(False, reason="dev payments disabled (set X402_DEV_SECRET)")
        try:
            wallet = str(payload["wallet"])
            amount = int(payload["amount"])
            nonce = str(payload["nonce"])
            exp = int(payload["exp"])
            sig = str(payload["sig"])
        except (KeyError, ValueError, TypeError):
            return Settlement(False, reason="dev payment missing fields")
        if time.time() > exp:
            return Settlement(False, reason="dev payment expired")
        expected = _dev_sig(wallet, amount, nonce, exp, self.cfg.dev_secret)
        if not hmac.compare_digest(expected, sig):
            return Settlement(False, reason="bad dev payment signature")
        if amount <= 0:
            return Settlement(False, reason="amount must be positive")
        if not self.ledger.record_payment(nonce, wallet, amount, "dev", nonce):
            return Settlement(False, reason="payment nonce already used (replay)")
        return Settlement(True, wallet=wallet, amount=amount, scheme="dev", ref=nonce,
                          receipt={"scheme": "dev", "nonce": nonce, "network": self.cfg.network})

    async def _settle_facilitator(self, payload: dict, requirements: dict) -> Settlement:
        body = {"x402Version": 1, "paymentPayload": payload, "paymentRequirements": requirements}
        base = self.cfg.facilitator_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=self.cfg.max_timeout_seconds) as client:
                v = await client.post(f"{base}/verify", json=body)
                v.raise_for_status()
                vr = v.json()
                if not vr.get("isValid"):
                    return Settlement(False, reason=vr.get("invalidReason", "facilitator rejected payment"))
                s = await client.post(f"{base}/settle", json=body)
                s.raise_for_status()
                sr = s.json()
        except httpx.HTTPError as exc:
            return Settlement(False, reason=f"facilitator error: {exc}")
        if not sr.get("success"):
            return Settlement(False, reason=sr.get("errorReason", "settlement failed"))
        wallet = sr.get("payer") or vr.get("payer") or "unknown"
        tx = sr.get("transaction", "")
        amount = int(requirements.get("maxAmountRequired", self.cfg.price_atomic))
        # replay protection keyed on the settlement tx / payer+nonce
        nonce = tx or f"{wallet}:{payload.get('nonce','')}"
        if not self.ledger.record_payment(nonce, wallet, amount, "exact", tx):
            return Settlement(False, reason="payment already settled (replay)")
        return Settlement(True, wallet=wallet, amount=amount, scheme="exact", ref=tx, receipt=sr)


def encode_payment_response(receipt: dict) -> str:
    return base64.b64encode(json.dumps(receipt).encode()).decode()


# --------------------------------------------------------------------------------------------
# ASGI middleware: charge per tools/call on the MCP endpoint
# --------------------------------------------------------------------------------------------

PAID_METHODS = {"tools/call"}


def _is_paid_body(raw: bytes) -> bool:
    if not raw:
        return False
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return False
    msgs = data if isinstance(data, list) else [data]
    return any(isinstance(m, dict) and m.get("method") in PAID_METHODS for m in msgs)


def _jsonrpc_id(raw: bytes):
    """The JSON-RPC id of the (first) paid message, so a refusal can be returned in-protocol."""
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    msgs = data if isinstance(data, list) else [data]
    for m in msgs:
        if isinstance(m, dict) and m.get("method") in PAID_METHODS:
            return m.get("id")
    return msgs[0].get("id") if msgs and isinstance(msgs[0], dict) else None


def _bearer(scope) -> str:
    for k, v in scope.get("headers", []):
        if k == b"authorization":
            val = v.decode()
            if val.lower().startswith("bearer "):
                return val[7:].strip()
    return ""


def _header(scope, name: bytes) -> str:
    for k, v in scope.get("headers", []):
        if k == name:
            return v.decode()
    return ""


class X402Middleware:
    """Pure-ASGI gate. Only intercepts POST <MCP_PATH> carrying a tools/call; all else passes."""

    def __init__(self, app, cfg: X402Config, ledger: Ledger, verifier: Verifier):
        self.app = app
        self.cfg = cfg
        self.ledger = ledger
        self.verifier = verifier

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" \
                or not scope.get("path", "").rstrip("/").endswith(MCP_PATH.rstrip("/")):
            await self.app(scope, receive, send)
            return

        # Buffer body so we can inspect the JSON-RPC method, then replay it downstream.
        body = b""
        while True:
            msg = await receive()
            if msg["type"] == "http.request":
                body += msg.get("body", b"")
                if not msg.get("more_body", False):
                    break
            elif msg["type"] == "http.disconnect":
                break

        paid = _is_paid_body(body)
        if _DEBUG:
            import sys as _sys
            print(f"[x402] POST {scope.get('path')} paid={paid} bearer={_bearer(scope)[:10]!r} "
                  f"body0={body[:120]!r}", file=_sys.stderr, flush=True)
        if not paid:
            await self.app(scope, _replayer(body, receive), send)  # free: initialize, tools/list...
            return

        resource = self.cfg_resource(scope)
        rid = _jsonrpc_id(body)
        wallet = self.ledger.address_for_key(_bearer(scope))
        if _DEBUG:
            import sys as _sys
            print(f"[x402] paid call: wallet={wallet} balance={self.ledger.balance(wallet) if wallet else None}",
                  file=_sys.stderr, flush=True)

        # Path A: prepaid credits via API key.
        if wallet:
            ok, remaining = self.ledger.debit(wallet, self.cfg.price_atomic, "tools/call")
            if ok:
                await self._forward_charged(scope, body, receive, send, wallet, remaining)
                return
            # key valid but out of credit -> fall through to 402

        # Path B: inline per-call payment (x402-native clients).
        pay = _header(scope, b"x-payment")
        if pay:
            reqs = build_requirements(self.cfg, resource, "NexiFuse Integrator MCP - per-call access")
            settlement = await self.verifier.verify_and_settle(pay, reqs)
            if settlement.ok:
                self.ledger.credit(settlement.wallet, settlement.amount)
                ok, remaining = self.ledger.debit(settlement.wallet, self.cfg.price_atomic, "tools/call")
                if ok:
                    await self._forward_charged(scope, body, receive, send, settlement.wallet,
                                                remaining, receipt=settlement.receipt)
                    return
                await self._send_402(send, resource, "payment did not cover the call price", rid)
                return
            await self._send_402(send, resource, f"payment rejected: {settlement.reason}", rid)
            return

        await self._send_402(send, resource,
                             "payment required: top up at /x402/topup or send an X-PAYMENT header", rid)

    def cfg_resource(self, scope) -> str:
        host = _header(scope, b"host") or f"{self.cfg.network}"
        scheme = "https" if scope.get("scheme") == "https" else "http"
        return f"{scheme}://{host}{scope.get('path', MCP_PATH)}"

    async def _forward_charged(self, scope, body, receive, send, wallet, remaining,
                               receipt: dict | None = None):
        charged = {"amount": self.cfg.price_atomic, "wallet": wallet, "refunded": False}
        status = {"code": 200}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = MutableHeaders(raw=message.setdefault("headers", []))
                headers["X-Nexifuse-Balance"] = str(remaining)
                headers["X-Nexifuse-Price"] = str(self.cfg.price_atomic)
                if receipt:
                    headers["X-Payment-Response"] = encode_payment_response(receipt)
            await send(message)

        await self.app(scope, _replayer(body, receive), send_wrapper)
        # Refund transport-level failures (tool-level errors come back as 200 + isError).
        if status["code"] >= 500 and not charged["refunded"]:
            self.ledger.refund(wallet, charged["amount"])

    async def _send_402(self, send, resource, error, request_id=None):
        desc = "NexiFuse Integrator MCP - per-call access"
        accepts = build_requirements(self.cfg, resource, desc)
        if request_id is not None:
            # In-protocol refusal: MCP clients raise a clean McpError instead of a transport
            # exception. Carry the x402 payment info in error.data so x402-aware agents can pay.
            payload = {"jsonrpc": "2.0", "id": request_id, "error": {
                "code": 402,
                "message": f"Payment required: {error}",
                "data": {"x402Version": 1, "accepts": [accepts], "topup": "/x402/topup"},
            }}
            body = json.dumps(payload).encode()
            status = 200
        else:
            body = json.dumps(payment_required_body(self.cfg, resource, desc, error)).encode()
            status = 402
        await send({"type": "http.response.start", "status": status, "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]})
        await send({"type": "http.response.body", "body": body})


def _replayer(body: bytes, original_receive):
    """A receive() that replays the buffered body once, then delegates to the original
    receive. Delegating (rather than faking an immediate http.disconnect) is essential: the
    ASGI server polls receive() for disconnects while streaming an SSE response, and a premature
    disconnect would abort the stream mid-flight (breaking the MCP session)."""
    sent = {"done": False}

    async def receive():
        if not sent["done"]:
            sent["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await original_receive()

    return receive
