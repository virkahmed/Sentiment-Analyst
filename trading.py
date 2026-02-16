"""
Kalshi API client: market data, balance (for sizing logic), and simulation/analysis logging to SQLite.
Supports analysis-only (log only) and optional simulation logging.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

# kalshi-python: package may be kalshi-python or kalshi_python_sync
try:
    from kalshi_python import Configuration, KalshiClient, CreateOrderRequest
except ImportError:
    try:
        from kalshi_python import Configuration, KalshiClient
        from kalshi_python.models.create_order_request import CreateOrderRequest
    except ImportError:
        try:
            from kalshi_python_sync import Configuration, KalshiClient, CreateOrderRequest
        except ImportError:
            Configuration = None  # type: ignore
            KalshiClient = None  # type: ignore
            CreateOrderRequest = None  # type: ignore

DEFAULT_MAX_POSITION_PCT = 0.05
DEFAULT_MAX_CONTRACTS = 100


def _load_private_key() -> str:
    path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            return f.read()
    key = os.environ.get("KALSHI_PRIVATE_KEY", "").strip()
    if key:
        return key.replace("\\n", "\n")
    raise ValueError("Set KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY in .env")


def get_kalshi_client() -> Any:
    """Build Kalshi client from env (KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY)."""
    if Configuration is None or KalshiClient is None:
        raise ImportError(
            "Kalshi SDK not found. Install with: pip install -r requirements.txt"
        )
    config = Configuration(host="https://api.elections.kalshi.com/trade-api/v2")
    config.api_key_id = os.environ.get("KALSHI_API_KEY_ID", "")
    config.private_key_pem = _load_private_key()
    return KalshiClient(config)


def ensure_trades_table(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                action TEXT NOT NULL,
                count INTEGER NOT NULL,
                yes_price_cents INTEGER NOT NULL,
                implied_prob REAL NOT NULL,
                confidence REAL NOT NULL,
                dry_run INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                raw_llm_response TEXT,
                balance_snapshot_cents INTEGER
            )
            """
        )


def get_balance_cents(client: Any) -> int:
    """Return available balance in cents."""
    resp = client.get_balance()
    return int(getattr(resp, "balance", 0) or 0)


def compute_position_size(
    balance_cents: int,
    yes_price_cents: int,
    max_pct: float = DEFAULT_MAX_POSITION_PCT,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
) -> int:
    """Max contracts for this trade: floor(5% of balance / yes_price_cents), capped by max_contracts."""
    if yes_price_cents <= 0:
        return 0
    budget_cents = int(balance_cents * max_pct)
    count = budget_cents // yes_price_cents
    return max(0, min(count, max_contracts))


def should_trade(
    recommendation: str,
    confidence: float,
    implied_prob: float,
    kalshi_yes_price_decimal: float,
    min_delta: float,
    confidence_threshold: float,
) -> bool:
    """True if BUY_YES, confidence above threshold, and implied prob - kalshi price >= min_delta."""
    if recommendation != "BUY_YES":
        return False
    if confidence < confidence_threshold:
        return False
    return (implied_prob - kalshi_yes_price_decimal) >= min_delta


def log_trade(
    db_path: str,
    ticker: str,
    side: str,
    action: str,
    count: int,
    yes_price_cents: int,
    implied_prob: float,
    confidence: float,
    dry_run: bool,
    raw_llm_response: str | None = None,
    balance_snapshot_cents: int | None = None,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO trades (ticker, side, action, count, yes_price_cents, implied_prob, confidence, dry_run, created_at, raw_llm_response, balance_snapshot_cents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                side,
                action,
                count,
                yes_price_cents,
                implied_prob,
                confidence,
                1 if dry_run else 0,
                int(time.time()),
                raw_llm_response,
                balance_snapshot_cents,
            ),
        )


def execute_signal(
    client: Any,
    db_path: str,
    ticker: str,
    implied_prob: float,
    confidence: float,
    kalshi_yes_price_cents: int,
    llm_response: dict[str, Any],
    *,
    dry_run: bool = True,
    min_delta: float = 0.10,
    confidence_threshold: float = 0.75,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
) -> bool:
    """
    If recommendation is BUY_YES and confidence/delta pass, either log (dry run) or place order.
    Returns True if a trade was logged or placed.
    """
    recommendation = str(llm_response.get("recommendation", "HOLD")).strip().upper()
    if not should_trade(
        recommendation,
        confidence,
        implied_prob,
        kalshi_yes_price_cents / 100.0,
        min_delta,
        confidence_threshold,
    ):
        return False

    balance_cents = get_balance_cents(client)
    count = compute_position_size(
        balance_cents,
        kalshi_yes_price_cents,
        max_pct=DEFAULT_MAX_POSITION_PCT,
        max_contracts=max_contracts,
    )
    if count <= 0:
        return False

    raw_json = json.dumps(llm_response)

    if dry_run:
        log_trade(
            db_path,
            ticker,
            side="yes",
            action="buy",
            count=count,
            yes_price_cents=kalshi_yes_price_cents,
            implied_prob=implied_prob,
            confidence=confidence,
            dry_run=True,
            raw_llm_response=raw_json,
            balance_snapshot_cents=balance_cents,
        )
        return True

    # Live order
    if CreateOrderRequest is None:
        log_trade(
            db_path,
            ticker,
            side="yes",
            action="buy",
            count=count,
            yes_price_cents=kalshi_yes_price_cents,
            implied_prob=implied_prob,
            confidence=confidence,
            dry_run=True,
            raw_llm_response=raw_json + " (SDK CreateOrderRequest not available)",
            balance_snapshot_cents=balance_cents,
        )
        return True

    try:
        req = CreateOrderRequest(
            ticker=ticker,
            side="yes",
            action="buy",
            count=count,
            type="limit",
            yes_price=kalshi_yes_price_cents,
            time_in_force="good_till_canceled",
        )
        client.create_order(req)
    except Exception:
        # Log failed attempt without double-ordering
        log_trade(
            db_path,
            ticker,
            side="yes",
            action="buy",
            count=count,
            yes_price_cents=kalshi_yes_price_cents,
            implied_prob=implied_prob,
            confidence=confidence,
            dry_run=True,
            raw_llm_response=raw_json + " (order API failed)",
            balance_snapshot_cents=balance_cents,
        )
        return False

    log_trade(
        db_path,
        ticker,
        side="yes",
        action="buy",
        count=count,
        yes_price_cents=kalshi_yes_price_cents,
        implied_prob=implied_prob,
        confidence=confidence,
        dry_run=False,
        raw_llm_response=raw_json,
        balance_snapshot_cents=balance_cents,
    )
    return True
