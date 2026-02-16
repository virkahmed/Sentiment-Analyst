"""
Entry point for the Kalshi sentiment analyzer.
Loop: fetch active markets -> match to subreddits -> scrape -> LLM analysis -> log results.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project modules
from brain import estimate_probability
from matcher import match_markets_to_corners
from scraper import SignalScraper
from trading import (
    ensure_trades_table,
    execute_signal,
    get_kalshi_client,
)

DB_PATH = os.environ.get("KALSHI_BOT_DB", "kalshi_bot.sqlite")
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() in ("1", "true", "yes")
POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "120"))
MIN_DELTA = float(os.environ.get("MIN_DELTA", "0.10"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))
MAX_CONTRACTS = int(os.environ.get("MAX_CONTRACTS_PER_TRADE", "100"))


def init_db() -> None:
    import sqlite3
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    ensure_trades_table(DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS seen_posts (post_id TEXT PRIMARY KEY, seen_at INTEGER NOT NULL)"
        )


def get_yes_price_cents_from_orderbook(client, ticker: str) -> int | None:
    """Return best YES bid in cents, or None if unavailable."""
    try:
        ob = client.get_market_orderbook(ticker, depth=5)
        orderbook = getattr(ob, "orderbook", None) or getattr(ob, "orderbook_fp", None)
        if orderbook is None:
            return None
        yes_bids = getattr(orderbook, "yes", None) or getattr(orderbook, "yes_bids", [])
        if not yes_bids:
            return None
        # Bids are [price, quantity]; best bid is typically last (highest price)
        if isinstance(yes_bids[0], (list, tuple)):
            return int(yes_bids[-1][0])
        return int(getattr(orderbook, "yes_bid", 50))
    except Exception:
        return None


def get_market_description(client, ticker: str) -> str:
    """Return title + rules/description for the market."""
    try:
        m = client.get_market(ticker)
        title = getattr(m, "title", None) or getattr(m, "event_ticker", "") or ""
        rules = getattr(m, "rules", None) or getattr(m, "description", None) or getattr(m, "subtitle", "") or ""
        return f"{title}\n{rules}"[:3000]
    except Exception:
        return ""


def run_once(client, dry_run: bool) -> None:
    matched = match_markets_to_corners(client, status="open", limit=100)
    if not matched:
        return

    # Group by (subreddits, keywords) to avoid duplicate scrapes
    scrape_keys: dict[tuple, set[str]] = {}  # (tuple of subreddits, tuple of keywords) -> set of tickers
    for ticker, title, keywords, subreddits in matched:
        if not subreddits or not keywords:
            continue
        key = (tuple(sorted(subreddits)), tuple(sorted(keywords)))
        scrape_keys.setdefault(key, set()).add(ticker)

    reddit_id = os.environ.get("REDDIT_CLIENT_ID")
    reddit_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    reddit_ua = os.environ.get("REDDIT_USER_AGENT")
    openai_key = os.environ.get("OPENAI_API_KEY")

    for (subreddits, keywords), tickers in scrape_keys.items():
        scraper = SignalScraper(
            subreddits=list(subreddits),
            keywords=list(keywords),
            db_path=DB_PATH,
            reddit_client_id=reddit_id,
            reddit_client_secret=reddit_secret,
            reddit_user_agent=reddit_ua,
        )
        try:
            threads = scraper.scrape()
        except Exception:
            threads = []
        if not threads:
            continue
        if not openai_key:
            continue

        for ticker in tickers:
            yes_cents = get_yes_price_cents_from_orderbook(client, ticker)
            if yes_cents is None:
                yes_cents = 50
            description = get_market_description(client, ticker)
            try:
                result = estimate_probability(
                    description,
                    yes_cents / 100.0,
                    threads,
                    api_key=openai_key,
                )
            except Exception:
                continue
            implied = result["implied_probability"]
            confidence = result["confidence_score"]
            execute_signal(
                client,
                DB_PATH,
                ticker,
                implied,
                confidence,
                yes_cents,
                result,
                dry_run=dry_run,
                min_delta=MIN_DELTA,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                max_contracts=MAX_CONTRACTS,
            )


def main() -> None:
    init_db()
    dry_run = DRY_RUN
    try:
        client = get_kalshi_client()
    except Exception as e:
        print(f"Kalshi client init failed: {e}", file=sys.stderr)
        sys.exit(1)

    shutdown = [False]

    def handler(_signum, _frame):
        shutdown[0] = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print(f"Starting analyzer (dry_run={dry_run}, poll_interval={POLL_INTERVAL_SEC}s). Ctrl+C to stop.")
    while not shutdown[0]:
        try:
            run_once(client, dry_run)
        except Exception as e:
            print(f"Run error: {e}", file=sys.stderr)
        for _ in range(POLL_INTERVAL_SEC):
            if shutdown[0]:
                break
            time.sleep(1)
    print("Shutdown complete.")


if __name__ == "__main__":
    main()
