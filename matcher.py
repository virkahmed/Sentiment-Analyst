"""
Market matcher: fetch active Kalshi markets and map them to subreddits via keywords.
"""

from __future__ import annotations

from typing import Any

from scraper import extract_keywords_from_text


# Keyword -> list of subreddit names (without r/). Expand as needed.
KEYWORD_TO_SUBREDDITS: dict[str, list[str]] = {
    "fed": ["fedwatch", "economics", "investing", "wallstreetbets"],
    "rate": ["fedwatch", "economics", "investing"],
    "interest": ["fedwatch", "economics", "investing"],
    "cpi": ["economics", "investing", "inflation"],
    "inflation": ["economics", "investing", "inflation"],
    "senate": ["politics", "neutralpolitics", "politicaldiscussion"],
    "congress": ["politics", "neutralpolitics", "politicaldiscussion"],
    "vote": ["politics", "neutralpolitics", "politicaldiscussion"],
    "legislative": ["politics", "neutralpolitics", "politicaldiscussion"],
    "economy": ["economics", "investing", "fedwatch"],
    "gdp": ["economics", "investing"],
    "jobs": ["economics", "investing"],
    "employment": ["economics", "investing"],
    "recession": ["economics", "investing", "stockmarket"],
    "election": ["politics", "neutralpolitics", "politicaldiscussion"],
    "trump": ["politics", "neutralpolitics", "politicaldiscussion"],
    "biden": ["politics", "neutralpolitics", "politicaldiscussion"],
    "fomc": ["fedwatch", "economics", "investing"],
    "powell": ["fedwatch", "economics", "investing"],
}


def get_keyword_to_subreddits() -> dict[str, list[str]]:
    """Return the static mapping (can be overridden by config later)."""
    return dict(KEYWORD_TO_SUBREDDITS)


def extract_market_keywords(market: Any) -> list[str]:
    """Extract tokens from market title and subtitle/description."""
    title = getattr(market, "title", None) or getattr(market, "title", "") or ""
    subtitle = getattr(market, "subtitle", None) or ""
    # Some SDKs use different attr names
    description = getattr(market, "description", None) or getattr(market, "rules", None) or ""
    combined = " ".join(filter(None, [str(title), str(subtitle), str(description)]))
    return extract_keywords_from_text(combined, min_len=2)


def markets_to_subreddits(market_keywords: list[str]) -> list[str]:
    """Map a list of keywords to a union of subreddits."""
    mapping = get_keyword_to_subreddits()
    subreddits: set[str] = set()
    for kw in market_keywords:
        for s in mapping.get(kw.lower(), []):
            subreddits.add(s)
    return sorted(subreddits)


def match_markets_to_corners(
    client: Any,
    *,
    status: str = "open",
    limit: int = 200,
    min_close_ts: int | None = None,
) -> list[tuple[str, str, list[str], list[str]]]:
    """
    Fetch active markets from Kalshi and map each to (ticker, title, keywords, subreddits).
    Returns list of (market_ticker, market_title, keywords, subreddits).
    """
    results: list[tuple[str, str, list[str], list[str]]] = []
    cursor: str | None = None

    while True:
        kwargs: dict[str, Any] = {"limit": limit, "status": status}
        if cursor:
            kwargs["cursor"] = cursor
        if min_close_ts is not None:
            kwargs["min_close_ts"] = min_close_ts

        resp = client.get_markets(**kwargs)
        markets = getattr(resp, "markets", None) or getattr(resp, "market_list", []) or []
        if not isinstance(markets, list):
            markets = []

        for m in markets:
            ticker = getattr(m, "ticker", None) or ""
            title = getattr(m, "title", None) or getattr(m, "event_ticker", "") or ""
            if not ticker:
                continue
            keywords = extract_market_keywords(m)
            subreddits = markets_to_subreddits(keywords)
            if not keywords:
                # Still include market; scraper can use title-derived keywords
                keywords = extract_keywords_from_text(title)
                subreddits = markets_to_subreddits(keywords)
            results.append((ticker, title, keywords, subreddits))

        cursor = getattr(resp, "cursor", None)
        if not cursor or not markets:
            break

    return results
