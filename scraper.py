"""
Modular scraper for Reddit (and optional web) to collect content for signal analysis.
Deduplicates via SQLite; returns content grouped by thread for LLM context.
"""

import re
import sqlite3
import time
from typing import Any

import praw


# Default stopwords for keyword filtering (minimal set)
STOPWORDS = frozenset(
    {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "is", "it", "be", "by", "as"}
)


class SignalScraper:
    """
    Class-based scraper that watches a list of subreddits for keywords,
    deduplicates via SQLite, and returns new content grouped by thread.
    """

    def __init__(
        self,
        subreddits: list[str],
        keywords: list[str],
        db_path: str,
        *,
        time_filter: str = "day",
        search_limit: int = 100,
        comments_limit: int = 50,
        delay_between_subreddits: float = 1.5,
        reddit_client_id: str | None = None,
        reddit_client_secret: str | None = None,
        reddit_user_agent: str | None = None,
    ) -> None:
        self.subreddits = [s.lstrip("r/").lower() for s in subreddits]
        self.keywords = keywords
        self.db_path = db_path
        self.time_filter = time_filter
        self.search_limit = search_limit
        self.comments_limit = comments_limit
        self.delay_between_subreddits = delay_between_subreddits
        self._ensure_seen_posts_table()
        self._reddit: praw.Reddit | None = None
        self._reddit_client_id = reddit_client_id
        self._reddit_client_secret = reddit_client_secret
        self._reddit_user_agent = reddit_user_agent

    def _reddit_client(self) -> praw.Reddit:
        if self._reddit is None:
            if not all([self._reddit_client_id, self._reddit_client_secret, self._reddit_user_agent]):
                raise ValueError(
                    "Reddit credentials required: reddit_client_id, reddit_client_secret, reddit_user_agent"
                )
            self._reddit = praw.Reddit(
                client_id=self._reddit_client_id,
                client_secret=self._reddit_client_secret,
                user_agent=self._reddit_user_agent,
            )
        return self._reddit

    def _ensure_seen_posts_table(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_posts (
                    post_id TEXT PRIMARY KEY,
                    seen_at INTEGER NOT NULL
                )
                """
            )

    def _is_seen(self, post_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM seen_posts WHERE post_id = ?",
                (post_id,),
            ).fetchone()
        return row is not None

    def _mark_seen(self, post_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO seen_posts (post_id, seen_at) VALUES (?, ?)",
                (post_id, int(time.time())),
            )

    def _thread_from_submission(self, submission: Any) -> dict[str, Any]:
        comments: list[dict[str, Any]] = []
        try:
            submission.comment_sort = "top"
            submission.comments.replace_more(limit=0)
            for c in submission.comments.list()[: self.comments_limit]:
                if getattr(c, "body", None) is None:
                    continue
                comments.append({
                    "author": getattr(c.author, "name", "[deleted]"),
                    "body": (c.body or "")[:2000],
                    "score": getattr(c, "score", 0),
                })
        except Exception:
            pass
        return {
            "thread_id": submission.id,
            "title": (submission.title or "")[:500],
            "body": (getattr(submission, "selftext", None) or "")[:5000],
            "subreddit": str(submission.subreddit).lower(),
            "url": getattr(submission, "url", ""),
            "permalink": getattr(submission, "permalink", ""),
            "created_utc": getattr(submission, "created_utc", 0),
            "comments": comments,
        }

    def scrape(self) -> list[dict[str, Any]]:
        """
        Search each subreddit for each keyword; deduplicate; return new threads
        grouped as list[dict] for the brain (one dict per thread).
        """
        reddit = self._reddit_client()
        collected: dict[str, dict[str, Any]] = {}  # post_id -> thread

        for sub_name in self.subreddits:
            try:
                sub = reddit.subreddit(sub_name)
                for keyword in self.keywords:
                    try:
                        for submission in sub.search(
                            keyword,
                            time_filter=self.time_filter,
                            limit=self.search_limit,
                        ):
                            if submission.id in collected:
                                continue
                            if self._is_seen(submission.id):
                                continue
                            collected[submission.id] = self._thread_from_submission(submission)
                    except Exception:
                        continue
                time.sleep(self.delay_between_subreddits)
            except Exception:
                continue

        # Mark all collected as seen and return as list (new content only)
        threads = list(collected.values())
        for t in threads:
            self._mark_seen(t["thread_id"])
        return threads


def extract_keywords_from_text(text: str, min_len: int = 2) -> list[str]:
    """Normalize and extract tokens; remove stopwords. Used by matcher."""
    if not text:
        return []
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [w for w in normalized.split() if len(w) >= min_len and w not in STOPWORDS]
    return list(dict.fromkeys(tokens))  # preserve order, dedupe


# --- Optional web scraping (httpx + BeautifulSoup, robots.txt) ---

def fetch_urls(
    urls: list[str],
    db_path: str | None = None,
    delay_per_domain: float = 1.0,
    max_content_chars: int = 50000,
) -> list[dict[str, Any]]:
    """
    Fetch URLs with httpx and BeautifulSoup; optionally dedupe via SQLite.
    Respects robots.txt via urllib.robotparser. Returns list of {url, title, text}.
    """
    import urllib.robotparser
    from urllib.parse import urlparse

    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    seen: set[str] = set()
    if db_path:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS seen_urls (url TEXT PRIMARY KEY, seen_at INTEGER)"
            )
            for row in conn.execute("SELECT url FROM seen_urls").fetchall():
                seen.add(row[0])

    results: list[dict[str, Any]] = []
    last_domain: str | None = None

    for url in urls:
        try:
            parsed = urlparse(url)
            domain = f"{parsed.scheme}://{parsed.netloc}/"
            if domain != last_domain:
                rp = urllib.robotparser.RobotsFileParser()
                rp.set_url(parsed.scheme + "://" + parsed.netloc + "/robots.txt")
                try:
                    rp.read()
                except Exception:
                    pass
                if not rp.can_fetch("SentimentAnalyzer", url):
                    continue
                if last_domain is not None:
                    time.sleep(delay_per_domain)
                last_domain = domain

            if url in seen:
                continue
            with httpx.Client(follow_redirects=True, timeout=15.0) as client:
                r = client.get(url)
                r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)[:max_content_chars]
            title = ""
            if soup.title:
                title = (soup.title.string or "").strip()[:500]
            results.append({"url": url, "title": title, "text": text})
            seen.add(url)
            if db_path:
                with sqlite3.connect(db_path) as conn:
                    conn.execute(
                        "INSERT OR IGNORE INTO seen_urls (url, seen_at) VALUES (?, ?)",
                        (url, int(time.time())),
                    )
        except Exception:
            continue
    return results
