# Public Sentiment Analyzer

A research tool that monitors niche subreddits (and optional web sources) for discussion about events listed on Kalshi. It uses an LLM to analyze sentiment and estimate implied probabilities from forum content, and logs analysis results for comparison with market prices.

**Use for research and education only. Prediction markets are volatile; this tool does not provide financial advice.**

## Requirements

- Python 3.10+
- API keys: Kalshi (read market data), Reddit (PRAW), OpenAI

## Setup

1. **Clone / enter project and install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Kalshi API**

   - Log in at [Kalshi](https://kalshi.com) → Account & Security → API Keys → Create Key.
   - Download the private key (`.key` or `.pem`); it cannot be retrieved later.
   - Set `KALSHI_API_KEY_ID` and either `KALSHI_PRIVATE_KEY_PATH` (path to file) or `KALSHI_PRIVATE_KEY` (raw PEM string).

3. **Reddit API**

   - Create an application at [Reddit Apps](https://www.reddit.com/prefs/apps) (script type).
   - Set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT` (e.g. `SentimentAnalyzer/1.0 by your_username`).

4. **OpenAI**

   - Set `OPENAI_API_KEY` in your environment or `.env`.

5. **Environment**

   - Copy `.env.example` to `.env` and fill in the values above.
   - Optional: `DRY_RUN=true`, `MIN_DELTA=0.10`, `CONFIDENCE_THRESHOLD=0.75`, `POLL_INTERVAL_SEC=120`, `MAX_CONTRACTS_PER_TRADE=100`.

## Usage

- **Analysis only (default):** Logs sentiment and implied-probability estimates to SQLite; no external actions.

  ```bash
  python main.py
  ```

- **With simulation logging:** Set `DRY_RUN=false` in `.env` to also record simulation outcomes (for backtesting and research).

The analyzer runs every `POLL_INTERVAL_SEC` seconds: fetches open Kalshi markets for reference, maps them to subreddits via keywords, scrapes new Reddit threads, runs the LLM analyst per market, and logs sentiment, implied probability, and recommendation. Results are stored locally for comparison with market prices.

## Project layout

- `main.py` — Entry point; DB init; main loop (matcher → scraper → brain → output).
- `scraper.py` — Reddit scraper (PRAW), dedupe and thread grouping; optional web fetcher.
- `matcher.py` — Fetches active markets from Kalshi and maps them to subreddits by keywords.
- `brain.py` — LLM (GPT-4o-mini) analyst prompt and structured output (probability, recommendation).
- `trading.py` — Kalshi API client (market data, balance for sizing logic), simulation logging to SQLite.

## Data and safety

- SQLite DB (default `kalshi_bot.sqlite`): `seen_posts` (dedupe), `trades` (analysis/simulation log).
- Reddit and Kalshi rate limits are respected; analysis-only mode is the default.
- Do not commit `.env` or `.pem`/`.key` files; they are in `.gitignore`.
