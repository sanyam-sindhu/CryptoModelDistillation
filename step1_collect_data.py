"""
STEP 1: Collect crypto data from free APIs
- CoinGecko: price + volume data
- CryptoPanic: news headlines
- Alternative.me: Fear & Greed index (no key needed)

Run: python step1_collect_data.py
Output: raw_data.json
"""

import requests
import json
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()  # reads from .env file

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CRYPTOPANIC_API_KEY = os.environ["CRYPTOPANIC_API_KEY"]
COINGECKO_API_KEY   = os.environ.get("COINGECKO_API_KEY", "")
COINS               = ["bitcoin", "ethereum", "solana"]
DAYS_HISTORY        = 180                       # how many days of price data
OUTPUT_FILE         = "raw_data.json"
# ──────────────────────────────────────────────────────────────────────────────


def get_fear_greed():
    """Fear & Greed index — no API key needed."""
    url = "https://api.alternative.me/fng/?limit=90&format=json"
    res = requests.get(url, timeout=10)
    data = res.json()["data"]
    return {item["timestamp"]: int(item["value"]) for item in data}


def get_price_data(coin: str):
    """Get 90 days of daily OHLCV data from CoinGecko free tier."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": "usd", "days": DAYS_HISTORY, "interval": "daily"}
    headers = {"x-cg-demo-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
    res = requests.get(url, params=params, headers=headers, timeout=10)
    if res.status_code == 429:
        print("  Rate limited by CoinGecko, waiting 60s...")
        time.sleep(60)
        res = requests.get(url, params=params, headers=headers, timeout=10)
    data = res.json()
    prices  = data.get("prices", [])
    volumes = data.get("total_volumes", [])
    result  = []
    for i in range(1, len(prices)):
        prev_price = prices[i-1][1]
        curr_price = prices[i][1]
        pct_change = ((curr_price - prev_price) / prev_price) * 100 if prev_price else 0
        result.append({
            "timestamp": int(prices[i][0] / 1000),
            "date":      datetime.fromtimestamp(prices[i][0] / 1000).strftime("%Y-%m-%d"),
            "price":     round(curr_price, 2),
            "price_change_24h_pct": round(pct_change, 2),
            "volume":    round(volumes[i][1], 0) if i < len(volumes) else 0,
        })
    return result


def get_news(coin_symbol: str, pages: int = 5):
    """Get latest crypto news from CryptoPanic."""
    symbol_map = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL"}
    symbol = symbol_map.get(coin_symbol, "BTC")
    headlines = []
    for page in range(1, pages + 1):
        url = "https://cryptopanic.com/api/developer/v2/posts/"
        params = {
            "auth_token": CRYPTOPANIC_API_KEY,
            "currencies": symbol,
            "public":     "true",
            "page":       page,
        }
        res = requests.get(url, params=params, timeout=10)
        if res.status_code != 200:
            print(f"  CryptoPanic error {res.status_code}, skipping page {page}")
            break
        posts = res.json().get("results", [])
        for post in posts:
            headlines.append({
                "title":     post.get("title", ""),
                "published": post.get("published_at", "")[:10],
                "source":    post.get("source", {}).get("title", ""),
            })
        time.sleep(1)  # be polite to the API
    return headlines


def build_samples(coin: str, prices: list, fear_greed: dict, news: list):
    """Combine price + sentiment + fear/greed into labeled samples."""
    samples = []
    news_by_date = {}
    for n in news:
        date = n["published"][:10]
        news_by_date.setdefault(date, []).append(n["title"])

    for entry in prices:
        date      = entry["date"]
        fg_index  = fear_greed.get(str(entry["timestamp"]), 50)
        headlines = news_by_date.get(date, ["No major news today."])
        headline  = headlines[0]  # use top headline of the day

        samples.append({
            "coin":                 coin,
            "date":                 date,
            "price":                entry["price"],
            "price_change_24h_pct": entry["price_change_24h_pct"],
            "volume":               entry["volume"],
            "fear_greed_index":     fg_index,
            "headline":             headline,
            "label":                None,  # will be filled in step 2
        })
    return samples


def main():
    print("=" * 50)
    print("STEP 1: Collecting crypto data")
    print("=" * 50)

    print("\n[1/3] Fetching Fear & Greed index...")
    fear_greed = get_fear_greed()
    print(f"  Got {len(fear_greed)} days of data")

    all_samples = []

    for coin in COINS:
        print(f"\n[2/3] Fetching price data for {coin}...")
        prices = get_price_data(coin)
        print(f"  Got {len(prices)} price entries")
        time.sleep(2)  # avoid CoinGecko rate limit

        print(f"[3/3] Fetching news for {coin}...")
        news = get_news(coin, pages=3)
        print(f"  Got {len(news)} headlines")
        time.sleep(1)

        samples = build_samples(coin, prices, fear_greed, news)
        all_samples.extend(samples)
        print(f"  Built {len(samples)} samples for {coin}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\n✓ Saved {len(all_samples)} total samples to {OUTPUT_FILE}")
    print("  Next: run step2_label_data.py")


if __name__ == "__main__":
    main()
