"""
STEP 2: Teacher labeling with Claude (knowledge distillation)
- Claude acts as the expert teacher model
- Labels each sample BUY / SELL / HOLD
- TinyBERT student learns to mimic Claude's labels

Run: python step2_label_data.py
"""

import anthropic
import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
INPUT_FILE        = "raw_data.json"
OUTPUT_FILE       = "labeled_data.json"
MODEL             = "claude-sonnet-4-6"
BATCH_SIZE        = 50
MAX_SAMPLES       = 10000
RETRY_DELAY       = 5
# ──────────────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Few-shot examples calibrate Claude's output distribution
# These are drawn from real data edge cases across BUY / SELL / HOLD
FEW_SHOT_EXAMPLES = """Here are example labels to calibrate your responses:

Example 1 — BUY (strong upward momentum):
  Coin: BTC | 24h Change: +12.2% | Fear/Greed: 6 (extreme fear) | Volume: $128B
  Label: BUY
  Reason: +12% move off extreme fear bottom = strong reversal signal

Example 2 — SELL (crash with panic volume):
  Coin: BTC | 24h Change: -14.1% | Fear/Greed: 9 (extreme fear) | Volume: $142B
  Label: SELL
  Reason: -14% with record volume = capitulation / distribution, exit signal

Example 3 — BUY (moderate move + extreme fear bounce):
  Coin: ETH | 24h Change: +5.2% | Fear/Greed: 9 (extreme fear) | Volume: $20B
  Label: BUY
  Reason: +5% bounce from extreme fear = high-probability reversal

Example 4 — SELL (dump + fear confirmation):
  Coin: ETH | 24h Change: -6.25% | Fear/Greed: 16 (extreme fear) | Volume: $37B
  Label: SELL
  Reason: -6% with fear = sellers in control, momentum is down

Example 5 — HOLD (small move, no strong signal):
  Coin: BTC | 24h Change: +0.5% | Fear/Greed: 24 (fear) | Volume: $53B
  Label: HOLD
  Reason: Flat price + no sentiment extreme = no actionable signal

Example 6 — HOLD (moderate drop but fear already priced in):
  Coin: SOL | 24h Change: -2.9% | Fear/Greed: 23 (fear) | Volume: $5.8B
  Label: HOLD
  Reason: Small drop in fear zone — could go either way, wait for confirmation

Example 7 — SELL (moderate drop + greed reversal):
  Coin: BTC | 24h Change: -2.5% | Fear/Greed: 49 (neutral) | Volume: $47B
  Label: SELL
  Reason: Market was at neutral/greed, now dropping with volume = distribution

Example 8 — BUY (strong move + greed momentum):
  Coin: ETH | 24h Change: +7.4% | Fear/Greed: 48 (neutral→greed) | Volume: $35B
  Label: BUY
  Reason: +7% with rising sentiment = momentum breakout

Now label the following sample. Reply with ONLY one word: BUY, SELL, or HOLD."""


def build_prompt(sample: dict) -> str:
    fg = sample["fear_greed_index"]
    fg_label = (
        "extreme fear" if fg < 25 else
        "fear"         if fg < 45 else
        "neutral"      if fg < 55 else
        "greed"        if fg < 75 else
        "extreme greed"
    )
    return f"""{FEW_SHOT_EXAMPLES}

Coin: {sample['coin'].upper()}
Date: {sample['date']}
Price: ${sample['price']:,}
24h Change: {sample['price_change_24h_pct']}%
Volume: ${sample['volume']:,.0f}
Fear & Greed: {fg}/100 ({fg_label})
Headline: {sample['headline']}

Label:"""


def label_sample(sample: dict, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=5,
                messages=[{"role": "user", "content": build_prompt(sample)}],
            )
            label = response.content[0].text.strip().upper()
            if label in ("BUY", "SELL", "HOLD"):
                return label
            # handle cases like "BUY." or "HOLD\n"
            for word in ("BUY", "SELL", "HOLD"):
                if word in label:
                    return word
            print(f"  Unexpected: '{label}', defaulting HOLD")
            return "HOLD"

        except anthropic.RateLimitError:
            wait = RETRY_DELAY * (attempt + 1)
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  Error attempt {attempt+1}: {e}")
            time.sleep(2)

    return "HOLD"


def main():
    print("=" * 50)
    print("STEP 2: Teacher labeling (Claude distillation)")
    print("=" * 50)

    with open(INPUT_FILE) as f:
        samples = json.load(f)

    print(f"\nLoaded {len(samples)} samples from {INPUT_FILE}")

    labeled = []
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE) as f:
            labeled = json.load(f)
        print(f"Resuming from checkpoint: {len(labeled)} already labeled")

    to_label = samples[len(labeled):MAX_SAMPLES]
    print(f"Samples to label: {len(to_label)}")

    tokens_per_sample = 600  # few-shot prompt is larger
    cost_estimate = (len(to_label) * tokens_per_sample / 1_000_000) * 3.0  # sonnet $3/1M
    print(f"Estimated cost: ${cost_estimate:.3f}\n")

    for i, sample in enumerate(to_label):
        label = label_sample(sample)
        sample["label"] = label
        labeled.append(sample)

        if (i + 1) % 10 == 0:
            buy  = sum(1 for s in labeled if s["label"] == "BUY")
            sell = sum(1 for s in labeled if s["label"] == "SELL")
            hold = sum(1 for s in labeled if s["label"] == "HOLD")
            print(f"  [{i+1}/{len(to_label)}] BUY={buy} SELL={sell} HOLD={hold}")

        if (i + 1) % BATCH_SIZE == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(labeled, f, indent=2)
            print(f"  Checkpoint saved ({len(labeled)} samples)")

        time.sleep(0.15)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(labeled, f, indent=2)

    buy  = sum(1 for s in labeled if s["label"] == "BUY")
    sell = sum(1 for s in labeled if s["label"] == "SELL")
    hold = sum(1 for s in labeled if s["label"] == "HOLD")
    total = len(labeled)

    print(f"\n✓ Labeled {total} samples saved to {OUTPUT_FILE}")
    print(f"  BUY={buy} ({buy/total*100:.1f}%)  SELL={sell} ({sell/total*100:.1f}%)  HOLD={hold} ({hold/total*100:.1f}%)")
    print("  Next: run step3_train_student.ipynb")


if __name__ == "__main__":
    main()
