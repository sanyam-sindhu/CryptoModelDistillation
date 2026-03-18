# Crypto Distillation

Train a tiny crypto trading signal classifier using knowledge distillation. Claude acts as the teacher — it reads market data and labels each sample BUY, SELL, or HOLD. TinyBERT learns to mimic those labels and runs the final inference in under 10ms with no cloud dependency.

---

## How it works

```
CoinGecko + CryptoPanic + Fear&Greed
            ↓
      raw_data.json        (Step 1)
            ↓
   Claude labels each sample   (Step 2)
            ↓
      labeled_data.json
            ↓
   TinyBERT fine-tuned         (Step 3)
            ↓
   crypto_student_model/
            ↓
   Export → ONNX → Quantized   (Step 4)
            ↓
   FastAPI server + React UI   (Step 5-6)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your_key
CRYPTOPANIC_API_KEY=your_key
COINGECKO_API_KEY=your_key   # optional, free tier works without it
```

---

## Running the Pipeline

**Step 1 — Collect data**
```bash
python step1_collect_data.py
```
Pulls price, volume, Fear & Greed index, and news headlines for BTC, ETH, SOL from the last 180 days. Outputs `raw_data.json`.

**Step 2 — Label with Claude**
```bash
python step2_label_data.py
```
Sends each sample to Claude with a structured prompt. Claude returns BUY, SELL, or HOLD. Checkpoints every 50 samples so you can resume if it breaks. Costs ~$0.90 per 500 samples. Outputs `labeled_data.json`.

**Step 3 — Train student model**
```bash
python step3_train_student.py
```
Fine-tunes TinyBERT (~14M params) on the labeled data using a distillation loss — part cross-entropy on the hard labels, part KL divergence on soft targets. Runs on CPU but a GPU or Colab speeds it up significantly. Outputs `crypto_student_model/`.

> For Colab: upload `labeled_data.json`, open `step3_train_student.ipynb`, run all cells, download `crypto_student_model/`.

**Step 4 — Export to ONNX**
```bash
python step4_export_model.py
```
Converts the PyTorch model to ONNX, then quantizes to INT8. Goes from ~55MB down to ~14MB. Runs 3 test predictions at the end to verify everything works. Outputs `crypto_signal.onnx` and `crypto_signal_quantized.onnx`.

**Step 5 — Run the API server**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```
Loads the quantized ONNX model and serves predictions over HTTP. No external calls at inference time.

**Step 6 — Run the UI**
```bash
cd crypto-ui
npm install
npm run dev
```
Opens at `http://localhost:5173`. Make sure the server is running first.

---

## API

```bash
curl -X POST http://localhost:8000/signal \
  -H "Content-Type: application/json" \
  -d '{"coin":"BTC","price":67000,"price_change_24h_pct":8.5,"fear_greed_index":80,"headline":"Bitcoin hits all-time high"}'
```

```json
{
  "signal": "BUY",
  "confidence": 0.91,
  "probabilities": { "BUY": 0.91, "SELL": 0.04, "HOLD": 0.05 },
  "latency_ms": 4.2
}
```

---

## Deployment

Copy these to any machine and run:

```bash
pip install fastapi uvicorn onnxruntime transformers
uvicorn server:app --host 0.0.0.0 --port 8000
```

Files needed:
```
crypto_signal_quantized.onnx
crypto_student_model/
server.py
```

---

## Data Sources

- [CoinGecko](https://www.coingecko.com/en/api) — price & volume
- [CryptoPanic](https://cryptopanic.com/developers/api/) — news headlines
- [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/) — Fear & Greed index (no key needed)
