"""
STEP 5: FastAPI server — deploy this on your edge device

Loads the quantized ONNX model and serves predictions via HTTP API.
Your UI (React, Flutter, mobile app) hits this endpoint.

Install:  pip install fastapi uvicorn onnxruntime transformers
Run:      uvicorn server:app --host 0.0.0.0 --port 8000

API Endpoints:
  POST /signal     ← main prediction endpoint
  GET  /health     ← health check
  GET  /           ← API info
"""

import numpy as np
import time
import os
from pathlib import Path
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("Run: pip install fastapi uvicorn")

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Run: pip install onnxruntime")

try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("Run: pip install transformers")


# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH  = os.environ.get("MODEL_PATH",  "crypto_signal_quantized.onnx")
TOKENIZER   = os.environ.get("TOKENIZER",   "crypto_student_model")
MAX_LEN     = 128
LABEL_MAP   = {0: "BUY", 1: "SELL", 2: "HOLD"}
# ──────────────────────────────────────────────────────────────────────────────


# ─── GLOBAL MODEL STATE ───────────────────────────────────────────────────────
class ModelState:
    session:   ort.InferenceSession = None
    tokenizer: AutoTokenizer        = None
    load_time: float                = 0.0

state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    print(f"Loading model from {MODEL_PATH}...")
    t0 = time.time()
    state.session   = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    state.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    state.load_time = time.time() - t0
    print(f"Model loaded in {state.load_time:.2f}s ✓")
    yield
    print("Shutting down...")


# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crypto Signal API",
    description="On-device crypto trading signal classifier (distilled TinyBERT)",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── SCHEMAS ──────────────────────────────────────────────────────────────────
class SignalRequest(BaseModel):
    coin:                 str   = Field(...,  example="BTC",              description="Coin symbol")
    price:                float = Field(...,  example=45000.0,            description="Current price in USD")
    price_change_24h_pct: float = Field(...,  example=-3.5,               description="24h price change %")
    fear_greed_index:     int   = Field(50,   ge=0, le=100,               description="Fear & Greed index 0-100")
    headline:             str   = Field("",   example="Bitcoin dips amid market correction", description="Top news headline")
    volume:               float = Field(0.0,  description="24h volume in USD (optional)")


class SignalResponse(BaseModel):
    signal:      str                  # BUY | SELL | HOLD
    confidence:  float                # probability of the predicted class
    probabilities: dict[str, float]   # {BUY: 0.1, SELL: 0.8, HOLD: 0.1}
    latency_ms:  float                # inference time in ms
    input_text:  str                  # what was fed to the model


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def build_input_text(req: SignalRequest) -> str:
    fg = req.fear_greed_index
    fg_label = (
        "extreme fear" if fg < 25 else
        "fear"         if fg < 45 else
        "neutral"      if fg < 55 else
        "greed"        if fg < 75 else
        "extreme greed"
    )
    headline = req.headline or "No major news today."
    return (
        f"{req.coin.upper()} price ${req.price:,.2f}, "
        f"24h change {req.price_change_24h_pct:.1f}%, "
        f"fear greed {fg} ({fg_label}). "
        f"News: {headline}"
    )


def predict(text: str) -> tuple[str, float, dict[str, float], float]:
    """Run ONNX inference and return (signal, confidence, probs, latency_ms)."""
    enc = state.tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    t0 = time.perf_counter()
    logits = state.session.run(
        ["logits"],
        {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        },
    )[0][0]
    latency_ms = (time.perf_counter() - t0) * 1000

    probs    = softmax(logits)
    pred_id  = int(np.argmax(probs))
    signal   = LABEL_MAP[pred_id]
    confidence = float(probs[pred_id])
    probs_dict = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(probs)}

    return signal, confidence, probs_dict, latency_ms


# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name":    "Crypto Signal API",
        "model":   MODEL_PATH,
        "status":  "running",
        "endpoints": {
            "POST /signal": "Get trading signal (BUY/SELL/HOLD)",
            "GET  /health": "Health check",
        }
    }


@app.get("/health")
def health():
    if state.session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status":    "ok",
        "model":     MODEL_PATH,
        "load_time": f"{state.load_time:.2f}s",
    }


@app.post("/signal", response_model=SignalResponse)
def get_signal(req: SignalRequest):
    if state.session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = build_input_text(req)
    signal, confidence, probs, latency_ms = predict(text)

    return SignalResponse(
        signal=signal,
        confidence=round(confidence, 4),
        probabilities=probs,
        latency_ms=round(latency_ms, 2),
        input_text=text,
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
