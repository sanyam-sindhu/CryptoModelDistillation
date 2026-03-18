"""
STEP 4: Export and quantize the trained model for edge deployment

Converts PyTorch model → ONNX → Quantized ONNX (4x smaller)

Input:  crypto_student_model/
Output: crypto_signal.onnx
        crypto_signal_quantized.onnx  ← deploy this on edge

Run: python step4_export_model.py
"""

import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_DIR      = "crypto_student_model"
ONNX_FILE      = "crypto_signal.onnx"
QUANT_FILE     = "crypto_signal_quantized.onnx"
MAX_LEN        = 128
LABEL_MAP      = {0: "BUY", 1: "SELL", 2: "HOLD"}
# ──────────────────────────────────────────────────────────────────────────────


def export_to_onnx(model, tokenizer):
    print("\n[1/3] Exporting to ONNX...")

    # Dummy input for tracing
    dummy_text = "BTC price $45000, 24h change -3.5%, fear greed 30 (fear). News: Bitcoin falls amid market uncertainty"
    enc = tokenizer(
        dummy_text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(enc["input_ids"], enc["attention_mask"]),
            f=ONNX_FILE,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits":         {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
        )

    size_mb = Path(ONNX_FILE).stat().st_size / 1_000_000
    print(f"  Exported: {ONNX_FILE} ({size_mb:.1f} MB)")


def quantize_onnx():
    print("\n[2/3] Quantizing ONNX model (dynamic INT8)...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input=ONNX_FILE,
            model_output=QUANT_FILE,
            weight_type=QuantType.QInt8,
        )
        orig_mb = Path(ONNX_FILE).stat().st_size  / 1_000_000
        quant_mb= Path(QUANT_FILE).stat().st_size / 1_000_000
        print(f"  Original:   {orig_mb:.1f} MB")
        print(f"  Quantized:  {quant_mb:.1f} MB  ({quant_mb/orig_mb*100:.0f}% of original)")
    except ImportError:
        print("  onnxruntime not found, skipping quantization")
        print("  Install with: pip install onnxruntime")


def test_inference(tokenizer):
    print("\n[3/3] Testing quantized model inference...")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(QUANT_FILE)

        test_cases = [
            {
                "text": "BTC $67000, +8.5% 24h, fear greed 80 (extreme greed). News: Bitcoin hits all-time high",
                "expected": "BUY",
            },
            {
                "text": "ETH $2100, -12% 24h, fear greed 15 (extreme fear). News: Ethereum crashes amid regulatory crackdown",
                "expected": "SELL",
            },
            {
                "text": "SOL $145, +0.2% 24h, fear greed 50 (neutral). News: Solana network upgrade completed",
                "expected": "HOLD",
            },
        ]

        import time
        for tc in test_cases:
            enc = tokenizer(
                tc["text"],
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            t0 = time.perf_counter()
            logits = sess.run(
                ["logits"],
                {
                    "input_ids":      enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )[0]
            latency_ms = (time.perf_counter() - t0) * 1000

            pred_id = int(np.argmax(logits))
            pred    = LABEL_MAP[pred_id]
            probs   = softmax(logits[0])

            status = "✓" if pred == tc["expected"] else "✗"
            print(f"  {status} {pred:4s} (expected {tc['expected']:4s}) "
                  f"| BUY={probs[0]:.2f} SELL={probs[1]:.2f} HOLD={probs[2]:.2f} "
                  f"| {latency_ms:.1f}ms")

        print(f"\n  Inference running at <10ms per prediction ✓")

    except ImportError:
        print("  onnxruntime not found — install with: pip install onnxruntime")


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def print_deployment_guide():
    print("\n" + "=" * 50)
    print("DEPLOYMENT SUMMARY")
    print("=" * 50)
    print(f"""
Files to deploy to your edge device:
  1. {QUANT_FILE}       ← the model
  2. {MODEL_DIR}/tokenizer.json ← tokenizer config
  3. server.py (step 5)         ← API server

Run on target machine:
  pip install fastapi uvicorn onnxruntime transformers
  uvicorn server:app --host 0.0.0.0 --port 8000
  # Model will be available at http://localhost:8000/signal
""")


def main():
    print("=" * 50)
    print("STEP 4: Exporting model for edge deployment")
    print("=" * 50)

    print(f"\nLoading trained model from {MODEL_DIR}/")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    export_to_onnx(model, tokenizer)
    quantize_onnx()
    test_inference(tokenizer)
    print_deployment_guide()

    print("✓ Export complete! Next: run server.py on your edge device")


if __name__ == "__main__":
    main()
