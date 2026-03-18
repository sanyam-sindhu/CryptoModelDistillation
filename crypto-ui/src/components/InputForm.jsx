import { useState } from "react";

const COINS = ["BTC", "ETH", "SOL", "BNB", "ADA", "XRP", "DOGE", "AVAX"];

const DEFAULT = {
  coin:        "BTC",
  price:       "45000",
  priceChange: "-3.5",
  fearGreed:   "35",
  headline:    "",
  volume:      "",
};

export default function InputForm({ onSubmit, loading }) {
  const [form, setForm] = useState(DEFAULT);

  const set = (field) => (e) => setForm((prev) => ({ ...prev, [field]: e.target.value }));

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(form);
  };

  const fearLabel = (v) => {
    const n = parseInt(v);
    if (n < 25) return { text: "Extreme Fear",  color: "#dc2626" };
    if (n < 45) return { text: "Fear",           color: "#ea580c" };
    if (n < 55) return { text: "Neutral",         color: "#ca8a04" };
    if (n < 75) return { text: "Greed",           color: "#16a34a" };
    return              { text: "Extreme Greed",  color: "#15803d" };
  };

  const fg = fearLabel(form.fearGreed);

  return (
    <form className="input-form" onSubmit={handleSubmit}>

      {/* Coin selector */}
      <div className="field">
        <label className="label">Coin</label>
        <div className="coin-grid">
          {COINS.map((c) => (
            <button
              key={c}
              type="button"
              className={`coin-btn ${form.coin === c ? "coin-btn--active" : ""}`}
              onClick={() => setForm((p) => ({ ...p, coin: c }))}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      {/* Price + Change row */}
      <div className="row-2">
        <div className="field">
          <label className="label">Price (USD)</label>
          <div className="input-wrap">
            <span className="input-prefix">$</span>
            <input
              className="input"
              type="number"
              step="0.01"
              min="0"
              required
              placeholder="45000"
              value={form.price}
              onChange={set("price")}
            />
          </div>
        </div>

        <div className="field">
          <label className="label">24h Change (%)</label>
          <div className="input-wrap">
            <input
              className={`input ${parseFloat(form.priceChange) >= 0 ? "input--pos" : "input--neg"}`}
              type="number"
              step="0.01"
              required
              placeholder="-3.5"
              value={form.priceChange}
              onChange={set("priceChange")}
            />
            <span className="input-suffix">%</span>
          </div>
        </div>
      </div>

      {/* Fear & Greed slider */}
      <div className="field">
        <label className="label">
          Fear &amp; Greed Index &nbsp;
          <span className="fg-badge" style={{ color: fg.color }}>
            {form.fearGreed} — {fg.text}
          </span>
        </label>
        <div className="slider-wrap">
          <span className="slider-label">😨 Fear</span>
          <input
            type="range"
            className="slider"
            min="0"
            max="100"
            value={form.fearGreed}
            onChange={set("fearGreed")}
          />
          <span className="slider-label">😏 Greed</span>
        </div>
        <div className="slider-track-labels">
          <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
        </div>
      </div>

      {/* Headline */}
      <div className="field">
        <label className="label">Top News Headline <span className="label-opt">(optional)</span></label>
        <input
          className="input"
          type="text"
          maxLength={200}
          placeholder="e.g. Bitcoin falls amid regulatory fears..."
          value={form.headline}
          onChange={set("headline")}
        />
      </div>

      {/* Volume */}
      <div className="field">
        <label className="label">24h Volume (USD) <span className="label-opt">(optional)</span></label>
        <div className="input-wrap">
          <span className="input-prefix">$</span>
          <input
            className="input"
            type="number"
            min="0"
            placeholder="1500000000"
            value={form.volume}
            onChange={set("volume")}
          />
        </div>
      </div>

      {/* Submit */}
      <button
        type="submit"
        className={`submit-btn ${loading ? "submit-btn--loading" : ""}`}
        disabled={loading}
      >
        {loading ? (
          <>
            <span className="spinner" /> Analyzing...
          </>
        ) : (
          <>◈ Get Signal</>
        )}
      </button>
    </form>
  );
}
