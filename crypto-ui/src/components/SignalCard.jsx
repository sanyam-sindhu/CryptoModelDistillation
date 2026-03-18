const SIGNAL_META = {
  BUY:  { emoji: "↑", label: "BUY",  color: "#16a34a", bg: "#f0fdf4", border: "#bbf7d0" },
  SELL: { emoji: "↓", label: "SELL", color: "#dc2626", bg: "#fef2f2", border: "#fecaca" },
  HOLD: { emoji: "→", label: "HOLD", color: "#d97706", bg: "#fffbeb", border: "#fde68a" },
};

function ProbBar({ label, value, color }) {
  const pct = Math.round(value * 100);
  return (
    <div className="prob-row">
      <span className="prob-label">{label}</span>
      <div className="prob-track">
        <div
          className="prob-fill"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="prob-pct">{pct}%</span>
    </div>
  );
}

export default function SignalCard({ result, loading }) {
  if (loading) {
    return (
      <div className="signal-card signal-card--loading">
        <div className="pulse-ring" />
        <p className="loading-text">Running on-device inference...</p>
        <p className="loading-sub">Model analyzing market data</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="signal-card signal-card--empty">
        <div className="empty-icon">◈</div>
        <p className="empty-text">Submit market data to get a signal</p>
        <p className="empty-sub">Prediction runs locally — no internet needed</p>
      </div>
    );
  }

  const meta  = SIGNAL_META[result.signal] || SIGNAL_META.HOLD;
  const probs = result.probabilities || {};

  return (
    <div
      className="signal-card signal-card--result"
      style={{ borderColor: meta.border, backgroundColor: meta.bg }}
    >
      {/* Big signal badge */}
      <div className="signal-badge" style={{ color: meta.color }}>
        <span className="signal-arrow">{meta.emoji}</span>
        <span className="signal-word">{meta.label}</span>
      </div>

      {/* Coin + time */}
      <p className="signal-meta">
        <strong>{result.coin}</strong> ·{" "}
        {result.timestamp instanceof Date
          ? result.timestamp.toLocaleTimeString()
          : new Date(result.timestamp).toLocaleTimeString()}
      </p>

      {/* Confidence */}
      <div className="confidence-wrap">
        <span className="confidence-label">Confidence</span>
        <span className="confidence-val" style={{ color: meta.color }}>
          {Math.round((result.confidence || 0) * 100)}%
        </span>
      </div>

      {/* Probability bars */}
      <div className="prob-section">
        <p className="prob-title">Signal Probabilities</p>
        <ProbBar label="BUY"  value={probs.BUY  || 0} color="#16a34a" />
        <ProbBar label="SELL" value={probs.SELL || 0} color="#dc2626" />
        <ProbBar label="HOLD" value={probs.HOLD || 0} color="#d97706" />
      </div>

      {/* Latency */}
      <div className="latency-badge">
        ⚡ {result.latency_ms?.toFixed(1)}ms inference · on-device
      </div>

      {/* Input text used */}
      <details className="input-details">
        <summary>Model input</summary>
        <p className="input-text-preview">{result.input_text}</p>
      </details>
    </div>
  );
}
