import { useState, useCallback } from "react";
import InputForm    from "./components/InputForm.jsx";
import SignalCard   from "./components/SignalCard.jsx";
import SignalHistory from "./components/SignalHistory.jsx";
import StatusBar    from "./components/StatusBar.jsx";
import "./App.css";

const API_BASE = "/api"; // proxied to http://localhost:8000 in dev

export default function App() {
  const [result,  setResult]  = useState(null);   // latest signal response
  const [history, setHistory] = useState([]);      // past signals
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);
  const [apiOk,   setApiOk]   = useState(null);   // null=unchecked, true/false

  // ── Health check ────────────────────────────────────────────────────────────
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      setApiOk(res.ok);
    } catch {
      setApiOk(false);
    }
  }, []);

  // ── Signal fetch ─────────────────────────────────────────────────────────────
  const fetchSignal = useCallback(async (formData) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/signal`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          coin:                 formData.coin,
          price:                parseFloat(formData.price),
          price_change_24h_pct: parseFloat(formData.priceChange),
          fear_greed_index:     parseInt(formData.fearGreed),
          headline:             formData.headline,
          volume:               parseFloat(formData.volume || 0),
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const entry = { ...data, coin: formData.coin, timestamp: new Date() };
      setResult(entry);
      setHistory((prev) => [entry, ...prev].slice(0, 20)); // keep last 20
      setApiOk(true);
    } catch (e) {
      setError(e.message);
      setApiOk(false);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">◈</span>
            <span className="logo-text">CryptoSignal <span className="logo-ai">AI</span></span>
          </div>
          <StatusBar apiOk={apiOk} onCheck={checkHealth} />
        </div>
      </header>

      {/* ── Main ── */}
      <main className="main">
        <div className="grid">
          {/* Left column — form */}
          <section className="card form-section">
            <h2 className="section-title">Market Input</h2>
            <p className="section-sub">Enter current market data to get a trading signal</p>
            <InputForm onSubmit={fetchSignal} loading={loading} />
            {error && (
              <div className="error-box">
                <span className="error-icon">⚠</span> {error}
              </div>
            )}
          </section>

          {/* Right column — result */}
          <section className="card result-section">
            <h2 className="section-title">Signal Result</h2>
            <p className="section-sub">On-device prediction — no data leaves your device</p>
            <SignalCard result={result} loading={loading} />
          </section>
        </div>

        {/* History */}
        {history.length > 0 && (
          <section className="card history-section">
            <h2 className="section-title">Signal History</h2>
            <SignalHistory history={history} />
          </section>
        )}
      </main>

      <footer className="footer">
        Powered by distilled TinyBERT · Runs fully on-device · No cloud required
      </footer>
    </div>
  );
}
