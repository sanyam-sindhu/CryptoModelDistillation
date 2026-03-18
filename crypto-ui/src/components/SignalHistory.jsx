const COLORS = {
  BUY:  { color: "#16a34a", bg: "#f0fdf4" },
  SELL: { color: "#dc2626", bg: "#fef2f2" },
  HOLD: { color: "#d97706", bg: "#fffbeb" },
};

const ARROWS = { BUY: "↑", SELL: "↓", HOLD: "→" };

export default function SignalHistory({ history }) {
  return (
    <div className="history-table-wrap">
      <table className="history-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Coin</th>
            <th>Signal</th>
            <th>Confidence</th>
            <th>BUY</th>
            <th>SELL</th>
            <th>HOLD</th>
            <th>Latency</th>
          </tr>
        </thead>
        <tbody>
          {history.map((entry, i) => {
            const meta  = COLORS[entry.signal] || COLORS.HOLD;
            const probs = entry.probabilities  || {};
            const time  = entry.timestamp instanceof Date
              ? entry.timestamp.toLocaleTimeString()
              : new Date(entry.timestamp).toLocaleTimeString();

            return (
              <tr key={i} className={i === 0 ? "history-row history-row--new" : "history-row"}>
                <td className="td-time">{time}</td>
                <td className="td-coin">{entry.coin}</td>
                <td>
                  <span
                    className="signal-pill"
                    style={{ color: meta.color, backgroundColor: meta.bg }}
                  >
                    {ARROWS[entry.signal]} {entry.signal}
                  </span>
                </td>
                <td className="td-conf" style={{ color: meta.color }}>
                  {Math.round((entry.confidence || 0) * 100)}%
                </td>
                <td className="td-prob td-prob--buy">
                  {Math.round((probs.BUY  || 0) * 100)}%
                </td>
                <td className="td-prob td-prob--sell">
                  {Math.round((probs.SELL || 0) * 100)}%
                </td>
                <td className="td-prob td-prob--hold">
                  {Math.round((probs.HOLD || 0) * 100)}%
                </td>
                <td className="td-latency">
                  {entry.latency_ms?.toFixed(1)}ms
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
