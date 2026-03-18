export default function StatusBar({ apiOk, onCheck }) {
  return (
    <div className="status-bar">
      <button className="status-check-btn" onClick={onCheck}>
        Check API
      </button>
      <div className="status-indicator">
        <span
          className="status-dot"
          style={{
            backgroundColor:
              apiOk === null  ? "#94a3b8" :
              apiOk           ? "#16a34a" : "#dc2626",
          }}
        />
        <span className="status-text">
          {apiOk === null  ? "Not checked" :
           apiOk           ? "API online"  : "API offline"}
        </span>
      </div>
    </div>
  );
}
