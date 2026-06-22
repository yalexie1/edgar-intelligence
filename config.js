// API base URL — auto-detects local dev vs production.
// After deploying to Render, replace the placeholder below with your
// actual Render service URL (e.g. https://edgar-rag-api.onrender.com).
window.EDGAR_API_URL = (
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1" ||
  window.location.protocol === "file:"
) ? "http://127.0.0.1:8000"
  : "https://edgar-intelligence.onrender.com";
