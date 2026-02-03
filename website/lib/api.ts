const API_BASE = "/api";

export async function fetchPrediction(
  text: string,
  msr_strength: number,
  msr_enabled: boolean,
) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, msr_strength, msr_enabled }),
  });
  if (!res.ok) throw new Error("Prediction failed");
  return res.json();
}

export async function fetchExplanation(
  text: string,
  aspect: string,
  msr_strength: number,
  signal?: AbortSignal,
) {
  const res = await fetch(`${API_BASE}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      aspect,
      methods: ["ig"], // lightweight by default
      msr_strength,
    }),
    signal, // Pass the abort signal for timeout control
  });
  if (!res.ok) throw new Error("Explanation failed");
  return res.json();
}

export async function fetchMetrics() {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error("Failed to fetch metrics");
  return res.json();
}

export async function fetchLogs() {
  const res = await fetch(`${API_BASE}/logs`);
  if (!res.ok) return [];
  return res.json();
}
