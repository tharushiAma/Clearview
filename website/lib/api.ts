const API_BASE = "/api";

export async function predict(req: { text: string; msrEnabled: boolean; msrStrength: number }) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: req.text, msr_strength: req.msrStrength, msr_enabled: req.msrEnabled }),
  });
  if (!res.ok) throw new Error("Prediction failed");
  const data = await res.json();

  // Transform backend response to frontend types
  const predictions = (data.aspects || []).map((a: any) => ({
    aspect: a.name,
    label: a.label,
    confidence: a.confidence,
    topTokens: a.top_tokens || [],
    msrChanged: a.changed_by_msr || false,
    before: a.before ? { label: a.before.label, confidence: a.before.confidence } : undefined,
    after: a.after ? { label: a.after.label, confidence: a.after.confidence } : undefined
  }));

  const result = {
    predictions,
    conflictProbability: data.conflict_prob || 0,
    mixedSentimentDetected: (data.conflict_prob || 0) > 0.5,
  } as any;

  // Add before/after if present in the first aspect (indicates MSR comparison)
  if (data.aspects?.[0]?.before && data.aspects?.[0]?.after) {
    result.before = data.aspects.map((a: any) => ({
      aspect: a.name,
      label: a.before.label,
      confidence: a.before.confidence,
      topTokens: [],
      msrChanged: false
    }));
    result.after = data.aspects.map((a: any) => ({
      aspect: a.name,
      label: a.after.label,
      confidence: a.after.confidence,
      topTokens: [],
      msrChanged: a.changed_by_msr
    }));
  }

  return result;
}

export async function fetchPrediction(
  text: string,
  msr_strength: number,
  msr_enabled: boolean,
) {
  return predict({ text, msrEnabled: msr_enabled, msrStrength: msr_strength });
}

export async function explain(req: { text: string; aspect: string; methods: string[]; msrEnabled: boolean; msrStrength: number; signal?: AbortSignal }) {
  const res = await fetch(`${API_BASE}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: req.text,
      aspect: req.aspect,
      methods: req.methods,
      msr_strength: req.msrStrength,
      msr_enabled: req.msrEnabled,
    }),
    signal: req.signal,
  });
  if (!res.ok) throw new Error("Explanation failed");
  const data = await res.json();

  // Transform backend response into ExplanationBundle
  const explanations: any[] = [];
  const aspects = data.aspects || {};

  Object.keys(aspects).forEach((aspName) => {
    const aspData = aspects[aspName];
    
    // Handle IG
    if (aspData.ig_aspect) {
      explanations.push({
        aspect: aspName,
        method: "ig",
        tokens: (aspData.ig_aspect.top_tokens || []).map((t: any) => ({
          token: t[0],
          attribution: t[1],
          msrDelta: (aspData.msr_delta?.top_tokens || []).find((m: any) => m[0] === t[0])?.[1] || 0
        }))
      });
    }

    // Handle LIME/SHAP if they were added (placeholder for future expansion)
    if (aspData.lime_aspect) {
      explanations.push({
        aspect: aspName,
        method: "lime",
        tokens: (aspData.lime_aspect.top_tokens || []).map((t: any) => ({
          token: t[0],
          attribution: t[1]
        }))
      });
    }
  });

  return {
    text: data.text || req.text,
    explanations,
    rawJson: data
  };
}

export async function fetchExplanation(
  text: string,
  aspect: string,
  msr_strength: number,
  signal?: AbortSignal,
) {
  return explain({ text, aspect, methods: ["ig"], msrEnabled: true, msrStrength: msr_strength, signal });
}

export async function predictBulk(reviews: string[], msrEnabled = true) {
  const res = await fetch(`${API_BASE}/predict-bulk`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviews, msr_enabled: msrEnabled }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Bulk prediction failed" }));
    throw new Error(err.error || "Bulk prediction failed");
  }
  return res.json();
}

export async function fetchMetrics() {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error("Failed to fetch metrics");
  return res.json();
}

// Alias used by overview page
export const getMetrics = fetchMetrics;

export async function fetchLogs() {
  const res = await fetch(`${API_BASE}/logs`);
  if (!res.ok) return [];
  return res.json();
}

export const getLogs = fetchLogs;

/**
 * Helper to export log data as CSV
 */
export function exportLogsAsCSV(logs: any[]) {
  if (logs.length === 0) return;

  const headers = Object.keys(logs[0]).join(",");
  const rows = logs.map(log =>
    Object.values(log).map(val =>
      typeof val === 'string' ? `"${val.replace(/"/g, '""')}"` : val
    ).join(",")
  ).join("\n");

  const csv = `${headers}\n${rows}`;
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `clearview_logs_${new Date().toISOString()}.csv`;
  link.click();
}

/**
 * Helper to export log data as JSON
 */
export function exportLogsAsJSON(logs: any[]) {
  const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `clearview_logs_${new Date().toISOString()}.json`;
  link.click();
}
