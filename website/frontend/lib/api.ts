const API_BASE = "/api";

export async function predict(req: { text: string; msrEnabled: boolean; msrStrength: number }) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: req.text, msr_strength: req.msrStrength, msr_enabled: req.msrEnabled }),
  });
  if (!res.ok) throw new Error("Prediction failed");
  const data = await res.json();

  // Map backend lowercase labels to frontend uppercase SentimentLabel
  const toLabel = (l: string) =>
    ({ positive: "POS", negative: "NEG", neutral: "NEU", not_mentioned: "NULL" } as Record<string, string>)[l] ?? "NULL";

  // Transform backend response to frontend types
  const predictions = (data.aspects || []).map((a: any) => ({
    aspect: a.name,
    label: toLabel(a.label),
    confidence: a.confidence,
    topTokens: a.top_tokens || [],
    msrChanged: a.changed_by_msr || false,
    before: a.before ? { label: toLabel(a.before.label), confidence: a.before.confidence } : undefined,
    after: a.after ? { label: toLabel(a.after.label), confidence: a.after.confidence } : undefined
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
      label: toLabel(a.before.label),
      confidence: a.before.confidence,
      topTokens: [],
      msrChanged: false
    }));
    result.after = data.aspects.map((a: any) => ({
      aspect: a.name,
      label: toLabel(a.after.label),
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
  if (!res.ok) {
    let detail = `Explanation failed (HTTP ${res.status})`;
    try {
      const errJson = await res.json();
      detail = errJson.error || errJson.detail || detail;
    } catch {}
    throw new Error(detail);
  }
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
        }))
      });
    }

    // Handle LIME
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

    // Handle SHAP
    if (aspData.shap_aspect) {
      explanations.push({
        aspect: aspName,
        method: "shap",
        tokens: (aspData.shap_aspect.top_tokens || []).map((t: any) => ({
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
  const data = await res.json();

  // Normalize lowercase labels from backend to uppercase SentimentLabel
  const labelUp = (l: string) =>
    ({ positive: "POS", negative: "NEG", neutral: "NEU", not_mentioned: "NULL" } as Record<string, string>)[l] ?? l.toUpperCase().slice(0, 3);

  const normalizeCounts = (counts: Record<string, number>) => {
    const normalized: Record<string, number> = { POS: 0, NEG: 0, NEU: 0, NULL: 0 };
    Object.entries(counts).forEach(([k, v]) => {
      const key = labelUp(k);
      normalized[key] = (normalized[key] || 0) + v;
    });
    return normalized;
  };

  if (data.aspect_summary) {
    Object.keys(data.aspect_summary).forEach((asp) => {
      data.aspect_summary[asp] = normalizeCounts(data.aspect_summary[asp]);
    });
  }
  if (data.overall_counts) {
    data.overall_counts = normalizeCounts(data.overall_counts);
  }
  if (data.rows) {
    data.rows = data.rows.map((row: any) => ({
      ...row,
      aspects: (row.aspects || []).map((a: any) => ({
        ...a,
        label: labelUp(a.label ?? "NULL"),
      })),
    }));
  }

  return data;
}
