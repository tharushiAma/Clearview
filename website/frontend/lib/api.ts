// lib/api.ts
// This file acts as the "Middleman" between your Next.js Frontend and your Python FastAPI Backend.
// Whenever a React component needs data (like sentiment analysis), it calls a function in this file.

// We prefix all our requests with "/api", which Next.js forwards to "http://localhost:8000" (the backend).
const API_BASE = "/api";

// ---------------------------------------------------------------------------------------------------
// 1. predict()
// This function sends a single review text to the backend to get the sentiment (Positive/Negative/Neutral).
// ---------------------------------------------------------------------------------------------------
export async function predict(req: { text: string; msrEnabled: boolean; msrStrength: number }) {
  // We use the standard browser 'fetch' function to send an HTTP POST request to the backend.
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST", // POST means we are sending data, not just asking for it.
    headers: { "Content-Type": "application/json" }, // We tell the server: "Hey, we're giving you JSON."
    // JSON.stringify converts our JavaScript object into a pure string of JSON data.
    body: JSON.stringify({ text: req.text, msr_strength: req.msrStrength, msr_enabled: req.msrEnabled }),
  });

  // If the server crashes or sends us a 400/500 error code, res.ok will be false.
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Prediction failed" }));
    throw new Error(err.error || "Prediction failed");
  }

  // If successful, parse the JSON response from the server.
  const data = await res.json();

  // Python backend returns lowercase labels ("positive"), but our React components expect uppercase ("POS").
  // This helper function maps them dynamically. If the backend changes, the frontend won't break.
  const toLabel = (l: string) =>
    ({ positive: "POS", negative: "NEG", neutral: "NEU", not_mentioned: "NULL" } as Record<string, string>)[l] ?? "NULL";

  // Here we loop (.map) over every aspect (like price, smell, sizing) the AI found.
  // We clean up the raw data and format it into exactly what the frontend graphs expect.
  const predictions = (data.aspects || []).map((a: any) => ({
    aspect: a.name,
    label: toLabel(a.label),
    confidence: a.confidence,
    topTokens: a.top_tokens || [], // Used for fast attention attribution
    msrChanged: a.changed_by_msr || false,
    // If testing the Multi-Sentiment Resolver (MSR), store what the network thought *before* and *after*.
    before: a.before ? { label: toLabel(a.before.label), confidence: a.before.confidence } : undefined,
    after: a.after ? { label: toLabel(a.after.label), confidence: a.after.confidence } : undefined
  }));

  // Build the final result object that the React Component (like DemoPage) will receive.
  const result = {
    predictions,
    conflictProbability: data.conflict_prob || 0,
    mixedSentimentDetected: (data.conflict_prob || 0) > 0.5,
  } as any;

  // Add before/after if present in the first aspect (indicates MSR comparison)
  // This is specifically used to draw charts showing if the algorithm changed its mind.
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

  // Return the data directly to the React component that called predict()
  return result;
}

// Helper wrapper around predict() for simpler API signatures in some old UI files.
export async function fetchPrediction(
  text: string,
  msr_strength: number,
  msr_enabled: boolean,
) {
  return predict({ text, msrEnabled: msr_enabled, msrStrength: msr_strength });
}

// ---------------------------------------------------------------------------------------------------
// 2. explain()
// This function requests heavy XAI mathematical computations (SHAP, LIME) to see WHY the AI chose a label.
// ---------------------------------------------------------------------------------------------------
export async function explain(req: { text: string; aspect: string; methods: string[]; msrEnabled: boolean; msrStrength: number; signal?: AbortSignal }) {
  const res = await fetch(`${API_BASE}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: req.text,
      aspect: req.aspect,
      methods: req.methods, // e.g. ["shap", "lime", "ig"]
      msr_strength: req.msrStrength,
      msr_enabled: req.msrEnabled,
    }),
    signal: req.signal, // signal allows us to cancel the network request if the user navigates away!
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

  // Create an empty array to hold the clean formatted data we will pass to React.
  const explanations: any[] = [];
  const aspects = data.aspects || {};

  // For every aspect (price, smell)...
  Object.keys(aspects).forEach((aspName) => {
    const aspData = aspects[aspName];
    
    // If the Python server successfully ran Integrated Gradients (IG)...
    if (aspData.ig_aspect) {
      explanations.push({
        aspect: aspName,
        method: "ig",
        // Map the tokens! Example token: ["expensive", -0.8] -> token = expensive, attribution = -0.8
        tokens: (aspData.ig_aspect.top_tokens || []).map((t: any) => ({
          token: t[0],
          attribution: t[1],
        }))
      });
    }

    // Handle Local Interpretable Model-agnostic Explanations (LIME)
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

    // Handle Shapley Additive Explanations (SHAP)
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

    // Handle instant Attention networks
    if (aspData.attention_aspect) {
      explanations.push({
        aspect: aspName,
        method: "attention",
        tokens: (aspData.attention_aspect.top_tokens || []).map((t: any) => ({
          token: t[0],
          attribution: t[1]
        }))
      });
    }
  });

  return {
    text: data.text || req.text,
    explanations,
    rawJson: data // We pass the raw python data too, so you can debug what happened!
  };
}

// A simple wrapper to quickly fetch purely Integrated Gradients without passing a large object.
export async function fetchExplanation(
  text: string,
  aspect: string,
  msr_strength: number,
  signal?: AbortSignal,
) {
  return explain({ text, aspect, methods: ["ig"], msrEnabled: true, msrStrength: msr_strength, signal });
}

// ---------------------------------------------------------------------------------------------------
// 3. predictBulk()
// This sends thousands of reviews via a CSV file upload to get a massive batch result.
// ---------------------------------------------------------------------------------------------------
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

  // A tiny helper to add up the total counts so the dashboard charts are easy to draw
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
  
  // Clean up the rows before sending them to the table UI
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
