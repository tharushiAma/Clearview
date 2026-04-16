// Aspect types
export const ASPECTS = [
  "stayingpower",
  "texture",
  "smell",
  "price",
  "colour",
  "shipping",
  "packing",
] as const;

export type Aspect = (typeof ASPECTS)[number];

// Sentiment labels
export type SentimentLabel = "NEG" | "NEU" | "POS" | "NULL";

// Prediction types
export interface AspectPrediction {
  aspect: Aspect;
  label: SentimentLabel;
  confidence: number;
  topTokens: string[];
  msrChanged: boolean;
}

export interface PredictionResult {
  predictions: AspectPrediction[];
  conflictProbability: number;
  mixedSentimentDetected: boolean;
  before?: AspectPrediction[];
  after?: AspectPrediction[];
}

export interface PredictRequest {
  text: string;
  msrEnabled: boolean;
  msrStrength: number;
}

// Explanation types
export type ExplanationMethod = "ig" | "lime" | "shap" | "attention";

export interface TokenAttribution {
  token: string;
  attribution: number;
}

export interface AspectExplanation {
  aspect: Aspect;
  method: ExplanationMethod;
  tokens: TokenAttribution[];
}

export interface ExplanationBundle {
  text: string;
  explanations: AspectExplanation[];
  rawJson: Record<string, unknown>;
}

export interface ExplainRequest {
  text: string;
  aspect: Aspect | "all";
  methods: ExplanationMethod[];
  msrEnabled: boolean;
  msrStrength: number;
}

// Metrics types
export interface AspectMetrics {
  aspect: Aspect;
  precision: number;
  recall: number;
  f1: number;
}

export interface ConfusionMatrix {
  aspect: Aspect;
  matrix: number[][]; // 4x4 for NEG, NEU, POS, NULL
  labels: SentimentLabel[];
}

export interface EvaluationMetrics {
  overallMacroF1: number;
  perAspectMacroF1Avg: number;
  conflictAUC: number;
  avgLatencyMs: number;
  throughputReqPerSec: number;
  aspectMetrics: AspectMetrics[];
  confusionMatrices: ConfusionMatrix[];
  conflictScoreDistribution: { bin: string; count: number }[];
  balancedAccuracy: number;
  brierScore: number;
  msrErrorReduction: number;
  p95LatencyMs: number;
  memoryUsageMB: number;
}

// Log types
export interface LogEntry {
  id: string;
  timestamp: string;
  textPreview: string;
  msrEnabled: boolean;
  conflictProbability: number;
  overallOutcome: string;
}

// Settings types
export interface AppSettings {
  apiBaseUrl: string;
  defaultCheckpointPath: string;
  darkMode: boolean;
}

// Bulk Review types
export interface BulkAspectData {
  name: string;
  label: SentimentLabel;
  confidence: number;
}

export interface BulkReviewRow {
  review_index: number;
  text: string;
  aspects: BulkAspectData[];
  conflict_prob: number;
  error?: string;
}

export interface AspectSentimentCounts {
  POS: number;
  NEG: number;
  NEU: number;
  NULL: number;
  [key: string]: number;
}

export interface BulkPredictResult {
  total_reviews: number;
  total_processed: number;
  mixed_count: number;
  overall_counts: AspectSentimentCounts;
  aspect_summary: Record<string, AspectSentimentCounts>;
  avg_confidence: Record<string, number>;
  rows: BulkReviewRow[];
  timings: { total_ms: number };
}
