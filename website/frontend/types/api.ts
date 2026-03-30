// Shared types for API routes
export interface AspectPrediction {
  name: string;
  label: string;
  confidence: number;
  probs: number[];
  before: {
    label: string;
    confidence: number;
  };
  after: {
    label: string;
    confidence: number;
  };
  changed_by_msr: boolean;
}

export interface PredictResponse {
  aspects: AspectPrediction[];
  conflict_prob: number;
  timings: {
    total_ms: number;
  };
}

export interface TokenAttribution {
  token: string;
  attribution: number;
}

export interface XAIMethodResult {
  method: string;
  task: string;
  attributions: TokenAttribution[];
}

export interface AspectExplanation {
  ig_aspect: XAIMethodResult;
}

export interface ExplainResponse {
  text: string;
  requested_aspect: string;
  ig_conflict: XAIMethodResult;
  aspects: Record<string, AspectExplanation>;
}

export interface ConflictMetrics {
  conf_f1_macro: number;
  roc_auc: number;
  brier_score: number;
}

export interface MSRErrorReduction {
  total_reduction: number;
}

export interface MetricsResponse {
  overall_macro_f1_4class: number;
  overall_macro_f1_sentiment: number;
  conflict: ConflictMetrics;
  msr_error_reduction: MSRErrorReduction;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  [key: string]: string | number | boolean;
}

export type LogsResponse = LogEntry[];
