export interface AspectData {
  aspect: string;
  label: string;
  confidence: number;
  topTokens?: string[];
  probs?: number[];
  before?: {
    label: string;
    confidence: number;
  };
  after?: {
    label: string;
    confidence: number;
  };
  msrChanged: boolean;
}

export interface PredictResponse {
  predictions: AspectData[];
  conflictProbability: number;
  mixedSentimentDetected: boolean;
  timings?: {
    total_ms: number;
  };
  before?: AspectData[];
  after?: AspectData[];
}

export interface MetricData {
  overall_macro_f1: number;    // 3-class: negative / neutral / positive
  overall_macro_f1_sentiment: number;
  conflict: {
    conf_f1_macro: number;
    roc_auc: number;
    brier_score: number;
  };
  msr_error_reduction: {
    total_reduction: number;
  };
}

export interface LogEntry {
  epoch?: number;
  step?: number;
  loss?: number;
  [key: string]: any;
}

export interface ExplanationResponse {
  text: string;
  requested_aspect: string;
  ig_conflict?: {
    top_tokens: [string, number][];
  };
  aspects: {
    [key: string]: {
      ig_aspect: {
        top_tokens: [string, number][];
      };
    };
  };
}
