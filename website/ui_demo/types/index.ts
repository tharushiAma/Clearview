export interface AspectData {
  name: string;
  label: string;
  confidence: number;
  probs: number[];
  before?: {
    label: string;
    confidence: number;
  };
  after?: {
    label: string;
    confidence: number;
  };
  changed_by_msr: boolean;
}

export interface PredictResponse {
  aspects: AspectData[];
  conflict_prob: number;
  timings: {
    total_ms: number;
  };
}

export interface MetricData {
  overall_macro_f1_4class: number;
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
      msr_delta?: {
        prob_before: number[];
        prob_after: number[];
      };
    };
  };
}
