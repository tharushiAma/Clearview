import {
  ASPECTS,
  type Aspect,
  type AspectPrediction,
  type PredictionResult,
  type PredictRequest,
  type ExplanationBundle,
  type ExplainRequest,
  type EvaluationMetrics,
  type LogEntry,
  type SentimentLabel,
  type ExplanationMethod,
} from "./types";

function randomChoice(arr: readonly string[]): string {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomFloat(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

const SAMPLE_TOKENS = [
  "love", "hate", "great", "terrible", "amazing", "awful",
  "good", "bad", "nice", "poor", "excellent", "disappointing",
  "perfect", "horrible", "smooth", "sticky", "fresh", "stale",
  "expensive", "cheap", "fast", "slow", "beautiful", "ugly"
];

const LABELS: SentimentLabel[] = ["NEG", "NEU", "POS", "NULL"];

function generateMockPrediction(aspect: Aspect, msrEnabled: boolean): AspectPrediction {
  const tokens: string[] = [];
  for (let i = 0; i < 3; i++) {
    tokens.push(SAMPLE_TOKENS[randomInt(0, SAMPLE_TOKENS.length - 1)]);
  }
  return {
    aspect: aspect,
    label: LABELS[randomInt(0, 3)],
    confidence: randomFloat(0.6, 0.98),
    topTokens: tokens,
    msrChanged: msrEnabled && Math.random() > 0.7
  };
}

export async function predict(request: PredictRequest): Promise<PredictionResult> {
  await new Promise(function(resolve) { setTimeout(resolve, 500); });

  const predictions: AspectPrediction[] = [];
  for (const aspect of ASPECTS) {
    predictions.push(generateMockPrediction(aspect, request.msrEnabled));
  }

  let before: AspectPrediction[] | undefined = undefined;
  if (request.msrEnabled) {
    before = [];
    for (const aspect of ASPECTS) {
      before.push(generateMockPrediction(aspect, false));
    }
  }

  return {
    predictions: predictions,
    conflictProbability: randomFloat(0.05, 0.4),
    mixedSentimentDetected: Math.random() > 0.6,
    before: before,
    after: request.msrEnabled ? predictions : undefined
  };
}

export async function explain(request: ExplainRequest): Promise<ExplanationBundle> {
  await new Promise(function(resolve) { setTimeout(resolve, 800); });

  const aspectList: Aspect[] = request.aspect === "all" ? [...ASPECTS] : [request.aspect];
  
  const explanations: Array<{
    aspect: Aspect;
    method: ExplanationMethod;
    tokens: Array<{ token: string; attribution: number; msrDelta?: number }>;
  }> = [];

  for (const aspect of aspectList) {
    for (const method of request.methods) {
      const tokenCount = randomInt(5, 12);
      const tokenList: Array<{ token: string; attribution: number; msrDelta?: number }> = [];
      for (let i = 0; i < tokenCount; i++) {
        tokenList.push({
          token: SAMPLE_TOKENS[i % SAMPLE_TOKENS.length],
          attribution: randomFloat(-1, 1),
          msrDelta: request.msrEnabled ? randomFloat(-0.3, 0.3) : undefined
        });
      }
      explanations.push({
        aspect: aspect,
        method: method as ExplanationMethod,
        tokens: tokenList
      });
    }
  }

  const explanationsWithMeta = explanations.map(function(e) {
    return {
      aspect: e.aspect,
      method: e.method,
      tokens: e.tokens,
      metadata: { computeTimeMs: randomInt(50, 200) }
    };
  });

  return {
    text: request.text,
    explanations: explanations,
    rawJson: {
      model: "clearview-absa-v1",
      timestamp: new Date().toISOString(),
      config: {
        msrEnabled: request.msrEnabled,
        msrStrength: request.msrStrength
      },
      explanations: explanationsWithMeta
    }
  };
}

export async function getMetrics(): Promise<EvaluationMetrics> {
  await new Promise(function(resolve) { setTimeout(resolve, 300); });

  const aspectMetrics: Array<{ aspect: Aspect; precision: number; recall: number; f1: number }> = [];
  for (const aspect of ASPECTS) {
    aspectMetrics.push({
      aspect: aspect,
      precision: randomFloat(0.7, 0.95),
      recall: randomFloat(0.65, 0.92),
      f1: randomFloat(0.68, 0.9)
    });
  }

  const confusionMatrices: Array<{ aspect: Aspect; matrix: number[][]; labels: SentimentLabel[] }> = [];
  for (const aspect of ASPECTS) {
    const matrix: number[][] = [];
    for (let i = 0; i < 4; i++) {
      const row: number[] = [];
      for (let j = 0; j < 4; j++) {
        row.push(randomInt(5, 100));
      }
      matrix.push(row);
    }
    confusionMatrices.push({
      aspect: aspect,
      matrix: matrix,
      labels: LABELS
    });
  }

  const conflictScoreDistribution = [
    { bin: "0.0-0.1", count: randomInt(100, 300) },
    { bin: "0.1-0.2", count: randomInt(80, 200) },
    { bin: "0.2-0.3", count: randomInt(60, 150) },
    { bin: "0.3-0.4", count: randomInt(40, 100) },
    { bin: "0.4-0.5", count: randomInt(30, 80) },
    { bin: "0.5-0.6", count: randomInt(20, 60) },
    { bin: "0.6-0.7", count: randomInt(15, 40) },
    { bin: "0.7-0.8", count: randomInt(10, 30) },
    { bin: "0.8-0.9", count: randomInt(5, 20) },
    { bin: "0.9-1.0", count: randomInt(2, 10) }
  ];

  return {
    overallMacroF1: randomFloat(0.75, 0.88),
    perAspectMacroF1Avg: randomFloat(0.72, 0.85),
    conflictAUC: randomFloat(0.8, 0.92),
    avgLatencyMs: randomFloat(45, 120),
    throughputReqPerSec: randomFloat(50, 150),
    aspectMetrics: aspectMetrics,
    confusionMatrices: confusionMatrices,
    conflictScoreDistribution: conflictScoreDistribution,
    balancedAccuracy: randomFloat(0.7, 0.85),
    brierScore: randomFloat(0.1, 0.25),
    msrErrorReduction: randomFloat(0.05, 0.15),
    p95LatencyMs: randomFloat(150, 300),
    memoryUsageMB: randomFloat(512, 2048)
  };
}

export async function getLogs(): Promise<LogEntry[]> {
  await new Promise(function(resolve) { setTimeout(resolve, 200); });

  const sampleTexts = [
    "This lipstick has amazing staying power but the smell is too strong.",
    "Great texture and color, but shipping took forever.",
    "Love the price point, packaging could be better though.",
    "The color is beautiful but it feels sticky on my lips.",
    "Fast shipping, nice packaging, but the smell is off-putting.",
    "Perfect for the price! Stays on all day.",
    "Disappointed with the texture, feels too heavy.",
    "Amazing product, worth every penny!"
  ];

  const outcomes = ["Mostly Positive", "Mixed Sentiment", "Mostly Negative", "Neutral"];

  const logs: LogEntry[] = [];
  const now = Date.now();
  
  for (let i = 0; i < 20; i++) {
    const text = sampleTexts[randomInt(0, sampleTexts.length - 1)];
    logs.push({
      id: "log-" + now + "-" + i,
      timestamp: new Date(now - randomInt(0, 7 * 24 * 60 * 60 * 1000)).toISOString(),
      textPreview: text.slice(0, 50) + "...",
      msrEnabled: Math.random() > 0.5,
      conflictProbability: randomFloat(0.05, 0.6),
      overallOutcome: outcomes[randomInt(0, outcomes.length - 1)]
    });
  }

  logs.sort(function(a, b) {
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
  });

  return logs;
}

export function exportLogsAsCSV(logs: LogEntry[]): string {
  const headers = ["ID", "Timestamp", "Text Preview", "MSR Enabled", "Conflict Probability", "Overall Outcome"];
  const rows: string[] = [];
  
  for (const log of logs) {
    const row = [
      log.id,
      log.timestamp,
      '"' + log.textPreview + '"',
      String(log.msrEnabled),
      log.conflictProbability.toFixed(3),
      log.overallOutcome
    ];
    rows.push(row.join(","));
  }
  
  return headers.join(",") + "\n" + rows.join("\n");
}

export function exportLogsAsJSON(logs: LogEntry[]): string {
  return JSON.stringify(logs, null, 2);
}
