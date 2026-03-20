"use client";

import { Loader2, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import type { PredictResponse, ExplanationResponse } from "@/types";

interface ExplainStep {
  name: string;
  status: "pending" | "progress" | "done";
}

interface ExplainTabProps {
  text: string;
  prediction: PredictResponse | null;
  isExplaining: boolean;
  explanation: ExplanationResponse | null;
  explainAspect: string;
  onAspectChange: (v: string) => void;
  explainSteps: ExplainStep[];
  error: string | null;
  onExplain: () => void;
}

export function ExplainTab({
  text,
  prediction,
  isExplaining,
  explanation,
  explainAspect,
  onAspectChange,
  explainSteps,
  error,
  onExplain,
}: ExplainTabProps) {
  return (
    <>
      {/* Controls card */}
      <Card>
        <CardHeader>
          <CardTitle>XAI Analysis</CardTitle>
          <CardDescription>
            Visualize token attributions using Integrated Gradients &amp; SHAP.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-end gap-4">
            <div className="flex-1">
              <Label>Focus Aspect</Label>
              <select
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background"
                value={explainAspect}
                onChange={(e) => onAspectChange(e.target.value)}
              >
                <option value="all">Analyze All Aspects</option>
                {prediction?.predictions?.map((a) => (
                  <option key={a.aspect} value={a.aspect}>
                    {a.aspect}
                  </option>
                ))}
              </select>
            </div>
            <Button onClick={onExplain} disabled={isExplaining}>
              {isExplaining ? <Loader2 className="animate-spin mr-2" /> : "Run XAI"}
            </Button>
          </div>

          {/* Progress tracker */}
          {explainSteps.length > 0 && (
            <div className="mt-4 p-4 border rounded-lg bg-linear-to-r from-blue-50 to-indigo-50 border-blue-200">
              <h4 className="text-sm font-semibold text-blue-900 mb-3">XAI Analysis Progress</h4>
              <div className="space-y-2">
                {explainSteps.map((step, idx) => (
                  <div key={idx} className="flex items-center gap-3">
                    {step.status === "done" && (
                      <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center shrink-0">
                        <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                    )}
                    {step.status === "progress" && (
                      <Loader2 className="w-5 h-5 text-blue-600 animate-spin shrink-0" />
                    )}
                    {step.status === "pending" && (
                      <div className="w-5 h-5 rounded-full border-2 border-gray-300 shrink-0" />
                    )}
                    <span
                      className={`text-sm ${
                        step.status === "done"
                          ? "text-green-700 font-medium"
                          : step.status === "progress"
                          ? "text-blue-700 font-semibold"
                          : "text-gray-500"
                      }`}
                    >
                      {step.name}
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-blue-600 mt-3">
                {isExplaining ? "This may take 1-3 minutes..." : "Analysis complete!"}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Error */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="pt-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
              <div className="flex-1">
                <h4 className="font-semibold text-red-900 mb-1">XAI Analysis Failed</h4>
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {explanation && (
        <div className="space-y-8">
          {/* Conflict drivers */}
          {explanation.ig_conflict && (
            <Card>
              <CardHeader><CardTitle>Conflict Drivers</CardTitle></CardHeader>
              <CardContent>
                <p className="text-sm text-gray-500 mb-4">Tokens increasing conflict probability:</p>
                <div className="flex flex-wrap gap-2">
                  {(explanation.ig_conflict.top_tokens || []).map(
                    ([token, score]: [string, number], idx: number) => (
                      <span
                        key={idx}
                        className="px-2 py-1 rounded text-sm font-mono"
                        style={{
                          backgroundColor: `rgba(239, 68, 68, ${Math.min(Math.abs(score) * 5, 0.8)})`,
                          color: Math.abs(score) > 0.1 ? "white" : "black",
                        }}
                      >
                        {token}
                      </span>
                    )
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Per-aspect attributions */}
          {Object.entries(explanation.aspects || {}).map(([aspName, data]) => (
            <Card key={aspName}>
              <CardHeader>
                <CardTitle className="capitalize">{aspName} Attribution</CardTitle>
              </CardHeader>
              <CardContent className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-sm font-semibold mb-2">Integrated Gradients</h4>
                  <div className="flex flex-wrap gap-2">
                    {(data.ig_aspect.top_tokens || []).map(
                      ([token, score]: [string, number], i: number) => (
                        <span
                          key={i}
                          className="px-2 py-1 rounded text-sm font-mono border"
                          style={{
                            backgroundColor:
                              score > 0
                                ? `rgba(34, 197, 94, ${Math.min(score * 5, 0.6)})`
                                : `rgba(239, 68, 68, ${Math.min(Math.abs(score) * 5, 0.6)})`,
                          }}
                        >
                          {token}
                        </span>
                      )
                    )}
                  </div>
                </div>
                {data.msr_delta && (
                  <div>
                    <h4 className="text-sm font-semibold mb-2">MSR Impact (Delta)</h4>
                    <div className="text-xs space-y-1">
                      <div className="flex justify-between">
                        <span>Before Prob:</span>
                        <span className="font-mono">
                          {JSON.stringify(data.msr_delta.prob_before.map((n) => Number(n.toFixed(2))))}
                        </span>
                      </div>
                      <div className="flex justify-between font-bold">
                        <span>After Prob:</span>
                        <span className="font-mono">
                          {JSON.stringify(data.msr_delta.prob_after.map((n) => Number(n.toFixed(2))))}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </>
  );
}
