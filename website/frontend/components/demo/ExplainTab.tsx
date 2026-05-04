// components/demo/ExplainTab.tsx
// This component physically draws the XAI mathematics to the screen. 
// Note that it DOES NOT manage its own memory or fetch data on its own. 
// It receives all data from its parent (ClearViewDemo.tsx) via "Props".
"use client";

import { Loader2, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import type { PredictResponse, ExplanationResponse } from "@/types";

// TypeScript Interface: Defines exactly what properties this component expects to be handed by the Parent.
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
  // Deconstructing all the props passed in so we can use them directly in the HTML.
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
      {/* ── Controls card ─────────────────────────────────────────────────────────── */}
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
              {/* This dropdown menu triggers the `onAspectChange` function from the Parent whenever a user changes it. */}
              <select
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background"
                value={explainAspect}
                onChange={(e) => onAspectChange(e.target.value)}
              >
                <option value="all">Analyze All Aspects</option>
                {/* Dynamically build dropdown options ONLY for aspects that were actually found during the Prediction step. */}
                {prediction?.predictions?.map((a) => (
                  <option key={a.aspect} value={a.aspect}>
                    {a.aspect}
                  </option>
                ))}
              </select>
            </div>
            {/* Run Button tied to the parent's logic */}
            <Button onClick={onExplain} disabled={isExplaining}>
              {isExplaining ? <Loader2 className="animate-spin mr-2" /> : "Run XAI"}
            </Button>
          </div>

          {/* ── Progress tracker UI ────────────────────────────────────────────────── */}
          {/* This block is entirely hidden until the explainSteps array has items inside it (i.e. someone clicked 'Run') */}
          {explainSteps.length > 0 && (
            <div className="mt-4 p-4 border rounded-lg bg-linear-to-r from-blue-50 to-indigo-50 border-blue-200">
              <h4 className="text-sm font-semibold text-blue-900 mb-3">XAI Analysis Progress</h4>
              <div className="space-y-2">
                {/* Loop out every step in the Tracker */}
                {explainSteps.map((step, idx) => (
                  <div key={idx} className="flex items-center gap-3">
                    {/* Render different SVGs based on whether the step string says 'done', 'progress', or 'pending' */}
                    {step.status === "done" && (
                      <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center shrink-0">
                         {/* Checkmark icon */}
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
                    
                    {/* Color the text based on the status! */}
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

      {/* ── Error Banner ────────────────────────────────────────────────────────── */}
      {/* If error string isn't null, physically render a red alert box */}
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

      {/* ── Final Results Visualization ────────────────────────────────────────── */}
      {/* Ensures we don't try to loop over data that doesn't exist yet */}
      {explanation && (
        <div className="space-y-8">
          
          {/* Section 1: Conflict drivers (If present in JSON) */}
          {explanation.ig_conflict && (
            <Card>
              <CardHeader><CardTitle>Conflict Drivers</CardTitle></CardHeader>
              <CardContent>
                <p className="text-sm text-gray-500 mb-4">Tokens increasing conflict probability:</p>
                <div className="flex flex-wrap gap-2">
                  {/* Destructure the nested arrays: [["word", 0.5], ["other", -0.2]] */}
                  {(explanation.ig_conflict.top_tokens || []).map(
                    ([token, score]: [string, number], idx: number) => (
                      <span
                        key={idx}
                        className="px-2 py-1 rounded text-sm font-mono"
                        style={{
                          // Dynamically change Red opacity based on how strong the mathematical score is!
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

          {/* Section 2: Standard Per-aspect attributions */}
          {/* Object.entries turns { "Price": {data}, "Smell": {data} } into an array we can loop over */}
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
                                ? `rgba(34, 197, 94, ${Math.min(score * 5, 0.6)})` // Positive math makes it Green
                                : `rgba(239, 68, 68, ${Math.min(Math.abs(score) * 5, 0.6)})`, // Negative makes it Red
                          }}
                        >
                          {token}
                        </span>
                      )
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </>
  );
}
