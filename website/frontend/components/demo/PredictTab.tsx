// components/demo/PredictTab.tsx
// This component handles the visual presentation of the "Predict" page in the Demo.
// Like the ExplainTab, it receives ALL its data via "Props" from the ClearViewDemo parent.

"use client";

import { BrainCircuit, Loader2, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { PredictResponse } from "@/types";

// TypeScript Interface: A contract guaranteeing what data the Parent MUST pass down!
interface PredictTabProps {
  text: string;
  onTextChange: (v: string) => void;
  isPredicting: boolean;
  prediction: PredictResponse | null;
  error: string | null;
  onPredict: () => void;
}

export function PredictTab({
  // Deconstructing properties from the contract above so we can use them directly in JSX
  text,
  onTextChange,
  isPredicting,
  prediction,
  error,
  onPredict,
}: PredictTabProps) {
  return (
    <>
      {/* ── Input Card ────────────────────────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BrainCircuit className="w-5 h-5 text-blue-500" />
            Input & Controls
          </CardTitle>
          <CardDescription>
            Enter review text and run Multi-Aspect Sentiment Resolution (MSR).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Review Text</Label>
            {/* The Text Box! */}
            <Textarea
              value={text} // Displays the parent's memory
              onChange={(e) => onTextChange(e.target.value)} // Triggers parent's function when typed in!
              rows={4}
              className="font-mono text-sm"
            />
          </div>
          <div className="flex justify-end pt-2">
            {/* Execute Button */}
            <Button onClick={onPredict} disabled={isPredicting} size="lg">
              {/* Shows a spinning wheel if isPredicting is true. Otherwise says "Run Prediction" */}
              {isPredicting ? (
                <Loader2 className="w-4 h-4 animate-spin mr-2" />
              ) : (
                "Run Prediction"
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* ── Error Banner ──────────────────────────────────────────────────────────── */}
       {/* If a Python error occurred, draw this red alert box */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="pt-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
              <div className="flex-1">
                <h4 className="font-semibold text-red-900 mb-1">Prediction Failed</h4>
                <p className="text-sm text-red-700">{error}</p>
                <p className="text-xs text-red-600 mt-2">
                  Hint: Start the backend with:{" "}
                  <code className="bg-red-100 px-1 py-0.5 rounded">
                    python backend_server.py
                  </code>
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Visual Results Dashboard ──────────────────────────────────────────────── */}
      {/* Logic operator `&&`: Only render the HTML below if `prediction` is NOT null. */}
      {prediction && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Section 1: Conflict Score Block */}
          {/* Draws a big percentage for conflict detection! */}
          <Card className="lg:col-span-1 border-l-4 border-l-purple-500">
            <CardHeader>
              <CardTitle>Conflict Detection</CardTitle>
            </CardHeader>
            <CardContent className="text-center space-y-4">
              <div className="text-5xl font-bold text-slate-900">
                {/* Convert 0.732 to 73.2% */}
                {((prediction.conflictProbability || 0) * 100).toFixed(1)}%
              </div>
              <p className="text-sm text-muted-foreground">Probability of Aspect Conflict</p>
              
              {/* Radix UI visual progress bar */}
              <Progress value={(prediction.conflictProbability || 0) * 100} className="h-2" />
              
              {/* Conditionally render a red warning or a green check badge based on math */}
              {(prediction.conflictProbability || 0) > 0.5 ? (
                <Badge variant="destructive" className="mt-2">High Conflict</Badge>
              ) : (
                <Badge variant="secondary" className="mt-2 bg-green-100 text-green-800">Coherent</Badge>
              )}
            </CardContent>
          </Card>

          {/* Section 2: Core Predictions Grid */}
          <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Loop through every prediction aspect found by the AI (price, color, smell, etc) */}
            {(prediction.predictions || []).map((asp) => (
              
              // We assign dynamic tailwind border colors depending on pos/neg/null sentiment!
              <Card
                key={asp.aspect}
                className={`relative border-l-4 ${
                  asp.label === "not_mentioned"
                    ? "border-l-slate-200 bg-slate-50/50 opacity-60"
                    : asp.label === "positive"
                    ? "border-l-green-500"
                    : asp.label === "negative"
                    ? "border-l-red-500"
                    : "border-l-slate-300"
                }`}
              >
                <CardHeader className="pb-2">
                  <div className="flex justify-between items-center">
                    <CardTitle
                      className={`capitalize text-lg ${
                        asp.label === "not_mentioned" ? "text-slate-400" : ""
                      }`}
                    >
                      {asp.aspect}
                    </CardTitle>
                    {/* The small sentiment label pill sitting at the top right of the card. */}
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                        asp.label === "positive"
                          ? "bg-green-100 text-green-800"
                          : asp.label === "negative"
                          ? "bg-red-100 text-red-800"
                          : asp.label === "not_mentioned"
                          ? "bg-slate-100 text-slate-400 italic"
                          : "bg-slate-100 text-slate-700"
                      }`}
                    >
                      {/* String formatting for the pill text */}
                      {asp.label === "positive"
                        ? "✓ positive"
                        : asp.label === "negative"
                        ? "✗ negative"
                        : asp.label === "not_mentioned"
                        ? "— not mentioned"
                        : "— neutral"}
                    </span>
                  </div>
                </CardHeader>
                
                <CardContent className="text-sm space-y-2">
                  {/* If the AI didn't find the aspect, just print gray text. */}
                  {asp.label === "not_mentioned" ? (
                    <p className="text-xs text-slate-400 italic text-center py-1">
                      Not referenced in this review
                    </p>
                  ) : (
                    // Otherwise, print the confidence bars and words!
                    <>
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Confidence</span>
                        <span className="font-medium">
                           {/* Math scaling: 0.98 -> 98.0% */}
                          {((asp.confidence || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      {/* Custom built Progress bar div that uses inline styles to map width to percentage */}
                      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            asp.label === "positive"
                              ? "bg-green-500"
                              : asp.label === "negative"
                              ? "bg-red-500"
                              : "bg-slate-400"
                          }`}
                          style={{ width: `${((asp.confidence || 0) * 100).toFixed(1)}%` }}
                        />
                      </div>
                      
                      {/* If the fast-attention words trickled through, draw gray keyword token tags! */}
                      {asp.topTokens && asp.topTokens.length > 0 && (
                        <div className="pt-1">
                          <p className="text-xs text-muted-foreground mb-1">Key words</p>
                          <div className="flex flex-wrap gap-1">
                            {asp.topTokens.map((token: string, i: number) => (
                              <span
                                key={i}
                                className="px-1.5 py-0.5 bg-slate-100 text-slate-700 rounded text-xs font-mono"
                              >
                                {token}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
