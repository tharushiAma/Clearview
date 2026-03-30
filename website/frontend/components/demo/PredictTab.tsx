"use client";

import { BrainCircuit, Loader2, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { PredictResponse } from "@/types";

interface PredictTabProps {
  text: string;
  onTextChange: (v: string) => void;
  isPredicting: boolean;
  prediction: PredictResponse | null;
  error: string | null;
  onPredict: () => void;
}

export function PredictTab({
  text,
  onTextChange,
  isPredicting,
  prediction,
  error,
  onPredict,
}: PredictTabProps) {
  return (
    <>
      {/* Input card */}
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
            <Textarea
              value={text}
              onChange={(e) => onTextChange(e.target.value)}
              rows={4}
              className="font-mono text-sm"
            />
          </div>
          <div className="flex justify-end pt-2">
            <Button onClick={onPredict} disabled={isPredicting} size="lg">
              {isPredicting ? (
                <Loader2 className="w-4 h-4 animate-spin mr-2" />
              ) : (
                "Run Prediction"
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Error */}
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

      {/* Results */}
      {prediction && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Conflict score */}
          <Card className="lg:col-span-1 border-l-4 border-l-purple-500">
            <CardHeader>
              <CardTitle>Conflict Detection</CardTitle>
            </CardHeader>
            <CardContent className="text-center space-y-4">
              <div className="text-5xl font-bold text-slate-900">
                {((prediction.conflictProbability || 0) * 100).toFixed(1)}%
              </div>
              <p className="text-sm text-muted-foreground">Probability of Aspect Conflict</p>
              <Progress value={(prediction.conflictProbability || 0) * 100} className="h-2" />
              {(prediction.conflictProbability || 0) > 0.5 ? (
                <Badge variant="destructive" className="mt-2">High Conflict</Badge>
              ) : (
                <Badge variant="secondary" className="mt-2 bg-green-100 text-green-800">Coherent</Badge>
              )}
            </CardContent>
          </Card>

          {/* Aspects grid */}
          <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-4">
            {(prediction.predictions || []).map((asp) => (
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
                  {asp.label === "not_mentioned" ? (
                    <p className="text-xs text-slate-400 italic text-center py-1">
                      Not referenced in this review
                    </p>
                  ) : (
                    <>
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Confidence</span>
                        <span className="font-medium">
                          {((asp.confidence || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
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
