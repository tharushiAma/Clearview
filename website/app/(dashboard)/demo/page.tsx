"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Spinner } from "@/components/ui/spinner";
import { predict } from "@/lib/api";
import type { PredictionResult, AspectPrediction, SentimentLabel } from "@/lib/types";
import { Play, AlertTriangle, ArrowRight } from "lucide-react";

const SENTIMENT_COLORS: Record<SentimentLabel, string> = {
  NEG: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
  NEU: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400",
  POS: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400",
  NULL: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
};

export default function DemoPage() {
  const [text, setText] = useState(
    "This lipstick has amazing staying power and the color is beautiful, but the smell is too strong and the packaging feels cheap."
  );
  const [msrEnabled, setMsrEnabled] = useState(false);
  const [msrStrength, setMsrStrength] = useState([0.3]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [compareMode, setCompareMode] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await predict({
        text,
        msrEnabled,
        msrStrength: msrStrength[0],
      });
      setResult(response);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Live Demo</h1>
        <p className="text-muted-foreground">
          Test the ABSA model with custom reviews
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Input Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Input</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="review">Review Text</Label>
              <Textarea
                id="review"
                placeholder="Enter a cosmetics review (1-3 sentences)..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                className="resize-none"
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label htmlFor="msr-toggle">Enable MSR</Label>
                <p className="text-xs text-muted-foreground">
                  Multi-Sentiment Regularization
                </p>
              </div>
              <Switch
                id="msr-toggle"
                checked={msrEnabled}
                onCheckedChange={setMsrEnabled}
              />
            </div>

            {msrEnabled && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>MSR Strength (λ)</Label>
                  <span className="text-sm text-muted-foreground">
                    {msrStrength[0].toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={msrStrength}
                  onValueChange={setMsrStrength}
                  min={0}
                  max={1}
                  step={0.05}
                />
              </div>
            )}

            <Button
              onClick={handlePredict}
              disabled={!text.trim() || loading}
              className="w-full"
            >
              {loading ? (
                <>
                  <Spinner className="h-4 w-4 mr-2" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Prediction
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Conflict Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Conflict Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            {result ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                  <div>
                    <p className="text-sm font-medium">Conflict Probability</p>
                    <p className="text-3xl font-bold">
                      {(result.conflictProbability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div
                    className="h-16 w-16 rounded-full flex items-center justify-center"
                    style={{
                      background: `conic-gradient(var(--chart-1) ${result.conflictProbability * 360}deg, var(--muted) 0deg)`,
                    }}
                  >
                    <div className="h-12 w-12 rounded-full bg-card" />
                  </div>
                </div>

                {result.mixedSentimentDetected && (
                  <div className="flex items-center gap-2 p-3 rounded-lg bg-amber-50 dark:bg-amber-950/30 text-amber-800 dark:text-amber-400">
                    <AlertTriangle className="h-4 w-4" />
                    <span className="text-sm font-medium">
                      Mixed sentiment detected
                    </span>
                  </div>
                )}

                {msrEnabled && result.before && result.after && (
                  <div className="flex items-center gap-2">
                    <Switch
                      id="compare-toggle"
                      checked={compareMode}
                      onCheckedChange={setCompareMode}
                    />
                    <Label htmlFor="compare-toggle" className="text-sm">
                      Compare BEFORE vs AFTER MSR
                    </Label>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-32 flex items-center justify-center text-muted-foreground text-sm">
                Run a prediction to see conflict analysis
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Results */}
      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Predictions</CardTitle>
          </CardHeader>
          <CardContent>
            {compareMode && result.before && result.after ? (
              <Tabs defaultValue="comparison" className="w-full">
                <TabsList>
                  <TabsTrigger value="comparison">Side-by-Side</TabsTrigger>
                  <TabsTrigger value="before">Before MSR</TabsTrigger>
                  <TabsTrigger value="after">After MSR</TabsTrigger>
                </TabsList>
                <TabsContent value="comparison">
                  <ComparisonTable before={result.before} after={result.after} />
                </TabsContent>
                <TabsContent value="before">
                  <PredictionTable predictions={result.before} />
                </TabsContent>
                <TabsContent value="after">
                  <PredictionTable predictions={result.after} showMsrBadge />
                </TabsContent>
              </Tabs>
            ) : (
              <PredictionTable
                predictions={result.predictions}
                showMsrBadge={msrEnabled}
              />
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function PredictionTable({
  predictions,
  showMsrBadge = false,
}: {
  predictions: AspectPrediction[];
  showMsrBadge?: boolean;
}) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Aspect</TableHead>
            <TableHead>Predicted Label</TableHead>
            <TableHead>Confidence</TableHead>
            <TableHead>Top Tokens</TableHead>
            {showMsrBadge && <TableHead>MSR</TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody>
          {(predictions || []).map((pred) => (
            <TableRow key={pred.aspect}>
              <TableCell className="font-medium capitalize">
                {pred.aspect}
              </TableCell>
              <TableCell>
                <Badge
                  variant="secondary"
                  className={SENTIMENT_COLORS[pred.label]}
                >
                  {pred.label}
                </Badge>
              </TableCell>
              <TableCell>{((pred.confidence || 0) * 100).toFixed(1)}%</TableCell>
              <TableCell className="text-muted-foreground text-sm">
                {(pred.topTokens || []).join(", ")}
              </TableCell>
              {showMsrBadge && (
                <TableCell>
                  {pred.msrChanged && (
                    <Badge variant="outline" className="text-xs">
                      Changed
                    </Badge>
                  )}
                </TableCell>
              )}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function ComparisonTable({
  before,
  after,
}: {
  before: AspectPrediction[];
  after: AspectPrediction[];
}) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Aspect</TableHead>
            <TableHead>Before</TableHead>
            <TableHead className="w-8"></TableHead>
            <TableHead>After</TableHead>
            <TableHead>Conf. Change</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {(before || []).map((b, i) => {
            const a = (after || [])[i];
            if (!a) return null;
            const confDiff = (a.confidence || 0) - (b.confidence || 0);
            const labelChanged = b.label !== a.label;
            return (
              <TableRow key={b.aspect}>
                <TableCell className="font-medium capitalize">
                  {b.aspect}
                </TableCell>
                <TableCell>
                  <Badge
                    variant="secondary"
                    className={SENTIMENT_COLORS[b.label]}
                  >
                    {b.label}
                  </Badge>
                  <span className="ml-2 text-sm text-muted-foreground">
                    {((b.confidence || 0) * 100).toFixed(0)}%
                  </span>
                </TableCell>
                <TableCell>
                  <ArrowRight
                    className={`h-4 w-4 ${labelChanged ? "text-amber-500" : "text-muted-foreground"}`}
                  />
                </TableCell>
                <TableCell>
                  <Badge
                    variant="secondary"
                    className={SENTIMENT_COLORS[a.label]}
                  >
                    {a.label}
                  </Badge>
                  <span className="ml-2 text-sm text-muted-foreground">
                    {((a.confidence || 0) * 100).toFixed(0)}%
                  </span>
                </TableCell>
                <TableCell>
                  <span
                    className={`text-sm font-medium ${confDiff > 0
                        ? "text-green-600 dark:text-green-400"
                        : confDiff < 0
                          ? "text-red-600 dark:text-red-400"
                          : "text-muted-foreground"
                      }`}
                  >
                    {confDiff > 0 ? "+" : ""}
                    {(confDiff * 100).toFixed(1)}%
                  </span>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
