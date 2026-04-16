"use client";

import { useState, useEffect } from "react";
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
import { useToast } from "@/hooks/use-toast";
import { Play, AlertTriangle } from "lucide-react";

const SENTIMENT_COLORS: Record<SentimentLabel, string> = {
  NEG: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
  NEU: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400",
  POS: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400",
  NULL: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
};

export default function DemoPage() {
  const { toast } = useToast();
  const [text, setText] = useState(
    "The color is beautiful as same as the picture, but the smell is bit strong for a lipstick and this is too expensive compared to other stores"
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await predict({
        text,
        msrEnabled: true,
        msrStrength: 0.5,
      });
      setResult(response);
    } catch (err) {
      toast({
        variant: "destructive",
        title: "Prediction failed",
        description:
          err instanceof Error
            ? err.message
            : "An unexpected error occurred. Please try again.",
      });
    } finally {
      setLoading(false);
    }
  };

  if (!isMounted) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-10 w-48 bg-muted rounded" />
        <div className="h-4 w-80 bg-muted rounded" />
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="h-64 bg-muted rounded-xl" />
          <div className="h-64 bg-muted rounded-xl" />
        </div>
      </div>
    );
  }

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
            <PredictionTable
              predictions={result.predictions}
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function PredictionTable({
  predictions,
}: {
  predictions: AspectPrediction[];
}) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Aspect</TableHead>
            <TableHead>Predicted Label</TableHead>
            <TableHead>Confidence</TableHead>
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
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}


