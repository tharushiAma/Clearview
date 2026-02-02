"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { getMetrics } from "@/lib/api";
import { ASPECTS, type Aspect, type EvaluationMetrics, type SentimentLabel } from "@/lib/types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  ChartContainer,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Clock, Cpu, HardDrive } from "lucide-react";

const chartConfig = {
  precision: { label: "Precision", color: "var(--chart-1)" },
  recall: { label: "Recall", color: "var(--chart-2)" },
  f1: { label: "F1", color: "var(--chart-3)" },
};

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [checkpoint, setCheckpoint] = useState("clearview-absa-v1");
  const [split, setSplit] = useState("test");
  const [selectedAspect, setSelectedAspect] = useState<Aspect>("stayingpower");

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .finally(() => setLoading(false));
  }, [checkpoint, split]);

  if (loading) {
    return <AnalyticsSkeleton />;
  }

  if (!metrics) {
    return <div className="text-muted-foreground">Failed to load metrics</div>;
  }

  const aspectMetricsData = metrics.aspectMetrics.map((m) => ({
    aspect: m.aspect,
    precision: +(m.precision * 100).toFixed(1),
    recall: +(m.recall * 100).toFixed(1),
    f1: +(m.f1 * 100).toFixed(1),
  }));

  const selectedConfusionMatrix = metrics.confusionMatrices.find(
    (cm) => cm.aspect === selectedAspect
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Analytics</h1>
        <p className="text-muted-foreground">
          Detailed evaluation metrics and performance analysis
        </p>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="checkpoint">Model Checkpoint</Label>
              <Input
                id="checkpoint"
                value={checkpoint}
                onChange={(e) => setCheckpoint(e.target.value)}
                placeholder="Checkpoint name"
              />
            </div>
            <div className="space-y-2">
              <Label>Dataset Split</Label>
              <Select value={split} onValueChange={setSplit}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="val">Validation</SelectItem>
                  <SelectItem value="test">Test</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Date Range</Label>
              <Input type="date" defaultValue="2025-01-01" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Metrics */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Balanced Accuracy"
          value={(metrics.balancedAccuracy * 100).toFixed(1) + "%"}
          subtitle="Across all aspects"
        />
        <MetricCard
          title="Conflict AUC"
          value={(metrics.conflictAUC * 100).toFixed(1) + "%"}
          subtitle="Area under curve"
        />
        <MetricCard
          title="Brier Score"
          value={metrics.brierScore.toFixed(3)}
          subtitle="Lower is better"
        />
        <MetricCard
          title="MSR Error Reduction"
          value={(metrics.msrErrorReduction * 100).toFixed(1) + "%"}
          subtitle="Improvement with MSR"
        />
      </div>

      {/* Per-aspect Metrics Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Per-aspect Precision / Recall / F1</CardTitle>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={aspectMetricsData}>
                <XAxis dataKey="aspect" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                <Tooltip content={<ChartTooltipContent />} />
                <Legend />
                <Bar dataKey="precision" fill="var(--chart-1)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="recall" fill="var(--chart-2)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" fill="var(--chart-3)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* Confusion Matrix */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between flex-wrap gap-4">
            <CardTitle className="text-base">Confusion Matrix</CardTitle>
            <Select
              value={selectedAspect}
              onValueChange={(v) => setSelectedAspect(v as Aspect)}
            >
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {ASPECTS.map((aspect) => (
                  <SelectItem key={aspect} value={aspect} className="capitalize">
                    {aspect}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          {selectedConfusionMatrix && (
            <ConfusionMatrixTable confusionMatrix={selectedConfusionMatrix} />
          )}
        </CardContent>
      </Card>

      {/* Runtime Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Runtime Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/50">
              <Clock className="h-8 w-8 text-muted-foreground" />
              <div>
                <p className="text-sm text-muted-foreground">Avg Latency</p>
                <p className="text-xl font-semibold">
                  {metrics.avgLatencyMs.toFixed(0)} ms
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/50">
              <Cpu className="h-8 w-8 text-muted-foreground" />
              <div>
                <p className="text-sm text-muted-foreground">P95 Latency</p>
                <p className="text-xl font-semibold">
                  {metrics.p95LatencyMs.toFixed(0)} ms
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/50">
              <HardDrive className="h-8 w-8 text-muted-foreground" />
              <div>
                <p className="text-sm text-muted-foreground">Memory Usage</p>
                <p className="text-xl font-semibold">
                  {metrics.memoryUsageMB.toFixed(0)} MB
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function MetricCard({
  title,
  value,
  subtitle,
}: {
  title: string;
  value: string;
  subtitle: string;
}) {
  return (
    <Card>
      <CardContent className="pt-6">
        <p className="text-sm text-muted-foreground">{title}</p>
        <p className="text-2xl font-bold mt-1">{value}</p>
        <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
      </CardContent>
    </Card>
  );
}

function ConfusionMatrixTable({
  confusionMatrix,
}: {
  confusionMatrix: { aspect: Aspect; matrix: number[][]; labels: SentimentLabel[] };
}) {
  const { matrix, labels } = confusionMatrix;
  const maxVal = Math.max(...matrix.flat());

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="p-3 text-left text-muted-foreground border-b">
              Actual \ Predicted
            </th>
            {labels.map((l) => (
              <th key={l} className="p-3 text-center font-medium border-b">
                {l}
              </th>
            ))}
            <th className="p-3 text-center font-medium border-b text-muted-foreground">
              Total
            </th>
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const rowTotal = row.reduce((a, b) => a + b, 0);
            return (
              <tr key={i}>
                <td className="p-3 font-medium border-b">{labels[i]}</td>
                {row.map((val, j) => {
                  const intensity = val / maxVal;
                  const isDiagonal = i === j;
                  return (
                    <td
                      key={j}
                      className="p-3 text-center border-b"
                      style={{
                        backgroundColor: isDiagonal
                          ? `oklch(0.7 0.15 145 / ${intensity * 0.6 + 0.1})`
                          : `oklch(0.6 0.15 250 / ${intensity * 0.4 + 0.05})`,
                        color: intensity > 0.6 ? "white" : "inherit",
                      }}
                    >
                      {val}
                    </td>
                  );
                })}
                <td className="p-3 text-center border-b text-muted-foreground">
                  {rowTotal}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function AnalyticsSkeleton() {
  return (
    <div className="space-y-6">
      <div>
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-4 w-64 mt-2" />
      </div>
      <Card>
        <CardContent className="pt-6">
          <div className="grid gap-4 sm:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-10 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}>
            <CardContent className="pt-6">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-8 w-16 mt-2" />
            </CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-48" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[300px] w-full" />
        </CardContent>
      </Card>
    </div>
  );
}
