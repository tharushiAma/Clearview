"use client";

import { ChartTooltipContent } from "@/components/ui/chart"

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { getMetrics } from "@/lib/api";
import type { EvaluationMetrics, Aspect } from "@/lib/types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  ChartContainer,
} from "@/components/ui/chart";
import { Activity, Gauge, Clock, Zap, Target } from "lucide-react";

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; dataKey: string }>; label?: string }) {
  if (!active || !payload || !payload.length) {
    return null;
  }
  return (
    <div className="bg-background border border-border rounded-lg px-3 py-2 shadow-lg text-sm">
      <p className="font-medium">{label}</p>
      {payload.map((entry, index) => (
        <p key={index} className="text-muted-foreground">
          {entry.dataKey === "f1" ? "F1 Score" : entry.dataKey === "count" ? "Count" : entry.dataKey}: {entry.value}{entry.dataKey === "f1" ? "%" : ""}
        </p>
      ))}
    </div>
  );
}

const chartConfig = {
  f1: { label: "F1 Score", color: "var(--chart-1)" },
  count: { label: "Count", color: "var(--chart-2)" },
  aspect: { label: "Aspect", color: "var(--chart-3)" },
  bin: { label: "Bin", color: "var(--chart-4)" },
};

export default function OverviewPage() {
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <OverviewSkeleton />;
  }

  if (!metrics) {
    return <div className="text-muted-foreground">Failed to load metrics</div>;
  }

  const kpiCards = [
    {
      title: "Overall Macro-F1",
      value: (metrics.overallMacroF1 * 100).toFixed(1) + "%",
      icon: Target,
      description: "Model performance score",
    },
    {
      title: "Per-aspect Macro-F1",
      value: (metrics.perAspectMacroF1Avg * 100).toFixed(1) + "%",
      icon: Gauge,
      description: "Average across aspects",
    },
    {
      title: "Conflict AUC",
      value: (metrics.conflictAUC * 100).toFixed(1) + "%",
      icon: Activity,
      description: "Conflict detection accuracy",
    },
    {
      title: "Avg Latency",
      value: metrics.avgLatencyMs.toFixed(0) + " ms",
      icon: Clock,
      description: "Response time",
    },
    {
      title: "Throughput",
      value: metrics.throughputReqPerSec.toFixed(0) + " req/s",
      icon: Zap,
      description: "Requests per second",
    },
  ];

  const f1ChartData = metrics.aspectMetrics.map((m) => ({
    aspect: m.aspect,
    f1: +(m.f1 * 100).toFixed(1),
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
        <p className="text-muted-foreground">
          Model performance metrics and statistics
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 grid-cols-2 lg:grid-cols-5">
        {kpiCards.map((kpi) => (
          <Card key={kpi.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {kpi.title}
              </CardTitle>
              <kpi.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{kpi.value}</div>
              <p className="text-xs text-muted-foreground">{kpi.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid gap-4 lg:grid-cols-2">
        {/* Per-aspect F1 Bar Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Per-aspect F1 Scores</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig} className="h-[250px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={f1ChartData} layout="vertical">
                  <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <YAxis type="category" dataKey="aspect" width={80} tick={{ fontSize: 12 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="f1" radius={[0, 4, 4, 0]}>
                    {f1ChartData.map((_, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={`var(--chart-${(index % 5) + 1})`}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* Conflict Score Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Conflict Score Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig} className="h-[250px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={metrics.conflictScoreDistribution}>
                  <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" fill="var(--chart-2)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>

      {/* Confusion Matrix Preview Cards */}
      <div>
        <h2 className="text-lg font-medium mb-3">Confusion Matrices</h2>
        <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-7">
          {metrics.confusionMatrices.map((cm) => (
            <ConfusionMatrixCard key={cm.aspect} confusionMatrix={cm} />
          ))}
        </div>
      </div>
    </div>
  );
}

function ConfusionMatrixCard({
  confusionMatrix,
}: {
  confusionMatrix: { aspect: Aspect; matrix: number[][]; labels: string[] };
}) {
  const { aspect, matrix, labels } = confusionMatrix;
  const maxVal = Math.max(...matrix.flat());

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Card className="cursor-pointer hover:bg-accent/50 transition-colors">
          <CardHeader className="p-3 pb-2">
            <CardTitle className="text-sm font-medium capitalize">
              {aspect}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-3 pt-0">
            <div className="grid grid-cols-4 gap-0.5">
              {matrix.map((row, i) =>
                row.map((val, j) => {
                  const intensity = val / maxVal;
                  return (
                    <div
                      key={`${i}-${j}`}
                      className="aspect-square rounded-sm text-[8px] flex items-center justify-center"
                      style={{
                        backgroundColor: `oklch(0.6 0.15 250 / ${intensity * 0.8 + 0.1})`,
                        color: intensity > 0.5 ? "white" : "inherit",
                      }}
                    >
                      {val}
                    </div>
                  );
                })
              )}
            </div>
          </CardContent>
        </Card>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="capitalize">{aspect} Confusion Matrix</DialogTitle>
        </DialogHeader>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className="p-2 text-left text-muted-foreground">Actual \ Pred</th>
                {labels.map((l) => (
                  <th key={l} className="p-2 text-center font-medium">
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, i) => (
                <tr key={i}>
                  <td className="p-2 font-medium">{labels[i]}</td>
                  {row.map((val, j) => {
                    const intensity = val / maxVal;
                    const isDiagonal = i === j;
                    return (
                      <td
                        key={j}
                        className="p-2 text-center rounded"
                        style={{
                          backgroundColor: isDiagonal
                            ? `oklch(0.7 0.15 145 / ${intensity * 0.7 + 0.1})`
                            : `oklch(0.6 0.15 250 / ${intensity * 0.5 + 0.05})`,
                          color: intensity > 0.6 ? "white" : "inherit",
                        }}
                      >
                        {val}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function OverviewSkeleton() {
  return (
    <div className="space-y-6">
      <div>
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-4 w-64 mt-2" />
      </div>
      <div className="grid gap-4 grid-cols-2 lg:grid-cols-5">
        {Array.from({ length: 5 }).map((_, i) => (
          <Card key={i}>
            <CardHeader className="pb-2">
              <Skeleton className="h-4 w-24" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-8 w-16" />
              <Skeleton className="h-3 w-20 mt-2" />
            </CardContent>
          </Card>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <Skeleton className="h-5 w-40" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-[250px] w-full" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <Skeleton className="h-5 w-48" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-[250px] w-full" />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
