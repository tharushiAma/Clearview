"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table";
import { Spinner } from "@/components/ui/spinner";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from "recharts";
import { ChartContainer } from "@/components/ui/chart";
import {
    UploadCloud,
    FileText,
    PlayCircle,
    Download,
    TrendingUp,
    TrendingDown,
    Minus,
    AlertTriangle,
    CheckCircle2,
    X,
    Sparkles,
} from "lucide-react";
import { Label } from "@/components/ui/label";
import { predictBulk } from "@/lib/api";
import type { BulkPredictResult, SentimentLabel } from "@/lib/types";
import { parseCSV, exportResultsCSV } from "@/lib/csv-utils";
import { KpiCard, pct } from "@/components/bulk/KpiCard";
import { XAIModal } from "@/components/bulk/XAIModal";

// ── Colour helpers ───────────────────────────────────────────────────────────

const SENTIMENT_COLORS: Record<SentimentLabel, string> = {
    POS: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400",
    NEG: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
    NEU: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400",
    NULL: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
};

const BAR_COLORS: Record<string, string> = {
    POS: "#10b981",
    NEG: "#ef4444",
    NEU: "#f59e0b",
    NULL: "#94a3b8",
};

const chartConfig = {
    POS: { label: "Positive", color: BAR_COLORS.POS },
    NEG: { label: "Negative", color: BAR_COLORS.NEG },
    NEU: { label: "Neutral", color: BAR_COLORS.NEU },
    NULL: { label: "Not Mentioned", color: BAR_COLORS.NULL },
};

// ── Page ─────────────────────────────────────────────────────────────────────

export default function BulkReviewsPage() {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [dragging, setDragging] = useState(false);
    const [csvFile, setCsvFile] = useState<File | null>(null);
    const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
    const [csvRows, setCsvRows] = useState<string[][]>([]);
    const [selectedColumn, setSelectedColumn] = useState<string>("");
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState<BulkPredictResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isMounted, setIsMounted] = useState(false);
    const [explainModalText, setExplainModalText] = useState<string | null>(null);

    useEffect(() => { setIsMounted(true); }, []);

    // ── File handling ────────────────────────────────────────────────────────
    const handleFile = useCallback((file: File) => {
        if (!file.name.endsWith(".csv")) {
            setError("Please upload a CSV file (.csv)");
            return;
        }
        setError(null);
        setResult(null);
        setCsvFile(file);
        const reader = new FileReader();
        reader.onload = (e) => {
            const { headers, rows } = parseCSV(e.target?.result as string);
            setCsvHeaders(headers);
            setCsvRows(rows);
            setSelectedColumn(headers[0] ?? "");
        };
        reader.readAsText(file);
    }, []);

    const onDrop = useCallback(
        (e: React.DragEvent<HTMLDivElement>) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
        },
        [handleFile]
    );

    // ── Process ──────────────────────────────────────────────────────────────
    const handleProcess = async () => {
        const colIndex = csvHeaders.indexOf(selectedColumn);
        if (colIndex === -1 || !csvRows.length) return;
        const reviews = csvRows.map((r) => r[colIndex] ?? "").filter((t) => t.trim());
        if (!reviews.length) { setError("No review text found in the selected column."); return; }

        setLoading(true);
        setProgress(0);
        setError(null);
        const interval = setInterval(() => setProgress((p) => Math.min(p + 2, 90)), 500);
        try {
            setResult(await predictBulk(reviews, true));
            setProgress(100);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Processing failed");
        } finally {
            clearInterval(interval);
            setLoading(false);
        }
    };

    const aspectNames = result ? Object.keys(result.aspect_summary) : [];
    const stackedChartData = aspectNames.map((asp) => ({
        aspect: asp.charAt(0).toUpperCase() + asp.slice(1),
        ...result!.aspect_summary[asp],
    }));

    if (!isMounted) return <div className="space-y-6 animate-pulse p-6"><div className="h-10 w-64 bg-muted rounded" /><div className="h-4 w-full max-w-lg bg-muted rounded mb-10" /></div>;

    return (
        <>
            <div className="space-y-6">
                <div>
                    <h1 className="text-2xl font-semibold tracking-tight">Bulk Reviews Dashboard</h1>
                    <p className="text-muted-foreground">Upload a CSV file of reviews for batch sentiment analysis — designed for brand managers</p>
                </div>

                {/* ── Upload + Settings ──────────────────────────────────── */}
                <div className="grid gap-6 lg:grid-cols-3">
                    <Card className="lg:col-span-2">
                        <CardHeader><CardTitle className="text-base">Upload CSV</CardTitle></CardHeader>
                        <CardContent className="space-y-4">
                            {/* Drop zone */}
                            <div
                                className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${dragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-muted/30"}`}
                                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                                onDragLeave={() => setDragging(false)}
                                onDrop={onDrop}
                                onClick={() => fileInputRef.current?.click()}
                            >
                                <input ref={fileInputRef} type="file" accept=".csv" className="hidden"
                                    onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
                                <UploadCloud className="h-10 w-10 mx-auto mb-3 text-muted-foreground" />
                                {csvFile ? (
                                    <div className="flex items-center justify-center gap-2">
                                        <FileText className="h-5 w-5 text-primary" />
                                        <span className="font-medium text-sm">{csvFile.name}</span>
                                        <span className="text-xs text-muted-foreground">({csvRows.length} rows, {csvHeaders.length} columns)</span>
                                        <button onClick={(e) => { e.stopPropagation(); setCsvFile(null); setCsvHeaders([]); setCsvRows([]); setResult(null); }} className="ml-1 text-muted-foreground hover:text-destructive transition-colors"><X className="h-4 w-4" /></button>
                                    </div>
                                ) : (
                                    <>
                                        <p className="text-sm font-medium">Drag & drop a CSV file here, or click to browse</p>
                                        <p className="text-xs text-muted-foreground mt-1">The CSV must have a header row with at least one column containing review text</p>
                                    </>
                                )}
                            </div>

                            {/* Column picker */}
                            {csvHeaders.length > 0 && (
                                <div className="flex items-center gap-3">
                                    <Label className="shrink-0">Review text column:</Label>
                                    <Select value={selectedColumn} onValueChange={setSelectedColumn}>
                                        <SelectTrigger className="flex-1"><SelectValue placeholder="Select column…" /></SelectTrigger>
                                        <SelectContent>{csvHeaders.map((h) => <SelectItem key={h} value={h}>{h}</SelectItem>)}</SelectContent>
                                    </Select>
                                </div>
                            )}

                            {/* CSV preview */}
                            {csvRows.length > 0 && (
                                <div className="rounded-lg border overflow-hidden">
                                    <div className="overflow-x-auto max-h-36">
                                        <table className="w-full text-xs">
                                            <thead className="bg-muted sticky top-0">
                                                <tr>{csvHeaders.map((h) => <th key={h} className={`px-3 py-2 text-left font-medium border-b ${h === selectedColumn ? "text-primary" : "text-muted-foreground"}`}>{h === selectedColumn ? `★ ${h}` : h}</th>)}</tr>
                                            </thead>
                                            <tbody>
                                                {csvRows.slice(0, 5).map((row, ri) => (
                                                    <tr key={ri} className="border-b last:border-0">
                                                        {row.map((cell, ci) => <td key={ci} className="px-3 py-1.5 truncate max-w-[200px]">{cell}</td>)}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                    {csvRows.length > 5 && <p className="text-xs text-muted-foreground text-center py-1.5 border-t">+{csvRows.length - 5} more rows not shown</p>}
                                </div>
                            )}

                            {error && (
                                <div className="flex items-center gap-2 p-3 rounded-lg bg-destructive/10 text-destructive text-sm">
                                    <AlertTriangle className="h-4 w-4 shrink-0" />{error}
                                </div>
                            )}
                        </CardContent>
                    </Card>

                    {/* Settings card */}
                    <Card>
                        <CardHeader><CardTitle className="text-base">Processing Settings</CardTitle></CardHeader>
                        <CardContent className="space-y-5">
                            <div className="space-y-1 p-3 rounded-lg bg-muted/50 text-sm">
                                <p className="font-medium">What will be analysed?</p>
                                <ul className="text-xs text-muted-foreground space-y-1 mt-1">
                                    <li>• Positive / Negative / Neutral / Null per aspect</li>
                                    <li>• Mixed reviews (conflicting aspects)</li>
                                    <li>• Confidence values per aspect</li>
                                    <li>• Overall sentiment breakdown</li>
                                </ul>
                            </div>
                            {csvRows.length > 0 && (
                                <div className="p-3 rounded-lg border text-sm">
                                    <p className="font-medium">{csvRows.length} reviews ready</p>
                                    <p className="text-xs text-muted-foreground mt-0.5">Column: <span className="text-primary font-mono">{selectedColumn}</span></p>
                                </div>
                            )}
                            {loading && (
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-xs text-muted-foreground"><span>Processing…</span><span>{progress}%</span></div>
                                    <div className="h-2 rounded-full bg-muted overflow-hidden"><div className="h-full bg-primary rounded-full transition-all duration-500" style={{ width: `${progress}%` }} /></div>
                                </div>
                            )}
                            <Button className="w-full" onClick={handleProcess} disabled={!csvFile || !selectedColumn || loading}>
                                {loading ? <><Spinner className="h-4 w-4 mr-2" />Processing {csvRows.length} reviews…</> : <><PlayCircle className="h-4 w-4 mr-2" />Process Reviews</>}
                            </Button>
                        </CardContent>
                    </Card>
                </div>

                {/* ── Results Dashboard ──────────────────────────────────── */}
                {result && (
                    <>
                        {/* KPI Strip */}
                        <div className="grid gap-4 grid-cols-2 sm:grid-cols-3 lg:grid-cols-5">
                            <KpiCard label="Total Reviews" value={result.total_reviews} icon={<FileText className="h-4 w-4" />} sub={`${result.total_processed} processed successfully`} />
                            <KpiCard label="Positive" value={result.overall_counts.POS} icon={<TrendingUp className="h-4 w-4 text-emerald-500" />} sub={pct(result.overall_counts.POS, result.overall_counts)} color="emerald" />
                            <KpiCard label="Negative" value={result.overall_counts.NEG} icon={<TrendingDown className="h-4 w-4 text-red-500" />} sub={pct(result.overall_counts.NEG, result.overall_counts)} color="red" />
                            <KpiCard label="Neutral" value={result.overall_counts.NEU} icon={<Minus className="h-4 w-4 text-amber-500" />} sub={pct(result.overall_counts.NEU, result.overall_counts)} color="amber" />
                            <KpiCard label="Mixed Reviews" value={result.mixed_count} icon={<AlertTriangle className="h-4 w-4 text-orange-500" />} sub={`${((result.mixed_count / Math.max(result.total_processed, 1)) * 100).toFixed(1)}% of processed`} color="orange" />
                        </div>

                        {/* Charts Row */}
                        <div className="grid gap-6 lg:grid-cols-2">
                            {/* Stacked bar */}
                            <Card>
                                <CardHeader className="flex flex-row items-center justify-between">
                                    <CardTitle className="text-base">Sentiment by Aspect</CardTitle>
                                    <p className="text-xs text-muted-foreground">Stacked counts</p>
                                </CardHeader>
                                <CardContent>
                                    <ChartContainer config={chartConfig} className="h-[280px] w-full">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={stackedChartData} layout="vertical">
                                                <XAxis type="number" tickFormatter={(v: number) => String(v)} />
                                                <YAxis type="category" dataKey="aspect" width={90} tick={{ fontSize: 12 }} />
                                                <Tooltip content={({ active, payload, label }: any) => {
                                                    if (!active || !payload?.length) return null;
                                                    return (
                                                        <div className="bg-background border border-border rounded-lg px-3 py-2 shadow-lg text-sm">
                                                            <p className="font-medium capitalize mb-1">{label}</p>
                                                            {payload.map((entry: { dataKey: string; value: number; color?: string }) => (
                                                                <p key={entry.dataKey} className="flex items-center gap-2">
                                                                    <span className="w-2 h-2 rounded-full inline-block" style={{ background: entry.color }} />
                                                                    <span>{entry.dataKey}: {entry.value}</span>
                                                                </p>
                                                            ))}
                                                        </div>
                                                    );
                                                }} />
                                                <Legend formatter={(val: string) => chartConfig[val as keyof typeof chartConfig]?.label ?? val} />
                                                {(["POS", "NEG", "NEU", "NULL"] as const).map((label) => (
                                                    <Bar key={label} dataKey={label} stackId="a" fill={BAR_COLORS[label]} radius={label === "NULL" ? [0, 4, 4, 0] : undefined} />
                                                ))}
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </ChartContainer>
                                </CardContent>
                            </Card>

                            {/* Confidence heatmap */}
                            <Card>
                                <CardHeader><CardTitle className="text-base">Average Confidence by Aspect</CardTitle></CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        {aspectNames.map((asp) => {
                                            const conf = result.avg_confidence[asp] ?? 0;
                                            const summary = result.aspect_summary[asp];
                                            const dominant = (["POS", "NEG", "NEU", "NULL"] as const).reduce(
                                                (a, b) => summary[a] >= summary[b] ? a : b,
                                                "POS" as SentimentLabel
                                            );
                                            return (
                                                <div key={asp} className="flex items-center gap-3">
                                                    <span className="text-sm capitalize w-24 shrink-0">{asp}</span>
                                                    <div className="flex-1 h-5 rounded-full bg-muted overflow-hidden">
                                                        <div className="h-full rounded-full transition-all" style={{ width: `${(conf * 100).toFixed(1)}%`, background: BAR_COLORS[dominant], opacity: 0.8 }} />
                                                    </div>
                                                    <span className="text-xs text-muted-foreground w-12 text-right shrink-0">{(conf * 100).toFixed(1)}%</span>
                                                    <Badge variant="secondary" className={`text-xs ${SENTIMENT_COLORS[dominant]} shrink-0`}>{dominant}</Badge>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </CardContent>
                            </Card>
                        </div>

                        {/* Mixed Review Spotlight */}
                        {result.mixed_count > 0 && (
                            <Card className="border-orange-200 dark:border-orange-900/50">
                                <CardHeader className="flex flex-row items-center gap-2">
                                    <AlertTriangle className="h-5 w-5 text-orange-500" />
                                    <CardTitle className="text-base">Mixed Sentiment Reviews ({result.mixed_count})</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <p className="text-sm text-muted-foreground mb-3">These reviews expressed both <span className="text-emerald-600 font-medium">Positive</span> and <span className="text-red-600 font-medium">Negative</span> sentiments across different aspects.</p>
                                    <div className="overflow-x-auto rounded-lg border">
                                        <Table>
                                            <TableHeader>
                                                <TableRow>
                                                    <TableHead>#</TableHead>
                                                    <TableHead>Review Preview</TableHead>
                                                    <TableHead>Conflict Prob.</TableHead>
                                                    {aspectNames.map((a) => <TableHead key={a} className="capitalize">{a}</TableHead>)}
                                                </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                                {result.rows
                                                    .filter((row) => { const nn = row.aspects.map((a) => a.label).filter((l) => l !== "NULL"); return nn.includes("POS") && nn.includes("NEG"); })
                                                    .slice(0, 8)
                                                    .map((row) => {
                                                        const asp = row.aspects.reduce((acc, a) => ({ ...acc, [a.name]: a }), {} as Record<string, { label: SentimentLabel; confidence: number }>);
                                                        return (
                                                            <TableRow key={row.review_index}>
                                                                <TableCell className="text-muted-foreground text-xs">#{row.review_index + 1}</TableCell>
                                                                <TableCell className="max-w-[200px] truncate text-sm">{row.text?.slice(0, 90)}{(row.text?.length ?? 0) > 90 ? "…" : ""}</TableCell>
                                                                <TableCell><span className={`text-sm font-medium ${row.conflict_prob > 0.5 ? "text-red-600 dark:text-red-400" : "text-amber-600 dark:text-amber-400"}`}>{(row.conflict_prob * 100).toFixed(1)}%</span></TableCell>
                                                                {aspectNames.map((a) => {
                                                                    const d = asp[a];
                                                                    return <TableCell key={a}>{d ? <Badge variant="secondary" className={`text-xs ${SENTIMENT_COLORS[d.label]}`}>{d.label}</Badge> : <span className="text-muted-foreground text-xs">-</span>}</TableCell>;
                                                                })}
                                                            </TableRow>
                                                        );
                                                    })}
                                            </TableBody>
                                        </Table>
                                    </div>
                                </CardContent>
                            </Card>
                        )}

                        {/* Full Results Table */}
                        <Card>
                            <CardHeader className="flex flex-row items-center justify-between">
                                <div>
                                    <CardTitle className="text-base">Full Results</CardTitle>
                                    <p className="text-xs text-muted-foreground mt-0.5">{result.total_processed} reviews processed in {(result.timings.total_ms / 1000).toFixed(1)}s</p>
                                </div>
                                <Button size="sm" variant="outline" onClick={() => exportResultsCSV(result, aspectNames)}>
                                    <Download className="h-4 w-4 mr-1.5" />Export CSV
                                </Button>
                            </CardHeader>
                            <CardContent>
                                <div className="overflow-x-auto rounded-lg border">
                                    <Table>
                                        <TableHeader>
                                            <TableRow>
                                                <TableHead>#</TableHead>
                                                <TableHead>Review</TableHead>
                                                <TableHead>Mixed?</TableHead>
                                                <TableHead>Conflict</TableHead>
                                                <TableHead>XAI</TableHead>
                                                {aspectNames.map((a) => <TableHead key={a} className="capitalize">{a}</TableHead>)}
                                            </TableRow>
                                        </TableHeader>
                                        <TableBody>
                                            {result.rows.slice(0, 50).map((row) => {
                                                const aspMap = row.aspects.reduce((acc, a) => ({ ...acc, [a.name]: a }), {} as Record<string, { label: SentimentLabel; confidence: number }>);
                                                const nn = row.aspects.map((a) => a.label).filter((l) => l !== "NULL");
                                                const isMixed = nn.includes("POS") && nn.includes("NEG");
                                                return (
                                                    <TableRow key={row.review_index}>
                                                        <TableCell className="text-muted-foreground text-xs">#{row.review_index + 1}</TableCell>
                                                        <TableCell className="max-w-[180px] truncate text-sm">{row.text?.slice(0, 70)}{(row.text?.length ?? 0) > 70 ? "…" : ""}</TableCell>
                                                        <TableCell>{isMixed ? <Badge variant="secondary" className="bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400 text-xs">Mixed</Badge> : <CheckCircle2 className="h-4 w-4 text-muted-foreground" />}</TableCell>
                                                        <TableCell><span className={`text-xs font-medium ${row.conflict_prob > 0.5 ? "text-red-600" : "text-muted-foreground"}`}>{(row.conflict_prob * 100).toFixed(0)}%</span></TableCell>
                                                        <TableCell>
                                                            <Button size="sm" variant="ghost" className="h-7 px-2 text-xs" onClick={() => setExplainModalText(row.text)}>
                                                                <Sparkles className="h-3.5 w-3.5 mr-1" />Explain
                                                            </Button>
                                                        </TableCell>
                                                        {aspectNames.map((a) => {
                                                            const d = aspMap[a];
                                                            return (
                                                                <TableCell key={a}>
                                                                    {d ? (
                                                                        <div className="flex flex-col gap-0.5">
                                                                            <Badge variant="secondary" className={`text-xs ${SENTIMENT_COLORS[d.label]}`}>{d.label}</Badge>
                                                                            <span className="text-[10px] text-muted-foreground">{(d.confidence * 100).toFixed(0)}%</span>
                                                                        </div>
                                                                    ) : <span className="text-muted-foreground text-xs">-</span>}
                                                                </TableCell>
                                                            );
                                                        })}
                                                    </TableRow>
                                                );
                                            })}
                                        </TableBody>
                                    </Table>
                                    {result.rows.length > 50 && <p className="text-xs text-center py-2 text-muted-foreground border-t">Showing first 50 rows. Export CSV to see all {result.rows.length} results.</p>}
                                </div>
                            </CardContent>
                        </Card>
                    </>
                )}
            </div>

            {explainModalText !== null && (
                <XAIModal text={explainModalText} onClose={() => setExplainModalText(null)} />
            )}
        </>
    );
}
