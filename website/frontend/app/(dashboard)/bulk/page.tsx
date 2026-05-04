// app/(dashboard)/bulk/page.tsx
// This file controls the "Bulk Analysis Dashboard" where users can upload complex CSV files.
"use client";

// -- REACT HOOKS --
// useRef: Holds a direct reference to an HTML DOM element (like a hidden file input button).
// useCallback: Caches a function so we don't trigger unnecessary re-renders when passing functions.
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
} from "recharts"; // 3rd party library imported exclusively for plotting graphs!
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

// ── Dictionaries & Configurations ───────────────────────────────────────────
// Setting up standard colors so our tables and our charts look uniform.
const SENTIMENT_COLORS: Record<SentimentLabel, string> = {
    POS: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400",
    NEG: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
    NEU: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400",
    NULL: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
};

const BAR_COLORS: Record<string, string> = {
    POS: "#10b981", // Emerald hex code
    NEG: "#ef4444", // Red hex
    NEU: "#f59e0b", // Amber hex
    NULL: "#94a3b8",// Grey
};

// Maps labels to Rechart configurations.
const chartConfig = {
    POS: { label: "Positive", color: BAR_COLORS.POS },
    NEG: { label: "Negative", color: BAR_COLORS.NEG },
    NEU: { label: "Neutral", color: BAR_COLORS.NEU },
    NULL: { label: "Not Mentioned", color: BAR_COLORS.NULL },
};

// ── Page Main Component ─────────────────────────────────────────────────────────────

export default function BulkReviewsPage() {
    // 1. STATE INITIALIZATION
    const fileInputRef = useRef<HTMLInputElement>(null); // Links directly to the <input type="file"> via a "ref" handle.
    
    // UI Drag-and-drop state
    const [dragging, setDragging] = useState(false);
    
    // File contents saved in memory
    const [csvFile, setCsvFile] = useState<File | null>(null);
    const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
    const [csvRows, setCsvRows] = useState<string[][]>([]);
    const [selectedColumn, setSelectedColumn] = useState<string>("");
    
    // Loading APIs
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0); // Fake progress bar % 
    
    const [result, setResult] = useState<BulkPredictResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isMounted, setIsMounted] = useState(false);
    const [explainModalText, setExplainModalText] = useState<string | null>(null);

    // Prevents Next.js Hydration errors (UI looking strange for 1 second on load).
    useEffect(() => { setIsMounted(true); }, []);

    // 2. CSV LOGIC HANDLING
    // Function that runs when you successfully give it a File (e.g. from clicking "upload")
    const handleFile = useCallback((file: File) => {
        // Validation check
        if (!file.name.endsWith(".csv")) {
            setError("Please upload a CSV file (.csv)");
            return;
        }
        setError(null);
        setResult(null); // Clear old results if we upload a SECOND file!
        setCsvFile(file);
        
        // This is pure JavaScript logic for reading files from a user's hard drive into string memory.
        const reader = new FileReader();
        reader.onload = (e) => {
            // Once the file is loaded into memory string...
            const { headers, rows } = parseCSV(e.target?.result as string); // Hand it to a Helper function we wrote.
            setCsvHeaders(headers); // Save columns
            setCsvRows(rows); // Save raw data lists
            setSelectedColumn(headers[0] ?? ""); // Auto-pick column A as default text.
        };
        reader.readAsText(file); // Trigger the reading!
    }, []);

    // Function that triggers when you drag a file specifically onto the boxed area
    const onDrop = useCallback(
        (e: React.DragEvent<HTMLDivElement>) => {
            e.preventDefault(); // Prevents your browser from trying to open the spreadsheet directly!
            setDragging(false);
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
        },
        [handleFile]
    );

    // 3. API PROCESSING
    // Sends the massive list of texts to FastAPI Python server
    const handleProcess = async () => {
        const colIndex = csvHeaders.indexOf(selectedColumn); // Figure out if text is in col 0, col 1, etc.
        if (colIndex === -1 || !csvRows.length) return;
        
        // Loop over the CSV rows and pluck out JUST the column with the review text inside it!
        const reviews = csvRows.map((r) => r[colIndex] ?? "").filter((t) => t.trim());
        if (!reviews.length) { setError("No review text found in the selected column."); return; }

        setLoading(true);
        setProgress(0);
        setError(null);
        
        // While Python is calculating, mathematically inch the progress bar up so the user doesn't get bored.
        const interval = setInterval(() => setProgress((p) => Math.min(p + 2, 90)), 500);
        try {
            // FIRE THE BATCH PREDICT!
            setResult(await predictBulk(reviews, true));
            setProgress(100); // 100% finished!
        } catch (err) {
            setError(err instanceof Error ? err.message : "Processing failed");
        } finally {
            clearInterval(interval); // Kills the fake progress timer
            setLoading(false);
        }
    };

    // Calculate chart data from the final Result object.
    const aspectNames = result ? Object.keys(result.aspect_summary) : [];
    
    // Formats the data so Recharts library can read it clearly.
    const stackedChartData = aspectNames.map((asp) => ({
        aspect: asp.charAt(0).toUpperCase() + asp.slice(1),
        ...result!.aspect_summary[asp],
    }));

    if (!isMounted) return <div className="space-y-6 animate-pulse p-6"><div className="h-10 w-64 bg-muted rounded" /><div className="h-4 w-full max-w-lg bg-muted rounded mb-10" /></div>;

    // 4. TSX (HTML) Rendering
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
                            {/* Drop zone UI Box */}
                            {/* Uses template literals ` ` to combine tailwind conditions based on dragging state */}
                            <div
                                className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${dragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-muted/30"}`}
                                onDragOver={(e) => { e.preventDefault(); setDragging(true); }} // Fire dragging!
                                onDragLeave={() => setDragging(false)} // Fire not dragging!
                                onDrop={onDrop} // Fire on-drop logic!
                                onClick={() => fileInputRef.current?.click()} // Forward a mouse click exactly to the hidden input
                            >
                                {/* The physical HTML upload button that is visually hidden! */}
                                <input ref={fileInputRef} type="file" accept=".csv" className="hidden"
                                    onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
                                
                                <UploadCloud className="h-10 w-10 mx-auto mb-3 text-muted-foreground" />
                                
                                {/* If a file EXISTS in memory, we replace the drag text with the File Name */}
                                {csvFile ? (
                                    <div className="flex items-center justify-center gap-2">
                                        <FileText className="h-5 w-5 text-primary" />
                                        <span className="font-medium text-sm">{csvFile.name}</span>
                                        <span className="text-xs text-muted-foreground">({csvRows.length} rows, {csvHeaders.length} columns)</span>
                                        {/* A button to clear memories if you select wrong file */}
                                        <button onClick={(e) => { e.stopPropagation(); setCsvFile(null); setCsvHeaders([]); setCsvRows([]); setResult(null); }} className="ml-1 text-muted-foreground hover:text-destructive transition-colors"><X className="h-4 w-4" /></button>
                                    </div>
                                ) : (
                                    <>
                                        <p className="text-sm font-medium">Drag & drop a CSV file here, or click to browse</p>
                                        <p className="text-xs text-muted-foreground mt-1">The CSV must have a header row with at least one column containing review text</p>
                                    </>
                                )}
                            </div>

                            {/* Column picker (Dropdown that dynamically builds `<SelectItem>` items for every Header in the CSV) */}
                            {csvHeaders.length > 0 && (
                                <div className="flex items-center gap-3">
                                    <Label className="shrink-0">Review text column:</Label>
                                    <Select value={selectedColumn} onValueChange={setSelectedColumn}>
                                        <SelectTrigger className="flex-1"><SelectValue placeholder="Select column…" /></SelectTrigger>
                                        <SelectContent>{csvHeaders.map((h) => <SelectItem key={h} value={h}>{h}</SelectItem>)}</SelectContent>
                                    </Select>
                                </div>
                            )}

                            {/* CSV Mini preview (just to show user it loaded properly) */}
                            {csvRows.length > 0 && (
                                <div className="rounded-lg border overflow-hidden">
                                    <div className="overflow-x-auto max-h-36">
                                        <table className="w-full text-xs">
                                            <thead className="bg-muted sticky top-0">
                                                <tr>{csvHeaders.map((h) => <th key={h} className={`px-3 py-2 text-left font-medium border-b ${h === selectedColumn ? "text-primary" : "text-muted-foreground"}`}>{h === selectedColumn ? `★ ${h}` : h}</th>)}</tr>
                                            </thead>
                                            <tbody>
                                                {/* Only draw the first 5 rows! Slice array at index 5. */}
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

                    {/* Settings card Component */}
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
                            
                            {/* Loading UI Progress Bar Element */}
                            {loading && (
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-xs text-muted-foreground"><span>Processing…</span><span>{progress}%</span></div>
                                    <div className="h-2 rounded-full bg-muted overflow-hidden">
                                        <div className="h-full bg-primary rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
                                    </div>
                                </div>
                            )}
                            <Button className="w-full" onClick={handleProcess} disabled={!csvFile || !selectedColumn || loading}>
                                {loading ? <><Spinner className="h-4 w-4 mr-2" />Processing {csvRows.length} reviews…</> : <><PlayCircle className="h-4 w-4 mr-2" />Process Reviews</>}
                            </Button>
                        </CardContent>
                    </Card>
                </div>

                {/* ── Results Dashboard (Only spawns if 'result' state is populated by Python) ──────────────────────────────────── */}
                {result && (
                    <>
                        {/* KPI Strip */}
                        {/* Custom visual Component <KpiCard> made by you. Passes numbers through dynamically! */}
                        <div className="grid gap-4 grid-cols-2 sm:grid-cols-3 lg:grid-cols-5">
                            <KpiCard label="Total Reviews" value={result.total_reviews} icon={<FileText className="h-4 w-4" />} sub={`${result.total_processed} processed successfully`} />
                            <KpiCard label="Positive" value={result.overall_counts.POS} icon={<TrendingUp className="h-4 w-4 text-emerald-500" />} sub={pct(result.overall_counts.POS, result.overall_counts)} color="emerald" />
                            <KpiCard label="Negative" value={result.overall_counts.NEG} icon={<TrendingDown className="h-4 w-4 text-red-500" />} sub={pct(result.overall_counts.NEG, result.overall_counts)} color="red" />
                            <KpiCard label="Neutral" value={result.overall_counts.NEU} icon={<Minus className="h-4 w-4 text-amber-500" />} sub={pct(result.overall_counts.NEU, result.overall_counts)} color="amber" />
                            <KpiCard label="Mixed Reviews" value={result.mixed_count} icon={<AlertTriangle className="h-4 w-4 text-orange-500" />} sub={`${((result.mixed_count / Math.max(result.total_processed, 1)) * 100).toFixed(1)}% of processed`} color="orange" />
                        </div>

                        {/* Charts Row */}
                        <div className="grid gap-6 lg:grid-cols-2">
                            {/* Stacked bar using "Recharts" library */}
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
                                                }} /> {/* Mathematical tooltips rendered on hover */}
                                                <Legend formatter={(val: string) => chartConfig[val as keyof typeof chartConfig]?.label ?? val} />
                                                {/* Iterates dynamically through the 4 states and fills colored bars in Chart */}
                                                {(["POS", "NEG", "NEU", "NULL"] as const).map((label) => (
                                                    <Bar key={label} dataKey={label} stackId="a" fill={BAR_COLORS[label]} radius={label === "NULL" ? [0, 4, 4, 0] : undefined} />
                                                ))}
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </ChartContainer>
                                </CardContent>
                            </Card>

                            {/* Confidence heatmap visually crafted using absolute math and css backgrounds */}
                            <Card>
                                <CardHeader><CardTitle className="text-base">Average Confidence by Aspect</CardTitle></CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        {/* Loops through price, quality, smell etc... */}
                                        {aspectNames.map((asp) => {
                                            const conf = result.avg_confidence[asp] ?? 0;
                                            const summary = result.aspect_summary[asp];
                                            
                                            // Array reducer: finds the most popular sentiment (Math.max basically)
                                            const dominant = (["POS", "NEG", "NEU", "NULL"] as const).reduce(
                                                (a, b) => summary[a] >= summary[b] ? a : b,
                                                "POS" as SentimentLabel
                                            );
                                            
                                            return (
                                                <div key={asp} className="flex items-center gap-3">
                                                    <span className="text-sm capitalize w-24 shrink-0">{asp}</span>
                                                    <div className="flex-1 h-5 rounded-full bg-muted overflow-hidden">
                                                        {/* CSS width matches the confidence! Example: 90% confidence = 90% width element */}
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
                        
                        {/* More complex Tables below here parsing the massive Bulk objects */}
                    </>
                )}
            </div>

            {/* This is a pop-up Modal overlay. It is invisible UNLESS explainModalText is not null! */}
            {/* We pass a custom property (props) called `onClose` which lets the child component force the parent state back to Null. */}
            {explainModalText !== null && (
                <XAIModal text={explainModalText} onClose={() => setExplainModalText(null)} />
            )}
        </>
    );
}
