// components/bulk/XAIModal.tsx
// This file is a specific "Pop up Window" (Modal / Dialog) used on the Bulk page when
// a user clicks "Explain" on a single specific row of their uploaded CSV table.
"use client";

import { useState } from "react";
import { Sparkles, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
// Radix UI components that handle the black overlay and centered popup box mathematics 
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
// Re-using the exact same python API function normally used by the main XAI Page!
import { explain } from "@/lib/api";
import { ASPECTS } from "@/lib/types";
import type { Aspect, ExplanationBundle, ExplanationMethod } from "@/lib/types";

// ── Token highlight visualiser ──────────────────────────────────────────────
// Sub-component scoped only to this file. 
// Uses Integrated Gradients math scores to paint words dynamically green and red.
function TokenHighlightViewer({
    tokens,
}: {
    tokens: { token: string; attribution: number }[];
}) {
    // Normalization trick to find the highest number
    const maxAttr = Math.max(
        ...(tokens || []).map((t) => Math.abs(t.attribution || 0)),
        0.001
    );
    return (
        <div className="flex flex-wrap gap-1 p-3 rounded-lg bg-muted/50">
            {(tokens || []).map((t, i) => {
                const norm = (t.attribution || 0) / maxAttr;
                const isPos = norm > 0;
                const intensity = Math.abs(norm);
                // Math driving CSS color injection!
                const bg = isPos
                    ? `oklch(0.7 0.15 145 / ${intensity * 0.6 + 0.1})`
                    : `oklch(0.65 0.2 25 / ${intensity * 0.6 + 0.1})`;
                return (
                    <span
                        key={i}
                        className="px-1.5 py-0.5 rounded text-sm relative group cursor-default"
                        style={{
                            backgroundColor: bg,
                            color: intensity > 0.5 ? "white" : "inherit",
                        }}
                    >
                        {t.token}
                        {/* Hover Tooltip logic */}
                        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 text-xs rounded bg-popover text-popover-foreground shadow-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
                            attribution: {t.attribution.toFixed(3)}
                        </span>
                    </span>
                );
            })}
        </div>
    );
}

// ── XAI Modal Controller Component ─────────────────────────────────────────────────
// Accepts text from the Bulk dashboard, and a function `onClose` to shut itself off when X is clicked.
interface XAIModalProps {
    text: string;
    onClose: () => void;
}

export function XAIModal({ text, onClose }: XAIModalProps) {
    // 1. STATE MEMORY
    const [selectedAspect, setSelectedAspect] = useState<Aspect | "all">("all");
    const [selectedMethod, setSelectedMethod] = useState<ExplanationMethod>("ig");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<ExplanationBundle | null>(null);
    const [error, setError] = useState<string | null>(null);

    // 2. LOGIC
    // Talks directly to the python backend just like the main XAIPage does!
    const handleExplain = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await explain({
                text,
                aspect: selectedAspect,
                methods: [selectedMethod], // User selects LIME, SHAP, etc from a drop-down.
                msrEnabled: true,
                msrStrength: 0.5,
            });
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Explanation failed");
        } finally {
            setLoading(false);
        }
    };

    // 3. JSX RENDERER
    return (
        // The Dialog component listens to Escape key and clicks outside the box.
        // If triggered, it fires 'onOpenChange' which we mapped to fire the onClose() prop!
        <Dialog open onOpenChange={(open: boolean) => { if (!open) onClose(); }}>
            <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <Sparkles className="h-4 w-4 text-primary" />
                        Explain Review
                    </DialogTitle>
                </DialogHeader>

                <div className="space-y-4">
                    {/* Read-only Text box so user remembers exactly which row they clicked! */}
                    <div className="space-y-1.5">
                        <Label className="text-xs text-muted-foreground">Review Text</Label>
                        <Textarea value={text} readOnly rows={3} className="resize-none text-sm" />
                    </div>

                    <div className="flex flex-wrap gap-3">
                        {/* Aspect selector dropdown */}
                        <div className="flex-1 min-w-[160px] space-y-1.5">
                            <Label className="text-xs text-muted-foreground">Aspect</Label>
                            <Select
                                value={selectedAspect}
                                onValueChange={(v: string) => setSelectedAspect(v as Aspect | "all")}
                            >
                                <SelectTrigger><SelectValue /></SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="all">All aspects</SelectItem>
                                    {ASPECTS.map((a) => (
                                        <SelectItem key={a} value={a} className="capitalize">
                                            {a}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        {/* Method selector dropdown */}
                        <div className="flex-1 min-w-[160px] space-y-1.5">
                            <Label className="text-xs text-muted-foreground">Method</Label>
                            <Select
                                value={selectedMethod}
                                onValueChange={(v: string) => setSelectedMethod(v as ExplanationMethod)}
                            >
                                <SelectTrigger><SelectValue /></SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="ig">Integrated Gradients</SelectItem>
                                    <SelectItem value="lime">LIME</SelectItem>
                                    <SelectItem value="shap">SHAP</SelectItem>
                                    <SelectItem value="attention">Attention Weights</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="flex items-end">
                            <Button onClick={handleExplain} disabled={loading}>
                                {loading ? (
                                    <><Spinner className="h-4 w-4 mr-2" />Generating...</>
                                ) : (
                                    <><Sparkles className="h-4 w-4 mr-2" />Generate</>
                                )}
                            </Button>
                        </div>
                    </div>

                    {loading && (
                        <p className="text-xs text-muted-foreground text-center">
                            This may take a few minutes depending on the method...
                        </p>
                    )}

                    {error && (
                        <div className="flex items-center gap-2 p-3 rounded-lg bg-destructive/10 text-destructive text-sm">
                            <AlertTriangle className="h-4 w-4 shrink-0" />
                            {error}
                        </div>
                    )}

                    {/* Rendering the results using mapping and the TokenHighlightViewer */}
                    {result?.explanations && result.explanations.length > 0 && (
                        <div className="space-y-4">
                            <p className="text-xs text-muted-foreground">
                                Green = supports predicted sentiment · Red = opposes it
                            </p>
                            {/* Loop over every explanation (e.g. Price, Color, etc.) */}
                            {result.explanations.map((exp) => (
                                <div
                                    key={exp.aspect + "-" + exp.method}
                                    className="space-y-2 border rounded-lg p-4 bg-card"
                                >
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs font-semibold capitalize bg-primary/10 text-primary px-2 py-0.5 rounded">
                                            {exp.aspect}
                                        </span>
                                        <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded uppercase tracking-wider">
                                            {exp.method === "ig" ? "Integrated Gradients" : exp.method === "attention" ? "Attention Weights" : exp.method}
                                        </span>
                                    </div>
                                    <TokenHighlightViewer tokens={exp.tokens} />
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Fallback if Python completed but didn't actually send us valid math numbers */}
                    {result && (!result.explanations || result.explanations.length === 0) && (
                        <p className="text-sm text-muted-foreground text-center py-4">
                            No attribution data returned.
                        </p>
                    )}
                </div>
            </DialogContent>
        </Dialog>
    );
}
