"use client";

import { useState } from "react";
import { Sparkles, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
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
import { explain } from "@/lib/api";
import { ASPECTS } from "@/lib/types";
import type { Aspect, ExplanationBundle, ExplanationMethod } from "@/lib/types";

// ── Token highlight visualiser ──────────────────────────────────────────────

function TokenHighlightViewer({
    tokens,
}: {
    tokens: { token: string; attribution: number }[];
}) {
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
                        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 text-xs rounded bg-popover text-popover-foreground shadow-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
                            attribution: {t.attribution.toFixed(3)}
                        </span>
                    </span>
                );
            })}
        </div>
    );
}

// ── XAI Modal ───────────────────────────────────────────────────────────────

interface XAIModalProps {
    text: string;
    onClose: () => void;
}

export function XAIModal({ text, onClose }: XAIModalProps) {
    const [selectedAspect, setSelectedAspect] = useState<Aspect | "all">("all");
    const [selectedMethod, setSelectedMethod] = useState<ExplanationMethod>("ig");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<ExplanationBundle | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleExplain = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await explain({
                text,
                aspect: selectedAspect,
                methods: [selectedMethod],
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

    return (
        <Dialog open onOpenChange={(open: boolean) => { if (!open) onClose(); }}>
            <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <Sparkles className="h-4 w-4 text-primary" />
                        Explain Review
                    </DialogTitle>
                </DialogHeader>

                <div className="space-y-4">
                    <div className="space-y-1.5">
                        <Label className="text-xs text-muted-foreground">Review Text</Label>
                        <Textarea value={text} readOnly rows={3} className="resize-none text-sm" />
                    </div>

                    <div className="flex flex-wrap gap-3">
                        {/* Aspect selector */}
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

                        {/* Method selector */}
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

                    {result?.explanations && result.explanations.length > 0 && (
                        <div className="space-y-4">
                            <p className="text-xs text-muted-foreground">
                                Green = supports predicted sentiment · Red = opposes it
                            </p>
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
                                            {exp.method === "ig" ? "Integrated Gradients" : exp.method}
                                        </span>
                                    </div>
                                    <TokenHighlightViewer tokens={exp.tokens} />
                                </div>
                            ))}
                        </div>
                    )}

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
