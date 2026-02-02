"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Spinner } from "@/components/ui/spinner";
import { explain } from "@/lib/api";
import { ASPECTS, type Aspect, type ExplanationBundle, type ExplanationMethod } from "@/lib/types";
import { Sparkles, ChevronDown, ArrowUp, ArrowDown } from "lucide-react";

export default function XAIPage() {
  const [text, setText] = useState(
    "This lipstick has amazing staying power and the color is beautiful, but the smell is too strong."
  );
  const [selectedAspect, setSelectedAspect] = useState<Aspect | "all">("all");
  const [msrEnabled, setMsrEnabled] = useState(false);
  const [msrStrength, setMsrStrength] = useState([0.3]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExplanationBundle | null>(null);
  const [jsonOpen, setJsonOpen] = useState(false);

  const handleExplain = async () => {
    setLoading(true);
    try {
      const response = await explain({
        text,
        aspect: selectedAspect,
        methods: ["ig", "lime", "shap"],
        msrEnabled,
        msrStrength: msrStrength[0],
      });
      setResult(response);
    } finally {
      setLoading(false);
    }
  };

  const getMethodExplanations = (method: ExplanationMethod) => {
    if (!result) return [];
    return result.explanations.filter((e) => e.method === method);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Explainable AI
        </h1>
        <p className="text-muted-foreground">
          Understand model predictions with attribution methods
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Input Section */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-base">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="xai-text">Review Text</Label>
              <Textarea
                id="xai-text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={3}
                className="resize-none"
              />
            </div>

            <div className="space-y-2">
              <Label>Aspect</Label>
              <Select
                value={selectedAspect}
                onValueChange={(v) => setSelectedAspect(v as Aspect | "all")}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All aspects</SelectItem>
                  {ASPECTS.map((aspect) => (
                    <SelectItem key={aspect} value={aspect} className="capitalize">
                      {aspect}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="xai-msr">Enable MSR</Label>
              <Switch
                id="xai-msr"
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
              onClick={handleExplain}
              disabled={!text.trim() || loading}
              className="w-full"
            >
              {loading ? (
                <>
                  <Spinner className="h-4 w-4 mr-2" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Generate Explanations
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-base">Attribution Results</CardTitle>
          </CardHeader>
          <CardContent>
            {result ? (
              <Tabs defaultValue="ig">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="ig">Integrated Gradients</TabsTrigger>
                  <TabsTrigger value="lime">LIME</TabsTrigger>
                  <TabsTrigger value="shap">SHAP</TabsTrigger>
                </TabsList>
                {(["ig", "lime", "shap"] as ExplanationMethod[]).map((method) => (
                  <TabsContent key={method} value={method} className="space-y-4 mt-4">
                    {getMethodExplanations(method).map((exp) => (
                      <div key={`${exp.aspect}-${exp.method}`} className="space-y-2">
                        <h4 className="text-sm font-medium capitalize">
                          {exp.aspect}
                        </h4>
                        <TokenHighlightViewer
                          tokens={exp.tokens}
                          showMsrDelta={msrEnabled}
                        />
                      </div>
                    ))}
                  </TabsContent>
                ))}
              </Tabs>
            ) : (
              <div className="h-48 flex items-center justify-center text-muted-foreground text-sm">
                Generate explanations to view attributions
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* MSR Delta Section */}
      {result && msrEnabled && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">MSR Delta</CardTitle>
            <p className="text-sm text-muted-foreground">
              Tokens that gained or lost importance when MSR is enabled
            </p>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <h4 className="text-sm font-medium text-green-600 dark:text-green-400 mb-2 flex items-center gap-1">
                  <ArrowUp className="h-4 w-4" />
                  Gained Importance
                </h4>
                <div className="flex flex-wrap gap-2">
                  {result.explanations
                    .flatMap((e) => e.tokens)
                    .filter((t) => t.msrDelta && t.msrDelta > 0.1)
                    .slice(0, 10)
                    .map((t, i) => (
                      <span
                        key={i}
                        className="px-2 py-1 rounded text-sm bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
                      >
                        {t.token}{" "}
                        <span className="text-xs opacity-75">
                          +{((t.msrDelta || 0) * 100).toFixed(0)}%
                        </span>
                      </span>
                    ))}
                </div>
              </div>
              <div>
                <h4 className="text-sm font-medium text-red-600 dark:text-red-400 mb-2 flex items-center gap-1">
                  <ArrowDown className="h-4 w-4" />
                  Lost Importance
                </h4>
                <div className="flex flex-wrap gap-2">
                  {result.explanations
                    .flatMap((e) => e.tokens)
                    .filter((t) => t.msrDelta && t.msrDelta < -0.1)
                    .slice(0, 10)
                    .map((t, i) => (
                      <span
                        key={i}
                        className="px-2 py-1 rounded text-sm bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
                      >
                        {t.token}{" "}
                        <span className="text-xs opacity-75">
                          {((t.msrDelta || 0) * 100).toFixed(0)}%
                        </span>
                      </span>
                    ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Raw JSON */}
      {result && (
        <Collapsible open={jsonOpen} onOpenChange={setJsonOpen}>
          <Card>
            <CollapsibleTrigger asChild>
              <CardHeader className="cursor-pointer hover:bg-accent/50 transition-colors">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Raw JSON Bundle</CardTitle>
                  <ChevronDown
                    className={`h-4 w-4 transition-transform ${jsonOpen ? "rotate-180" : ""}`}
                  />
                </div>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent>
                <pre className="text-xs overflow-auto max-h-96 p-4 rounded-lg bg-muted font-mono">
                  {JSON.stringify(result.rawJson, null, 2)}
                </pre>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}
    </div>
  );
}

function TokenHighlightViewer({
  tokens,
  showMsrDelta,
}: {
  tokens: { token: string; attribution: number; msrDelta?: number }[];
  showMsrDelta: boolean;
}) {
  const maxAttr = Math.max(...tokens.map((t) => Math.abs(t.attribution)));

  return (
    <div className="flex flex-wrap gap-1 p-3 rounded-lg bg-muted/50">
      {tokens.map((t, i) => {
        const normalizedAttr = t.attribution / maxAttr;
        const isPositive = normalizedAttr > 0;
        const intensity = Math.abs(normalizedAttr);

        return (
          <span
            key={i}
            className="px-1.5 py-0.5 rounded text-sm relative group cursor-default"
            style={{
              backgroundColor: isPositive
                ? `oklch(0.7 0.15 145 / ${intensity * 0.6 + 0.1})`
                : `oklch(0.65 0.2 25 / ${intensity * 0.6 + 0.1})`,
              color: intensity > 0.5 ? "white" : "inherit",
            }}
          >
            {t.token}
            {/* Tooltip */}
            <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 text-xs rounded bg-popover text-popover-foreground shadow-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
              attr: {t.attribution.toFixed(3)}
              {showMsrDelta && t.msrDelta !== undefined && (
                <span className="block">delta: {t.msrDelta.toFixed(3)}</span>
              )}
            </span>
          </span>
        );
      })}
    </div>
  );
}
