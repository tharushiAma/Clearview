// app/(dashboard)/xai/page.tsx
// This file is responsible for the "Explainable AI" page where users dive deep into model attribution.
// The "use client" directive enables state, effects, and interactivity in the browser.
"use client";

// -- REACT HOOKS --
// useState: Saves data that changes the visual UI.
// useEffect: Runs code automatically (e.g. when page loads).
// useCallback: Caches a function in memory so it isn't recreated on every keystroke, improving performance.
import { useState, useEffect, useCallback } from "react";

// -- UI COMPONENTS --
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
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
import { predict, explain } from "@/lib/api"; // We pull in the API middleman functions!
import { ASPECTS, type Aspect, type ExplanationBundle, type ExplanationMethod } from "@/lib/types";
import { Sparkles, ChevronDown, Zap } from "lucide-react";

export default function XAIPage() {
  // 1. STATE INITIALIZATION
  // Storing the textual review currently being analyzed.
  const [text, setText] = useState(
    "The color is beautiful as same as the picture, but the smell is bit strong for a lipstick and this is too expensive compared to other stores"
  );
  
  // Which dropdown options the user has clicked
  const [selectedAspect, setSelectedAspect] = useState<Aspect | "all">("all");
  const [selectedMethod, setSelectedMethod] = useState<ExplanationMethod>("ig");
  
  // Loading states for different UI elements
  const [loading, setLoading] = useState(false); // For the slow Math-heavy XAI
  const [fastLoading, setFastLoading] = useState(false); // For the fast Attention tokens
  
  // Memory to store the data that comes back from Python
  const [result, setResult] = useState<ExplanationBundle | null>(null);
  const [predictions, setPredictions] = useState<{ aspect: string; label: string; confidence: number }[]>([]);
  const [fastTokens, setFastTokens] = useState<{ aspect: string; tokens: { token: string; attribution: number }[] }[]>([]);
  
  // UI states for expanding/collapsing sections
  const [jsonOpen, setJsonOpen] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  // 2. TIMING (LIFECYCLES)
  // Run instantly once when the page is loaded to display the UI.
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // 3. API CALL LOGIC
  // Memory-cached function (useCallback) that triggers the fast /predict route to get instantaneous token attention.
  const handleFastAttribution = useCallback(async (reviewText: string) => {
    if (!reviewText.trim()) return; // Stop if text box is empty!
    
    setFastLoading(true);
    setFastTokens([]); // Clear old visual data
    setPredictions([]); 
    
    try {
      const res = await predict({ text: reviewText, msrEnabled: true, msrStrength: 0.5 });
      
      // Save the overall Positive/Negative results into our state.
      setPredictions(
        (res.predictions || []).map((p: any) => ({
          aspect: p.aspect,
          label: p.label,
          confidence: p.confidence,
        }))
      );
      
      // Map the "topTokens" array from Python into an array of objects we can safely draw on screen.
      const tokens = (res.predictions || [])
        .filter((p: any) => p.topTokens && p.topTokens.length > 0)
        .map((p: any) => ({
          aspect: p.aspect,
          tokens: (p.topTokens as any[]).map((t: any, idx: number) => ({
            token: typeof t === "string" ? t : (t[0] ?? t.token ?? String(t)),
            attribution: typeof t === "string"
              ? 1.0 - idx * 0.1          
              : (Number(t[1] ?? t.attribution) || 0),
          })),
        }));
      setFastTokens(tokens); // This triggers React to visually draw the grey Keyword boxes!
    } catch {
      setFastTokens([]);
      setPredictions([]);
    } finally {
      setFastLoading(false);
    }
  }, []);

  // Run fast attribution AUTOMATICALLY as soon as the page finishes mounting.
  useEffect(() => {
    if (isMounted) handleFastAttribution(text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isMounted]);

  // Handle the heavy calculations (SHAP / LIME / IG).
  const handleExplain = async () => {
    setLoading(true);
    try {
      // Calls the /api/explain route. This hits Python and can take 2-4 minutes.
      const response = await explain({
        text,
        aspect: selectedAspect,
        methods: [selectedMethod], // Sends the selected method from the dropdown box!
        msrEnabled: true,
        msrStrength: 0.5,
      });
      setResult(response);
    } finally {
      setLoading(false);
    }
  };

  // Safe fallback while the browser loads the Next.js chunk files.
  if (!isMounted) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-10 w-48 bg-muted rounded" />
        <div className="h-4 w-96 bg-muted rounded" />
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-1 h-96 bg-muted rounded-xl" />
          <div className="lg:col-span-2 h-[500px] bg-muted rounded-xl" />
        </div>
      </div>
    );
  }

  // 4. HTML / JSX RENDER LAYER
  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Explainable AI
        </h1>
        <p className="text-muted-foreground">
          Understand model predictions with state-of-the-art attribution methods
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* -- Configuration Panel (Left side) -- */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-base">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="xai-text">Review Text</Label>
              <Textarea
                id="xai-text"
                value={text} // Links the UI explicitly to the 'text' memory variable
                onChange={(e) => setText(e.target.value)} // As the user types, update the text state
                rows={3}
                className="resize-none"
              />
            </div>

            {/* Fast attribution button */}
            <Button
              variant="secondary"
              onClick={() => handleFastAttribution(text)}
              disabled={!text.trim() || fastLoading}
              className="w-full"
            >
              {fastLoading ? (
                <>
                  <Spinner className="h-4 w-4 mr-2" />
                  Loading...
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4 mr-2" />
                  Show Attribution Tokens
                </>
              )}
            </Button>
            <p className="text-xs text-muted-foreground text-center -mt-2">
              Fast · attention-based · runs instantly
            </p>

            <div className="border-t pt-4 space-y-3">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Advanced Methods</p>

              {/* The Aspect Dropdown Menu */}
              <div className="space-y-2">
                <Label>Aspect</Label>
                <Select
                  value={selectedAspect}
                  onValueChange={(v) => setSelectedAspect(v as Aspect | "all")} // When they click an option, save it!
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All aspects</SelectItem>
                    {/* Loop through all predefined aspects so we don't have to copy-paste the HTML */}
                    {ASPECTS.map((aspect) => (
                      <SelectItem key={aspect} value={aspect} className="capitalize">
                        {aspect}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* The Explanation Method Dropdown Menu */}
              <div className="space-y-2">
                <Label>Explanation Method</Label>
                <Select
                  value={selectedMethod}
                  onValueChange={(v) => setSelectedMethod(v as ExplanationMethod)} // When clicked, flip the state between ig/lime/shap!
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ig">Integrated Gradients (Captum)</SelectItem>
                    <SelectItem value="lime">LIME Text</SelectItem>
                    <SelectItem value="shap">SHAP Partitions</SelectItem>
                  </SelectContent>
                </Select>
              </div>

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

              {loading && (
                <p className="text-xs text-muted-foreground text-center">
                  This may take a few minutes…
                </p>
              )}
              {!loading && (
                <p className="text-xs text-muted-foreground text-center">
                  ⏱ XAI analysis takes 3-4 min per aspect. Select one aspect for faster results.
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* -- Attribution Results (Right side) -- */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-base">Attribution Results</CardTitle>
            <p className="text-sm text-muted-foreground">
              For signed attribution (green/red), use the Advanced Methods below.
            </p>
          </CardHeader>
          <CardContent>
            {/* Logic check: Only show the Fast Tokens if 'fastTokens' array is NOT empty and result is NULL. */}
            {fastTokens.length > 0 && !result && (
              <div className="space-y-4">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="h-4 w-4 text-muted-foreground" />
                  <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Key Words Detected</span>
                </div>
                {/* Loop (map) through every aspect (like Price, Sizing) and render a Keyword strip! */}
                {fastTokens.map((asp) => {
                  const pred = predictions.find((p) => p.aspect === asp.aspect);
                  return (
                    <div key={asp.aspect} className="space-y-2 border rounded-lg p-4 bg-card shadow-sm">
                      <div className="flex items-center gap-2 flex-wrap">
                        <h4 className="text-sm font-semibold capitalize bg-primary/10 text-primary px-2.5 py-1 rounded-md">
                          Aspect: {asp.aspect}
                        </h4>
                        {/* Custom sub-component that draws the Green POS / Red NEG badges */}
                        {pred && <SentimentBadge label={pred.label} confidence={pred.confidence} />}
                      </div>
                      {/* Passing the raw keyword data into our visual helper component! */}
                      <KeywordViewer tokens={asp.tokens} />
                    </div>
                  );
                })}
                <p className="text-xs text-muted-foreground pt-1">
                  ⚡ These are the most relevant words detected for each aspect. For full signed attribution (green = supports · red = opposes), use the Advanced Methods.
                </p>
              </div>
            )}

            {/* Logic check: Show the Advanced Math Models if 'result' state exists. */}
            {result && result.explanations && result.explanations.length > 0 ? (
              <div className="space-y-6 mt-4">
                {result.explanations.map((exp) => {
                  const pred = predictions.find((p) => p.aspect === exp.aspect);
                  return (
                    <div key={`${exp.aspect}-${exp.method}`} className="space-y-4 border rounded-lg p-5 bg-card shadow-sm">
                      <div className="flex items-center gap-3 flex-wrap mb-2">
                        <h4 className="text-sm font-semibold capitalize bg-primary/10 text-primary px-2.5 py-1 rounded-md">
                          Aspect: {exp.aspect}
                        </h4>
                        {pred && <SentimentBadge label={pred.label} confidence={pred.confidence} />}
                        <span className="text-xs font-medium text-muted-foreground bg-muted px-2.5 py-1 rounded-md uppercase tracking-wider">
                          Method: {exp.method === 'ig' ? 'Integrated Gradients' : exp.method}
                        </span>
                      </div>
                      {/* Color legend - explaining the mathematics physically! */}
                      <div className="flex items-center gap-4 text-xs text-muted-foreground pb-1 border-b">
                        <span className="font-medium">Attribution key:</span>
                        <span className="flex items-center gap-1.5">
                          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: 'oklch(0.7 0.15 145)' }} />
                          Green = supports the {pred?.label ?? 'predicted'} sentiment
                        </span>
                        <span className="flex items-center gap-1.5">
                          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: 'oklch(0.65 0.2 25)' }} />
                          Red = opposes it
                        </span>
                      </div>
                      {/* We pass the math data to a custom component that paints words red or green! */}
                      <TokenHighlightViewer tokens={exp.tokens} />
                    </div>
                  );
                })}
              </div>
            ) : result ? (
              // Fallback states
              <div className="h-32 flex items-center justify-center text-muted-foreground text-sm">
                No attribution data returned.
              </div>
            ) : fastTokens.length === 0 && !fastLoading ? (
              <div className="h-48 flex items-center justify-center text-muted-foreground text-sm">
                Click "Show Attribution Tokens" to begin
              </div>
            ) : fastLoading ? (
              <div className="h-48 flex items-center justify-center gap-2 text-muted-foreground text-sm">
                <Spinner className="h-4 w-4" /> Loading attribution tokens…
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>

      {/* -- Raw JSON Debugging Panel -- */}
      {/* If result is null, this is totally hidden. If it exists, we show a clickable collapsible menu! */}
      {result && (
        <Collapsible open={jsonOpen} onOpenChange={setJsonOpen}>
          <Card>
            <CollapsibleTrigger asChild>
              <CardHeader className="cursor-pointer hover:bg-accent/50 transition-colors">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Raw JSON Response</CardTitle>
                  <ChevronDown
                    className={`h-4 w-4 transition-transform ${jsonOpen ? "rotate-180" : ""}`}
                  />
                </div>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent>
                <pre className="text-xs overflow-auto max-h-96 p-4 rounded-lg bg-muted font-mono">
                  {/* JSON.stringify turns actual object into a readable block of text */}
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

// ── Sub-components for styling logic ──────────────────────────────────────────────

// A Dictionary holding css classes for badging.
const LABEL_CONFIG: Record<string, { bg: string; text: string; label: string }> = {
  POS: { bg: "bg-emerald-100 dark:bg-emerald-900/40", text: "text-emerald-700 dark:text-emerald-300", label: "Positive" },
  NEG: { bg: "bg-red-100 dark:bg-red-900/40",     text: "text-red-700 dark:text-red-300",         label: "Negative" },
  NEU: { bg: "bg-sky-100 dark:bg-sky-900/40",     text: "text-sky-700 dark:text-sky-300",         label: "Neutral"  },
  NULL: { bg: "bg-muted",                          text: "text-muted-foreground",                  label: "N/A"      },
};

// Takes in a label string and confidence percentage and spits out a styled badge!
function SentimentBadge({ label, confidence }: { label: string; confidence: number }) {
  const cfg = LABEL_CONFIG[label] ?? LABEL_CONFIG.NULL;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${cfg.bg} ${cfg.text}`}>
      {cfg.label}
      <span className="opacity-70">({(confidence * 100).toFixed(0)}%)</span>
    </span>
  );
}

// Visualises the numbers from SHAP/LIME into actual background colors on text
function TokenHighlightViewer({
  tokens,
}: {
  tokens: { token: string; attribution: number }[];
}) {
  // Find the highest absolute math score so we can normalize the colors cleanly.
  const maxAttr = Math.max(...(tokens || []).map((t) => Math.abs(Number(t.attribution) || 0)), 0.001);

  return (
    <div className="flex flex-wrap gap-1 p-3 rounded-lg bg-muted/50">
      {(tokens || []).map((t, i) => {
        const attr = Number(t.attribution) || 0;
        const normalizedAttr = attr / maxAttr;
        const isPositive = normalizedAttr > 0; // Number > 0 means mathematical support (Green)
        const intensity = Math.abs(normalizedAttr); // 0.1 is pale color, 1.0 is vibrant bold color.

        return (
          <span
            key={i}
            className="px-1.5 py-0.5 rounded text-sm relative group cursor-default"
            // We use inline styling here because Tailwind can't do exact decimal opacity calculations smoothly!
            style={{
              backgroundColor: isPositive
                ? `oklch(0.7 0.15 145 / ${intensity * 0.6 + 0.1})` // Green
                : `oklch(0.65 0.2 25 / ${intensity * 0.6 + 0.1})`, // Red
              color: intensity > 0.5 ? "white" : "inherit", // If background gets too dark, turn text white.
            }}
          >
            {t.token}
            {/* Tooltip on hover: We physically draw a hidden span, and "group-hover:opacity-100" makes it appear when mouse touches parents! */}
            <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 text-xs rounded bg-popover text-popover-foreground shadow-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
              attribution: {attr.toFixed(3)}
            </span>
          </span>
        );
      })}
    </div>
  );
}

// A simpler viewer for non-math tokens (just spits out strings in grey boxes)
function KeywordViewer({
  tokens,
}: {
  tokens: { token: string; attribution: number }[];
}) {
  return (
    <div className="flex flex-wrap gap-2 p-3 rounded-lg bg-muted/30">
      {(tokens || []).map((t, i) => (
        <span
          key={i}
          className="px-2.5 py-1 rounded-md text-sm font-medium bg-muted text-muted-foreground border shadow-sm cursor-default"
        >
          {t.token}
        </span>
      ))}
    </div>
  );
}
