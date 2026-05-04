// app/(dashboard)/demo/page.tsx
// This file is the "Live Demo" page of your application (accessible at www.yoursite.com/demo).
// Since it's inside the Next.js `app` router, the filename MUST be `page.tsx`.

// The word "use client" tells Next.js: "This page needs to run in the user's browser, NOT on the server."
// We need this because we use click events, typing events, and 'useState' memory.
"use client";

// We import React Hooks (useState for memory, useEffect for triggering actions).
import { useState, useEffect } from "react";

// The rest of these imports are UI Components (custom Legos we built).
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
import { predict } from "@/lib/api"; // This imports the "middleman" python caller we viewed earlier!
import type { PredictionResult, AspectPrediction, SentimentLabel } from "@/lib/types"; // TypeScript type limits to prevent coding errors.
import { useToast } from "@/hooks/use-toast";
import { Play, AlertTriangle } from "lucide-react"; // Import tiny SVG icons.

// A constant dictionary mapping labels to Tailwind CSS color classes.
const SENTIMENT_COLORS: Record<SentimentLabel, string> = {
  NEG: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
  NEU: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400",
  POS: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400",
  NULL: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
};

// ---------------------------------------------------------------------------------------------------
// The main rendering function of the Demo Page
// ---------------------------------------------------------------------------------------------------
export default function DemoPage() {
  const { toast } = useToast(); // Loads the push notification system.

  // 1. STATE CONFIGURATION (The Component's Memory)
  // `text` stores what the user types in the textarea. Default is a long cosmetics review.
  const [text, setText] = useState(
    "The color is beautiful as same as the picture, but the smell is bit strong for a lipstick and this is too expensive compared to other stores"
  );
  
  // `loading` stores true/false representing if the Python API is currently calculating.
  const [loading, setLoading] = useState(false);
  
  // `result` stores the final prediction data once Python responds. Originally null.
  const [result, setResult] = useState<PredictionResult | null>(null);
  
  // `isMounted` forces the page to delay rendering until the browser is ready, preventing visual glitches.
  const [isMounted, setIsMounted] = useState(false);

  // 2. LIFECYCLE EFFECTS
  // When the component first drops onto the screen, this flips `isMounted` to true.
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // 3. EVENT HANDLERS
  // When you click the "Run Prediction" button, this code runs.
  const handlePredict = async () => {
    setLoading(true); // Turns the button into a spinning wheel immediately.
    try {
      // Calls Python. Await means "Pause this function and wait for Python to reply".
      const response = await predict({
        text, // The state of our textarea!
        msrEnabled: true,
        msrStrength: 0.5,
      });
      setResult(response); // We got response, save it to memory. The page will instantly redraw!
    } catch (err) {
      // If Python throws an error or the server is down, show a red error notification box.
      toast({
        variant: "destructive",
        title: "Prediction failed",
        description:
          err instanceof Error
            ? err.message
            : "An unexpected error occurred. Please try again.",
      });
    } finally {
      // No matter if it succeeded or crashed, stop the spinning wheel button.
      setLoading(false);
    }
  };

  // 4. HYDRATION GUARD
  // If the page hasn't finished loading in browser memory, show a blank grey skeleton box instead.
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

  // 5. THE VISUAL INTERFACE (The JSX Code)
  // Everything below here creates the HTML and CSS actually seen on the screen!
  return (
    <div className="space-y-6">
      {/* Title section */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Live Demo</h1>
        <p className="text-muted-foreground">
          Test the ABSA model with custom reviews
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* -- LEFT COLUMN: User Input Modal -- */}
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
                value={text} // Link the textbox visually to our 'text' state
                onChange={(e) => setText(e.target.value)} // When you type, update the 'text' memory!
                rows={4}
                className="resize-none"
              />
            </div>

            {/* Run Button */}
            <Button
              onClick={handlePredict} // Fire our function when clicked
              disabled={!text.trim() || loading} // Button goes grey if text box is empty OR if already loading
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

        {/* -- RIGHT COLUMN: Conflict Panel -- */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Conflict Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            {/* React shorthand: `{result ? ( show this ) : ( show that )}` */}
            {/* This conditionally hides the analysis if we haven't clicked predict yet! */}
            {result ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                  <div>
                    <p className="text-sm font-medium">Conflict Probability</p>
                    <p className="text-3xl font-bold">
                      {/* Convert math probability (0.8) to Percentage (80.0%) */}
                      {(result.conflictProbability * 100).toFixed(1)}%
                    </p>
                  </div>
                  {/* Draws a dynamic conic gradient circle chart based on the probablity math */}
                  <div
                    className="h-16 w-16 rounded-full flex items-center justify-center"
                    style={{
                      background: `conic-gradient(var(--chart-1) ${result.conflictProbability * 360}deg, var(--muted) 0deg)`,
                    }}
                  >
                    <div className="h-12 w-12 rounded-full bg-card" />
                  </div>
                </div>

                {/* If the mixed sentiment flag is true, physically render this alert box */}
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

      {/* -- BOTTOM ROW: Results Table -- */}
      {/* If result is NOT null, draw this Card. */}
      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Predictions</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Here we pass our data DOWN into a sub-component we built called PredictionTable */}
            <PredictionTable
              predictions={result.predictions}
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------
// A Sub-Component declared in the same file to keep things clean.
// It receives an array of 'predictions' from above, and simply draws a Table element.
// ---------------------------------------------------------------------------------------------------
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
          {/* We use `.map` to loop through the predictions array. 
              If the AI found 3 aspects, it loops 3 times and draws 3 <TableRow> tags! */}
          {(predictions || []).map((pred) => (
            <TableRow key={pred.aspect}>
              <TableCell className="font-medium capitalize">
                {pred.aspect}
              </TableCell>
              <TableCell>
                {/* Dynamically assign the CSS color (red/green) based on label mapping! */}
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
