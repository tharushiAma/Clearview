// components/ClearViewDemo.tsx
// This is the "Master Container" component. It holds both the PredictTab and ExplainTab,
// and manages all the state (memory) so data can flow seamlessly between the two tabs.

// "use client" allows the component to use React states and browser APIs.
"use client";

import React, { useState } from "react";
// UI components for the tab switcher
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
// Our API functions to talk to FastAPI Python
import { fetchPrediction, fetchExplanation } from "@/lib/api";
import type { PredictResponse, ExplanationResponse } from "@/types";
// Sub-components that actually render the UI for each tab
import { PredictTab } from "@/components/demo/PredictTab";
import { ExplainTab } from "@/components/demo/ExplainTab";

export default function ClearViewDemo() {
  // --- 1. GLOBAL STATE (Shared memory) ---
  // Tells the UI which tab the user is currently looking at (default: 'predict')
  const [activeTab, setActiveTab] = useState("predict");
  
  // The common text input box. Because it's stored here in the parent,
  // typing in the Predict tab means the text is instantly ready for the Explain tab too!
  const [text, setText] = useState(
    "Lipstick color is amazing, I don't like the smell and the price is bit high."
  );

  // --- 2. PREDICT STATE (Memory specifically for the Prediction logic) ---
  const [isPredicting, setIsPredicting] = useState(false); // Controls the loading spinner
  const [prediction, setPrediction] = useState<PredictResponse | null>(null); // Stores the Python results
  const [predictError, setPredictError] = useState<string | null>(null); // Stores any crash messages

  // --- 3. EXPLAIN STATE (Memory specifically for the XAI logic) ---
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [isExplaining, setIsExplaining] = useState(false);
  const [explainAspect, setExplainAspect] = useState("all");
  const [explainError, setExplainError] = useState<string | null>(null);
  
  // This array stores the status of the "Analyzing X Aspect" steps to build a real-time progress bar.
  const [explainSteps, setExplainSteps] = useState<
    Array<{ name: string; status: "pending" | "progress" | "done" }>
  >([]);

  // --- 4. PREDICTION LOGIC ---
  const handlePredict = async () => {
    // Reset all memories to default before starting a new run
    setIsPredicting(true);
    setPrediction(null);
    setPredictError(null);

    // Creates a "Timer bomb". If python takes longer than 30 seconds, it cancels the UI to prevent it spinning forever.
    const timeout = setTimeout(() => {
      setIsPredicting(false);
      setPredictError(
        "Request timed out. The backend server may still be loading models (this takes ~60 seconds on first startup)."
      );
    }, 30000);

    try {
      // Send text to backend.
      const data = await fetchPrediction(text, 0.5, true);
      clearTimeout(timeout); // We succeeded! Disarm the timer bomb.
      setPrediction(data);   // Save the data to memory (which instantly updates the screen)
    } catch (e: unknown) {
      clearTimeout(timeout); // Disarm the bomb.
      setPredictError(
        (e instanceof Error ? e.message : null) ||
          "Prediction failed. Please ensure the backend server is running."
      );
    } finally {
      // Regardless of success or crash, shut off the loading spinner.
      setIsPredicting(false);
    }
  };

  // --- 5. XAI EXPLANATION LOGIC ---
  const handleExplain = async () => {
    setIsExplaining(true);
    setExplanation(null);
    setExplainError(null);

    // Decide which aspects to send to Python based on the dropdown menu
    const aspectsToAnalyze =
      explainAspect === "all"
        ? ["Color", "Texture", "Price", "Effect", "Packing"]
        : [explainAspect];

    // Build the visual progress tracker list
    const allSteps = [
      { name: "Loading XAI explainer", status: "pending" as const },
      { name: "Computing conflict explanation", status: "pending" as const },
      ...aspectsToAnalyze.map((asp) => ({ name: `Analyzing ${asp} aspect`, status: "pending" as const })),
      { name: "Finalizing results", status: "pending" as const },
    ];
    setExplainSteps(allSteps);

    // We create an "AbortController", which allows us to physically cancel the network request to Python if it takes too long.
    const controller = new AbortController();
    const timeout = setTimeout(() => {
      controller.abort(); // Cancel the request!
      setIsExplaining(false);
      setExplainError("XAI analysis timed out after 3 minutes.");
    }, 180000);

    // This is a fake UI interval that just visually bumps the steps from "pending" to "done" every 3 seconds
    // to give the user a sense of loading while the 3-minute API call calculates in the background.
    let currentStep = 0;
    const stepInterval = setInterval(() => {
      setExplainSteps((prev) => {
        const updated = [...prev];
        if (currentStep < updated.length)
          updated[currentStep] = { ...updated[currentStep], status: "done" };
        currentStep++;
        if (currentStep < updated.length)
          updated[currentStep] = { ...updated[currentStep], status: "progress" };
        return updated;
      });
    }, 3000);

    // Start the first step!
    setExplainSteps((prev) => {
      const updated = [...prev];
      updated[0] = { ...updated[0], status: "progress" };
      return updated;
    });

    try {
      // Fire the API call and wait! Note we pass 'controller.signal' so we can abort it.
      const raw = await fetchExplanation(text, explainAspect, 0.5, controller.signal);
      
      // Stop the timers!
      clearInterval(stepInterval);
      clearTimeout(timeout);
      
      // Force all steps to show as "done" (green checkmark)
      setExplainSteps((prev) => prev.map((s) => ({ ...s, status: "done" as const })));
      
      // Map the raw API response to the strict Typescript shape the UI demands.
      const mapped: ExplanationResponse = {
        text: raw.text,
        requested_aspect: explainAspect,
        ig_conflict: (raw.rawJson as any)?.ig_conflict,
        aspects: (raw.rawJson as any)?.aspects ?? {},
      };
      setExplanation(mapped);
    } catch (e: unknown) {
      clearInterval(stepInterval);
      clearTimeout(timeout);
      
      // If the error was manually triggered by our 3-minute abort button, do nothing (handled above).
      if (e instanceof Error && (e.name === "AbortError" || e.message.includes("aborted"))) return;
      
      setExplainError(
        (e instanceof Error ? e.message : null) ||
          "XAI analysis failed. Please ensure the backend server is running."
      );
    } finally {
      clearInterval(stepInterval);
      setIsExplaining(false);
    }
  };

  // --- 6. JSX RENDERING ---
  return (
    <div className="space-y-6">
      {/* The master Tabs wrapper. It uses our 'activeTab' state to know which tab is currently showing. */}
      <Tabs defaultValue="predict" value={activeTab} onValueChange={setActiveTab} className="w-full">
        {/* The clickable buttons to swap tabs */}
        <TabsList className="grid w-full grid-cols-2 lg:w-[400px] mx-auto">
          <TabsTrigger value="predict">Predict</TabsTrigger>
          <TabsTrigger value="explain">Explain</TabsTrigger>
        </TabsList>

        {/* --- PREDICT TAB CONTENT --- */}
        <TabsContent value="predict" className="space-y-6">
          {/* We inject our sub-component, passing down ALL the state variables and functions as "Properties" (Props) */}
          <PredictTab
            text={text}
            onTextChange={setText}
            isPredicting={isPredicting}
            prediction={prediction}
            error={predictError}
            onPredict={handlePredict}
          />
        </TabsContent>

        {/* --- EXPLAIN TAB CONTENT --- */}
        <TabsContent value="explain" className="space-y-6">
          <ExplainTab
            text={text}
            prediction={prediction}
            isExplaining={isExplaining}
            explanation={explanation}
            explainAspect={explainAspect}
            onAspectChange={setExplainAspect}
            explainSteps={explainSteps}
            error={explainError}
            onExplain={handleExplain}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
