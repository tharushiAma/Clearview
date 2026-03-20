"use client";

import React, { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { fetchPrediction, fetchExplanation } from "@/lib/api";
import type { PredictResponse, ExplanationResponse } from "@/types";
import { PredictTab } from "@/components/demo/PredictTab";
import { ExplainTab } from "@/components/demo/ExplainTab";

export default function ClearViewDemo() {
  const [activeTab, setActiveTab] = useState("predict");
  const [text, setText] = useState(
    "Lipstick color is amazing, I don't like the smell and the price is bit high."
  );

  // Predict state
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);

  // Explain state
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [isExplaining, setIsExplaining] = useState(false);
  const [explainAspect, setExplainAspect] = useState("all");
  const [explainError, setExplainError] = useState<string | null>(null);
  const [explainSteps, setExplainSteps] = useState<
    Array<{ name: string; status: "pending" | "progress" | "done" }>
  >([]);

  const handlePredict = async () => {
    setIsPredicting(true);
    setPrediction(null);
    setPredictError(null);

    const timeout = setTimeout(() => {
      setIsPredicting(false);
      setPredictError(
        "Request timed out. The backend server may still be loading models (this takes ~60 seconds on first startup)."
      );
    }, 30000);

    try {
      const data = await fetchPrediction(text, 0.5, true);
      clearTimeout(timeout);
      setPrediction(data);
    } catch (e: unknown) {
      clearTimeout(timeout);
      setPredictError(
        (e instanceof Error ? e.message : null) ||
          "Prediction failed. Please ensure the backend server is running."
      );
    } finally {
      setIsPredicting(false);
    }
  };

  const handleExplain = async () => {
    setIsExplaining(true);
    setExplanation(null);
    setExplainError(null);

    const aspectsToAnalyze =
      explainAspect === "all"
        ? ["Color", "Texture", "Price", "Effect", "Packing"]
        : [explainAspect];

    const allSteps = [
      { name: "Loading XAI explainer", status: "pending" as const },
      { name: "Computing conflict explanation", status: "pending" as const },
      ...aspectsToAnalyze.map((asp) => ({ name: `Analyzing ${asp} aspect`, status: "pending" as const })),
      { name: "Finalizing results", status: "pending" as const },
    ];
    setExplainSteps(allSteps);

    const controller = new AbortController();
    const timeout = setTimeout(() => {
      controller.abort();
      setIsExplaining(false);
      setExplainError("XAI analysis timed out after 3 minutes.");
    }, 180000);

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

    setExplainSteps((prev) => {
      const updated = [...prev];
      updated[0] = { ...updated[0], status: "progress" };
      return updated;
    });

    try {
      const raw = await fetchExplanation(text, explainAspect, 0.5, controller.signal);
      clearInterval(stepInterval);
      clearTimeout(timeout);
      setExplainSteps((prev) => prev.map((s) => ({ ...s, status: "done" as const })));
      // Map the raw API response to the ExplanationResponse shape from @/types
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

  return (
    <div className="space-y-6">
      <Tabs defaultValue="predict" value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2 lg:w-[400px] mx-auto">
          <TabsTrigger value="predict">Predict</TabsTrigger>
          <TabsTrigger value="explain">Explain</TabsTrigger>
        </TabsList>

        <TabsContent value="predict" className="space-y-6">
          <PredictTab
            text={text}
            onTextChange={setText}
            isPredicting={isPredicting}
            prediction={prediction}
            error={predictError}
            onPredict={handlePredict}
          />
        </TabsContent>

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
