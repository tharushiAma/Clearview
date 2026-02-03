"use client";

import React, { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { fetchPrediction, fetchExplanation /*, fetchMetrics, fetchLogs */ } from "@/lib/api";
import { PredictResponse, ExplanationResponse /*, MetricData, LogEntry */ } from "@/types";
import { Loader2, AlertTriangle, BrainCircuit } from "lucide-react";

export default function ClearViewDemo() {
  const [activeTab, setActiveTab] = useState("predict");
  
  // Predict State
  const [text, setText] = useState("Lipstick color is amazing and packaging was great, but the price is bit high.");
  const [msrEnabled, setMsrEnabled] = useState(true);
  const [msrStrength, setMsrStrength] = useState(0.3);
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);

  // Explain State
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [isExplaining, setIsExplaining] = useState(false);
  const [explainAspect, setExplainAspect] = useState("all");
  const [explainError, setExplainError] = useState<string | null>(null);
  const [explainSteps, setExplainSteps] = useState<Array<{name: string, status: 'pending' | 'progress' | 'done'}>>([]);

  // Metrics State
  // const [metrics, setMetrics] = useState<MetricData | null>(null);

  // Logs State
  // const [logs, setLogs] = useState<LogEntry[]>([]);

  /*
  useEffect(() => {
    if (activeTab === "metrics") loadMetrics();
    if (activeTab === "logs") loadLogs();
  }, [activeTab]);
  */

  const handlePredict = async () => {
    setIsPredicting(true);
    setPrediction(null);
    setPredictError(null);
    
    // Set timeout to prevent infinite loading
    const timeout = setTimeout(() => {
      setIsPredicting(false);
      setPredictError("Request timed out. The backend server may still be loading models (this takes ~60 seconds on first startup).");
    }, 30000); // 30 second timeout
    
    try {
      const data = await fetchPrediction(text, msrStrength, msrEnabled);
      clearTimeout(timeout);
      setPrediction(data);
    } catch (e: unknown) {
      clearTimeout(timeout);
      console.error(e);
      setPredictError(
        (e instanceof Error ? e.message : null) || "Prediction failed. Please ensure the backend server is running."
      );
    } finally {
      setIsPredicting(false);
    }
  };


  const handleExplain = async () => {
    setIsExplaining(true);
    setExplanation(null);
    setExplainError(null);
    
    // Define all steps upfront based on selected aspect
    const aspectsToAnalyze = explainAspect === "all" 
      ? ["Color", "Texture", "Price", "Effect", "Packing"]
      : [explainAspect];
    
    const allSteps = [
      { name: "Loading XAI explainer", status: 'pending' as const },
      { name: "Computing conflict explanation", status: 'pending' as const },
      ...aspectsToAnalyze.map(asp => ({ 
        name: `Analyzing ${asp} aspect`, 
        status: 'pending' as const 
      })),
      { name: "Finalizing results", status: 'pending' as const }
    ];
    
    setExplainSteps(allSteps);
    
    // Set timeout for XAI (longer since it's very compute-intensive)
    const timeoutDuration = 180000; // 3 minute timeout
    const controller = new AbortController();
    const timeout = setTimeout(() => {
      controller.abort();
      setIsExplaining(false);
      setExplainError("XAI analysis timed out after 3 minutes. The computation may be too complex.");
    }, timeoutDuration);
    
    // Simulate step progression (in real implementation, backend would send these)
    let currentStep = 0;
    const stepInterval = setInterval(() => {
      setExplainSteps(prev => {
        const updated = [...prev];
        // Mark current completed
        if (currentStep < updated.length) {
          updated[currentStep] = { ...updated[currentStep], status: 'done' };
        }
        // Mark next as in progress
        currentStep++;
        if (currentStep < updated.length) {
          updated[currentStep] = { ...updated[currentStep], status: 'progress' };
        }
        return updated;
      });
    }, 3000); // Progress every 3 seconds
    
    // Start first step
    setExplainSteps(prev => {
      const updated = [...prev];
      updated[0] = { ...updated[0], status: 'progress' };
      return updated;
    });
    
    try {
      const data = await fetchExplanation(text, explainAspect, msrStrength, controller.signal);
      clearInterval(stepInterval);
      clearTimeout(timeout);
      
      // Mark all steps as done
      setExplainSteps(prev => prev.map(s => ({ ...s, status: 'done' as const })));
      setExplanation(data);
    } catch (e: unknown) {
      clearInterval(stepInterval);
      clearTimeout(timeout);
      console.error(e);
      
      // If the error is an AbortError (timeout), we've already set the proper error message
      if (e instanceof Error && (e.name === 'AbortError' || e.message.includes('aborted'))) {
        return;
      }
      
      setExplainError(
        (e instanceof Error ? e.message : null) || "XAI analysis failed. Please ensure the backend server is running."
      );
    } finally {
      clearInterval(stepInterval);
      setIsExplaining(false);
    }
  };

/*
  const loadMetrics = async () => {
    try {
      const data = await fetchMetrics();
      setMetrics(data);
    } catch (e) {
      console.error(e);
    }
  };

  const loadLogs = async () => {
    try {
      const data = await fetchLogs();
      setLogs(data);
    } catch (e) {
      console.error(e);
    }
  };
*/

  return (
    <div className="space-y-6">
      <Tabs defaultValue="predict" value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2 lg:w-[400px] mx-auto">
          <TabsTrigger value="predict">Predict</TabsTrigger>
          <TabsTrigger value="explain">Explain</TabsTrigger>
          {/* <TabsTrigger value="metrics">Metrics</TabsTrigger> */}
          {/* <TabsTrigger value="logs">Logs</TabsTrigger> */}
        </TabsList>

        {/* PREDICT TAB */}
        <TabsContent value="predict" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BrainCircuit className="w-5 h-5 text-blue-500" />
                Input & Controls
              </CardTitle>
              <CardDescription>Enter review text and configure Multi-Aspect Sentiment Resolution (MSR).</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Review Text</Label>
                <Textarea 
                  value={text} 
                  onChange={(e) => setText(e.target.value)} 
                  rows={4}
                  className="font-mono text-sm"
                />
              </div>

              <div className="flex items-center justify-between gap-8 p-4 border rounded-lg bg-slate-50">
                <div className="flex items-center gap-4">
                  <div className="space-y-0.5">
                    <Label>Enable MSR</Label>
                    <p className="text-xs text-muted-foreground">Apply conflict resolution logic</p>
                  </div>
                  <Switch checked={msrEnabled} onCheckedChange={setMsrEnabled} />
                </div>
                
                <div className="flex-1 space-y-2">
                  <div className="flex justify-between">
                    <Label>MSR Strength (λ): {msrStrength}</Label>
                  </div>
                  <Slider 
                    value={[msrStrength]} 
                    max={1.0} step={0.1} 
                    onValueChange={(v) => setMsrStrength(v[0])} 
                    disabled={!msrEnabled}
                  />
                </div>

                <Button onClick={handlePredict} disabled={isPredicting} size="lg">
                  {isPredicting ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : "Run Prediction"}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Error Display for Predictions */}
          {predictError && (
            <Card className="border-red-200 bg-red-50">
              <CardContent className="pt-6">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                  <div className="flex-1">
                    <h4 className="font-semibold text-red-900 mb-1">Prediction Failed</h4>
                    <p className="text-sm text-red-700">{predictError}</p>
                    <p className="text-xs text-red-600 mt-2">Hint: Start the backend server with: <code className="bg-red-100 px-1 py-0.5 rounded">python backend_server.py</code></p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {prediction && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Conflict Score */}
              <Card className="lg:col-span-1 border-l-4 border-l-purple-500">
                <CardHeader>
                  <CardTitle>Conflict Detection</CardTitle>
                </CardHeader>
                <CardContent className="text-center space-y-4">
                    <div className="text-5xl font-bold text-slate-900">
                      {(prediction.conflict_prob * 100).toFixed(1)}%
                    </div>
                    <p className="text-sm text-muted-foreground">Probability of Aspect Conflict</p>
                    <Progress value={prediction.conflict_prob * 100} className="h-2" />
                    {prediction.conflict_prob > 0.5 ? (
                      <Badge variant="destructive" className="mt-2">High Conflict</Badge>
                    ) : (
                      <Badge variant="secondary" className="mt-2 bg-green-100 text-green-800">Coherent</Badge>
                    )}
                </CardContent>
              </Card>

              {/* Aspects Grid */}
              <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-4">
                {prediction.aspects.map((asp) => (
                  <Card key={asp.name} className={`relative ${asp.changed_by_msr ? 'border-yellow-400 bg-yellow-50/30' : ''}`}>
                    <CardHeader className="pb-2">
                      <div className="flex justify-between items-center">
                        <CardTitle className="capitalize text-lg">{asp.name}</CardTitle>
                        <Badge variant={
                          asp.label === 'positive' ? 'default' : 
                          asp.label === 'negative' ? 'destructive' : 
                          asp.label === 'neutral' ? 'secondary' : 'outline'
                        }>
                          {asp.label}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="text-sm space-y-2">
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Confidence</span>
                        <span>{(asp.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <Progress 
                         value={asp.confidence * 100} 
                         className={`h-1.5 ${asp.changed_by_msr ? 'bg-yellow-200' : ''}`}
                      />
                      
                      {asp.changed_by_msr && asp.before && (
                         <div className="mt-3 p-2 bg-white rounded border text-xs flex items-center gap-2">
                           <AlertTriangle className="w-3 h-3 text-yellow-600" />
                           <span className="text-gray-600">
                             Changed from <span className="font-semibold">{asp.before.label}</span> ({ (asp.before.confidence * 100).toFixed(0)}%)
                           </span>
                         </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </TabsContent>

        {/* EXPLAIN TAB */}
        <TabsContent value="explain" className="space-y-6">
           <Card>
            <CardHeader>
              <CardTitle>XAI Analysis</CardTitle>
              <CardDescription>Visualize token attributions using Integrated Gradients & SHAP.</CardDescription>
            </CardHeader>
            <CardContent>
               <div className="flex items-end gap-4">
                 <div className="flex-1">
                   <Label>Focus Aspect</Label>
                   <select 
                     className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background"
                     value={explainAspect}
                     onChange={(e) => setExplainAspect(e.target.value)}
                   >
                     <option value="all">Analyze All Aspects</option>
                     {prediction?.aspects.map(a => <option key={a.name} value={a.name}>{a.name}</option>)}
                   </select>
                 </div>
                  <Button onClick={handleExplain} disabled={isExplaining}>
                    {isExplaining ? <Loader2 className="animate-spin mr-2" /> : "Run XAI"}
                  </Button>
                </div>
                
                {/* Step-by-Step Progress Tracker */}
                {explainSteps.length > 0 && (
                  <div className="mt-4 p-4 border rounded-lg bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
                    <h4 className="text-sm font-semibold text-blue-900 mb-3">XAI Analysis Progress</h4>
                    <div className="space-y-2">
                      {explainSteps.map((step, idx) => (
                        <div key={idx} className="flex items-center gap-3">
                          {step.status === 'done' && (
                            <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center flex-shrink-0">
                              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                              </svg>
                            </div>
                          )}
                          {step.status === 'progress' && (
                            <Loader2 className="w-5 h-5 text-blue-600 animate-spin flex-shrink-0" />
                          )}
                          {step.status === 'pending' && (
                            <div className="w-5 h-5 rounded-full border-2 border-gray-300 flex-shrink-0"></div>
                          )}
                          <span className={`text-sm ${
                            step.status === 'done' ? 'text-green-700 font-medium' :
                            step.status === 'progress' ? 'text-blue-700 font-semibold' :
                            'text-gray-500'
                          }`}>
                            {step.name}
                          </span>
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-blue-600 mt-3">
                      {isExplaining ? 'This may take 1-3 minutes...' : 'Analysis complete!'}
                    </p>
                  </div>
                )}
            </CardContent>
           </Card>

           {/* Error Display for XAI */}
           {explainError && (
             <Card className="border-red-200 bg-red-50">
               <CardContent className="pt-6">
                 <div className="flex items-start gap-3">
                   <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                   <div className="flex-1">
                     <h4 className="font-semibold text-red-900 mb-1">XAI Analysis Failed</h4>
                     <p className="text-sm text-red-700">{explainError}</p>
                     <p className="text-xs text-red-600 mt-2">The XAI computation requires the backend server to be fully initialized.</p>
                   </div>
                 </div>
               </CardContent>
             </Card>
           )}

           {explanation && (
             <div className="space-y-8">
               {/* 1. Conflict Explanation */}
               {explanation.ig_conflict && (
                 <Card>
                   <CardHeader><CardTitle>Conflict Drivers</CardTitle></CardHeader>
                   <CardContent>
                     <p className="text-sm text-gray-500 mb-4">Tokens increasing conflict probability:</p>
                     <div className="flex flex-wrap gap-2">
                       {explanation.ig_conflict.top_tokens.map((t: [string, number], idx: number) => (
                         <span 
                           key={idx}
                           className="px-2 py-1 rounded text-sm font-mono"
                           style={{ backgroundColor: `rgba(239, 68, 68, ${Math.min(Math.abs(t[1]) * 5, 0.8)})`, color: Math.abs(t[1]) > 0.1 ? 'white' : 'black' }}
                         >
                           {t[0]}
                         </span>
                       ))}
                     </div>
                   </CardContent>
                 </Card>
               )}

               {/* 2. Per Aspect */}
               {Object.entries(explanation.aspects).map(([aspName, data]) => (
                 <Card key={aspName}>
                   <CardHeader><CardTitle className="capitalize">{aspName} Attribution</CardTitle></CardHeader>
                   <CardContent className="grid md:grid-cols-2 gap-6">
                     <div>
                       <h4 className="text-sm font-semibold mb-2">Integrated Gradients</h4>
                       <div className="flex flex-wrap gap-2">
                         {data.ig_aspect.top_tokens.map((t: [string, number], i: number) => (
                           <span 
                             key={i}
                             className="px-2 py-1 rounded text-sm font-mono border"
                             style={{ 
                               backgroundColor: t[1] > 0 ? `rgba(34, 197, 94, ${Math.min(t[1]*5, 0.6)})` : `rgba(239, 68, 68, ${Math.min(Math.abs(t[1])*5, 0.6)})` 
                             }}
                           >
                             {t[0]}
                           </span>
                         ))}
                       </div>
                     </div>
                     {data.msr_delta && (
                        <div>
                          <h4 className="text-sm font-semibold mb-2">MSR Impact (Delta)</h4>
                          <div className="text-xs space-y-1">
                            <div className="flex justify-between">
                              <span>Before Prob:</span>
                              <span className="font-mono">{JSON.stringify(data.msr_delta.prob_before.map((n:number) => Number(n.toFixed(2))))}</span>
                            </div>
                            <div className="flex justify-between font-bold">
                              <span>After Prob:</span>
                              <span className="font-mono">{JSON.stringify(data.msr_delta.prob_after.map((n:number) => Number(n.toFixed(2))))}</span>
                            </div>
                          </div>
                        </div>
                     )}
                   </CardContent>
                 </Card>
               ))}
             </div>
           )}
        </TabsContent>

        {/* METRICS TAB */}
        {/* 
        <TabsContent value="metrics">
          {metrics ? (
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">Overall Macro F1</CardTitle></CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {metrics.overall_macro_f1_4class != null && !isNaN(metrics.overall_macro_f1_4class)
                      ? `${(metrics.overall_macro_f1_4class * 100).toFixed(1)}%`
                      : 'N/A'}
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">Conflict AUC</CardTitle></CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {metrics.conflict?.roc_auc != null && !isNaN(metrics.conflict.roc_auc)
                      ? metrics.conflict.roc_auc.toFixed(3)
                      : 'N/A'}
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">MSR Error Reduction</CardTitle></CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-600">
                    {metrics.msr_error_reduction?.total_reduction != null
                      ? `+${metrics.msr_error_reduction.total_reduction}`
                      : 'N/A'}
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">Conflict F1</CardTitle></CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {metrics.conflict?.conf_f1_macro != null && !isNaN(metrics.conflict.conf_f1_macro)
                      ? `${(metrics.conflict.conf_f1_macro * 100).toFixed(1)}%`
                      : 'N/A'}
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
             <div className="flex justify-center p-8"><Loader2 className="animate-spin" /></div>
          )}
        </TabsContent>
        */}

        {/* LOGS TAB */}
        {/* 
        <TabsContent value="logs">
          <Card>
            <CardHeader><CardTitle>Training Logs</CardTitle></CardHeader>
            <CardContent>
              <div className="h-[400px] overflow-auto border rounded-md">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Epoch</TableHead>
                      <TableHead>Step</TableHead>
                      <TableHead>Loss</TableHead>
                      <TableHead>LR</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {logs.map((log, i) => (
                      <TableRow key={i}>
                        <TableCell>{log.epoch}</TableCell>
                        <TableCell>{log.step}</TableCell>
                        <TableCell>{log.loss?.toFixed(4)}</TableCell>
                        <TableCell>{log.lr?.toExponential(2)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        */}
      </Tabs>
    </div>
  );
}
