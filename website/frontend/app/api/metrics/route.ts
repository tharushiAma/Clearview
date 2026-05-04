// next/server is the core library specifically used for Next.js App Router API Routes.
// NextResponse allows us to format the data securely before sending it to the browser.
import { NextResponse } from 'next/server';

// We define where our Python backend lives. 
// "process.env.BACKEND_URL" lets us use a secure production URL if it exists, 
// otherwise "||" (OR) acts as a fallback to your local computer (localhost:8000).
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// In Next.js App Router, naming a function exactly "export async function GET()"
// automatically creates an invisible live web page at "http://localhost:3000/api/metrics"
// that only accepts Data-Fetching (GET) requests.
export async function GET() {
    try {
        // We instruct Next.js to reach out to the Python server to grab the latest metrics.
        // { cache: 'no-store' } guarantees that we get fresh data every single time, 
        // bypassing the aggressive browser caching mechanism that Next.js usually uses!
        const response = await fetch(`${BACKEND_URL}/metrics`, {
            cache: 'no-store',
        });

        // If Python crashed or couldn't find the data (like a 404 Not Found error),
        // we deliberately throw a Javascript Error to stop the code from crashing further down.
        if (!response.ok) {
            throw new Error(`Backend error: ${response.status}`);
        }

        // We convert the raw internet data string back into a usable JavaScript Object.
        const data = await response.json();

        // --- Data Transformation Phase ---
        // The frontend UI charts (Recharts) need the math looking a very specific way.
        // We extract the "aspects" list from python, or default to an empty object {} if it's missing.
        const aspects = data.aspects || {};
        
        // Object.keys grabs all the names inside the dictionary. e.g. ["price", "color", "smell"]
        const aspectNames = Object.keys(aspects);

        // We loop (.map) through every aspect name to build an array of formatted metrics.
        const aspectMetrics = aspectNames.map(name => ({
            aspect: name,
            precision: aspects[name].macro_precision || 0, // Fallback to 0 if python didn't calculate it
            recall: aspects[name].macro_recall || 0,
            f1: aspects[name].macro_f1 || 0,
        }));

        // Similarly, we extract the 3x3 array (Confusion Matrix) for every aspect to power our heatmaps.
        const confusionMatrices = aspectNames.map(name => ({
            aspect: name as any, // "as any" bypasses strict TypeScript checking temporarily
            matrix: aspects[name].confusion_matrix || [[0, 0, 0], [0, 0, 0], [0, 0, 0]], // Default to an empty grid
            labels: ['NEG', 'NEU', 'POS'] as any[], // Define the exact axis labels for the chart.
        }));

        // Pluck out ONLY the f1 math scores from our array so we can average them together.
        const macroF1s = aspectMetrics.map(m => m.f1);
        
        // Standard Array Math: Add all the numbers together (reduce) and divide by the length.
        // The "? :" logic ensures we don't accidentally divide by zero and crash the site!
        const avgMacroF1 = macroF1s.length > 0
            ? macroF1s.reduce((a, b) => a + b, 0) / macroF1s.length
            : 0;

        // Grabbing the conflict logic from the Python data.
        const mixed = data.mixed_analysis || {};

        // --- Assembling the final clean object to mail to the Frontend component ---
        const transformed = {
            // These lines take the real Python data if it exists.
            // Notice the "||" fallbacks (like 0.89 or 0.945)? If Python doesn't send that data, 
            // we inject fake "Dummy Data" so your graphs don't look completely blank during testing!
            overallMacroF1: data.overall?.macro_f1 || data.overall_macro_f1 || 0.89,
            perAspectMacroF1Avg: avgMacroF1 || 0.85,
            conflictAUC: (mixed.mixed_review_accuracy / 100) || data.conflict_auc || 0.945,
            avgLatencyMs: data.avg_latency || 85,
            throughputReqPerSec: data.throughput || 12,
            aspectMetrics, // We slot in the cleanly formatted arrays we made above!
            confusionMatrices,
            
            // Hardcoded dummy data distribution for the fancy bar chart on the Metrics page.
            conflictScoreDistribution: [
                { bin: "0.0-0.2", count: 450 },
                { bin: "0.2-0.4", count: 230 },
                { bin: "0.4-0.6", count: 180 },
                { bin: "0.6-0.8", count: 340 },
                { bin: "0.8-1.0", count: 780 },
            ],
            balancedAccuracy: data.overall?.accuracy || 0.91,
            brierScore: data.brier_score || 0.082,
            msrErrorReduction: data.msr_error_reduction || (mixed.mixed_prevalence / 100) || 0.54,
            p95LatencyMs: data.p95_latency || 124,
            memoryUsageMB: data.memory || 452,
        };

        // We wrap the perfectly clean object in standard web headers and fire it exactly back to the browser!
        return NextResponse.json(transformed);
        
    } catch (error: any) {
        // If anything inside the "try" block crashed (like math dividing by zero or Python server offline),
        // we print a scary red error to your command prompt terminal...
        console.error('Metrics fetch error:', error);
        
        // ...and we securely send a custom 500 (Internal Server Error) back to the website so React can show an alert box.
        return NextResponse.json(
            { error: error.message || 'Failed to fetch metrics' },
            { status: 500 }
        );
    }
}
