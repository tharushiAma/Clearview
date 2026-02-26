import { NextRequest, NextResponse } from 'next/server';
import type { ExplainResponse } from '@/types/api';

export const maxDuration = 300; // 5 minutes for XAI

/**
 * XAI Explanation API Route
 * 
 * Next.js API route that proxies XAI explanation requests to the Python FastAPI backend.
 * Generates interpretability outputs using Integrated Gradients, LIME, or SHAP.
 * 
 * @route POST /api/explain
 * @param {ExplainRequest} body - Request containing review text, aspect, and XAI methods
 * @returns {ExplainResponse} Token attributions and MSR delta analysis
 * 
 * @note This endpoint has extended timeout (5 minutes) because XAI computations
 *       can take 2-3 minutes when analyzing all aspects.
 */

interface ExplainRequest {
  text: string;  // Review text to explain
  aspect?: string;  // Aspect to explain or "all" for all aspects
  methods?: string[];  // XAI methods to use: ["ig", "lime", "shap"]
  msr_enabled?: boolean;  // Whether to apply MSR during explanation
  msr_strength?: number;  // MSR strength parameter
  ckpt_path?: string;  // Optional custom model checkpoint
}

// FastAPI backend URL - can be configured via environment variable
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

/**
 * POST handler for explanation requests.
 * 
 * Flow:
 * 1. Validate incoming request
 * 2. Set up extended timeout (5 minutes) for XAI computation
 * 3. Forward to Python backend (/explain endpoint)
 * 4. Handle timeout/connection errors gracefully
 */
export async function POST(request: NextRequest) {
  try {
    const body: ExplainRequest = await request.json();
    
    if (!body.text) {
      return NextResponse.json(
        { error: 'Text field is required' },
        { status: 400 }
      );
    }
    
    // Call FastAPI backend with extended timeout
    // XAI can take 2-3 minutes, so we need custom fetch options
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes
    
    try {
      const response = await fetch(`${BACKEND_URL}/explain`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: body.text,
          aspect: body.aspect ?? 'all',
          methods: body.methods ?? ['ig'],
          msr_enabled: body.msr_enabled ?? true,
          msr_strength: body.msr_strength ?? 0.3,
          ckpt_path: body.ckpt_path,
        }),
        signal: controller.signal,
        // @ts-ignore - Next.js/Node fetch options
        keepalive: true,
      });
      
      clearTimeout(timeoutId);
    
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Backend error: ${response.status}`);
      }
      
      const result: ExplainResponse = await response.json();
      return NextResponse.json(result);
      
    } catch (fetchError) {
      clearTimeout(timeoutId);
      
      // Handle abort/timeout errors
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        return NextResponse.json(
          { error: 'Request timed out after 5 minutes. XAI explanations take 2-3 minutes for all aspects.' },
          { status: 504 }
        );
      }
      
      // Re-throw to be caught by outer catch
      throw fetchError;
    }
    
  } catch (error: unknown) {
    console.error('Explanation error:', error);
    
    // Check if it's a timeout error (undici HeadersTimeoutError)
    if (error instanceof Error && 
        (error.message.includes('HeadersTimeoutError') || error.message.includes('timeout'))) {
      return NextResponse.json(
        { error: 'Request timed out. XAI explanations can take 2-3 minutes. Please be patient or try analyzing a single aspect instead of "all".' },
        { status: 504 }
      );
    }
    
    // Check if it's a connection error
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        { 
          error: 'Backend server is not running. Please start the Python backend server first.',
          hint: 'Run: python backend_server.py'
        },
        { status: 503 }
      );
    }
    
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}
