import { NextRequest, NextResponse } from 'next/server';
import type { PredictResponse } from '@/types/api';

export const maxDuration = 60; // 60 seconds timeout

/**
 * Prediction API Route
 * 
 * Next.js API route that proxies prediction requests to the Python FastAPI backend.
 * Handles ABSA sentiment analysis with optional MSR refinement.
 * 
 * @route POST /api/predict
 * @param {PredictRequest} body - Request containing review text and MSR settings
 * @returns {PredictResponse} Aspect-level sentiments, conflict probability, and timing info
 */

interface PredictRequest {
  text: string;  // Review text to analyze
  msr_enabled?: boolean;  // Whether to apply MSR refinement (default: true)
  msr_strength?: number;  // MSR strength parameter 0.0-1.0 (default: 0.3)
  ckpt_path?: string;  // Optional custom model checkpoint path
}

// FastAPI backend URL - can be configured via environment variable
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

/**
 * POST handler for prediction requests.
 * 
 * Flow:
 * 1. Validate incoming request
 * 2. Forward to Python backend (/predict endpoint)
 * 3. Return results or handle errors
 */
export async function POST(request: NextRequest) {
  try {
    const body: PredictRequest = await request.json();

    if (!body.text) {
      return NextResponse.json(
        { error: 'Text field is required' },
        { status: 400 }
      );
    }

    // Call FastAPI backend instead of spawning Python process
    const response = await fetch(`${BACKEND_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: body.text,
        msr_enabled: body.msr_enabled ?? true,
        msr_strength: body.msr_strength ?? 0.3,
        ckpt_path: body.ckpt_path,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `Backend error: ${response.status}`);
    }

    const result: PredictResponse = await response.json();
    return NextResponse.json(result);

  } catch (error: unknown) {
    console.error('Prediction error:', error);

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

