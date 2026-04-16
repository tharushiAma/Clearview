import { NextRequest, NextResponse } from 'next/server';

export const maxDuration = 600; // 10 minutes for large batches

/**
 * Bulk Prediction API Route
 *
 * Proxies bulk review prediction requests to the Python FastAPI backend.
 * Accepts a list of review texts and returns aggregated sentiment dashboard data.
 *
 * @route POST /api/predict-bulk
 * @param {{ reviews: string[], msr_enabled?: boolean }} body
 * @returns Aggregated aspect-level sentiment summary + per-review rows
 */

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    if (!body.reviews || !Array.isArray(body.reviews) || body.reviews.length === 0) {
      return NextResponse.json(
        { error: 'reviews array is required and must not be empty' },
        { status: 400 }
      );
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 290000);

    try {
      const response = await fetch(`${BACKEND_URL}/predict-bulk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reviews: body.reviews,
          msr_enabled: body.msr_enabled ?? true,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Backend error: ${response.status}`);
      }

      const result = await response.json();
      return NextResponse.json(result);

    } catch (fetchError) {
      clearTimeout(timeoutId);
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        return NextResponse.json(
          { error: 'Bulk prediction timed out. Try uploading a smaller CSV file.' },
          { status: 504 }
        );
      }
      throw fetchError;
    }

  } catch (error: unknown) {
    console.error('Bulk prediction error:', error);

    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        {
          error: 'Backend server is not running. Please start the Python backend server first.',
          hint: 'Run: python backend_server.py',
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
