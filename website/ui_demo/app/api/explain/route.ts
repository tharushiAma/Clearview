import { NextRequest, NextResponse } from 'next/server';
import type { ExplainResponse } from '@/types/api';

export const maxDuration = 300; // 5 minutes for XAI

interface ExplainRequest {
  text: string;
  aspect?: string;
  methods?: string[];
  msr_enabled?: boolean;
  msr_strength?: number;
  ckpt_path?: string;
}

// FastAPI backend URL - can be configured via environment variable
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body: ExplainRequest = await request.json();
    
    if (!body.text) {
      return NextResponse.json(
        { error: 'Text field is required' },
        { status: 400 }
      );
    }
    
    // Call FastAPI backend instead of spawning Python process
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
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `Backend error: ${response.status}`);
    }
    
    const result: ExplainResponse = await response.json();
    return NextResponse.json(result);
    
  } catch (error: unknown) {
    console.error('Explanation error:', error);
    
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

