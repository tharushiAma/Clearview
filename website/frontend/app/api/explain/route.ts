import http from 'http';
import https from 'https';
import { NextRequest, NextResponse } from 'next/server';
import type { ExplainResponse } from '@/types/api';

export const maxDuration = 900; // 15 minutes for XAI

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
 * @note This endpoint uses the native HTTP module instead of fetch to bypass 
 * Next.js's hardcoded 5-minute timeout, as XAI computations can take 10+ minutes.
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

export async function POST(request: NextRequest) {
  try {
    const body: ExplainRequest = await request.json();

    if (!body.text) {
      return NextResponse.json(
        { error: 'Text field is required' },
        { status: 400 }
      );
    }

    const payload = JSON.stringify({
      text: body.text,
      aspect: body.aspect ?? 'all',
      methods: body.methods ?? ['ig'],
      msr_enabled: body.msr_enabled ?? true,
      msr_strength: body.msr_strength ?? 0.3,
      ckpt_path: body.ckpt_path,
    });

    // We use the native http/https module to bypass the global fetch 5-minute timeout.
    const result = await new Promise<ExplainResponse>((resolve, reject) => {
      let url: URL;
      try {
        url = new URL(`${BACKEND_URL}/explain`);
      } catch (e) {
        return reject(new Error('Invalid BACKEND_URL'));
      }

      const client = url.protocol === 'https:' ? https : http;
      
      const req = client.request(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload)
        },
        timeout: 900000 // 15 minutes
      }, (res) => {
        let data = '';
        res.on('data', (chunk) => {
          data += chunk;
        });

        res.on('end', () => {
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
            try {
              resolve(JSON.parse(data));
            } catch (e) {
              reject(new Error('Failed to parse backend response'));
            }
          } else {
            let errorMsg = `Backend error: ${res.statusCode}`;
            try {
              const parsed = JSON.parse(data);
              if (parsed.detail) errorMsg = parsed.detail;
            } catch (e) {
              // Ignore parse error, use raw data if possible
              if (data) errorMsg += ` - ${data.substring(0, 100)}`;
            }
            reject(new Error(errorMsg));
          }
        });
      });

      req.on('error', (err: any) => {
        if (err.code === 'ECONNREFUSED') {
          reject(new Error('Backend server is not running. Please start the Python backend server first.'));
        } else {
          reject(err);
        }
      });
      
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('TIMEOUT_ERROR'));
      });

      req.write(payload);
      req.end();
    });

    return NextResponse.json(result);

  } catch (error: any) {
    console.error('Explanation error:', error);

    if (error.message === 'TIMEOUT_ERROR') {
      return NextResponse.json(
        { error: 'Request timed out. XAI explanations can take 10+ minutes. Please be patient or try analyzing a single aspect instead of "all".' },
        { status: 504 }
      );
    }
    
    if (error.message && error.message.includes('Backend server is not running')) {
      return NextResponse.json(
        {
          error: error.message,
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
