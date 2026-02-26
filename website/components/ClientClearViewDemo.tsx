'use client';

import dynamic from 'next/dynamic';

// ssr:false must live inside a Client Component (Next.js 16 requirement)
// This prevents Dark Reader browser extension from causing hydration mismatches
const ClearViewDemo = dynamic(() => import('./ClearViewDemo'), { ssr: false });

export default function ClientClearViewDemo() {
  return <ClearViewDemo />;
}
