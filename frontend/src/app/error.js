'use client';

import { useEffect } from 'react';

export default function Error({ error, reset }) {
  useEffect(() => {
    console.error('App Router Error:', error);
  }, [error]);

  return (
    <div className="p-8 bg-slate-900 text-slate-100 min-h-screen flex flex-col items-center justify-center">
      <div className="max-w-md p-6 bg-slate-800 border border-slate-700 rounded-xl space-y-4 text-center">
        <h2 className="text-xl font-bold text-red-400">Application Error</h2>
        <p className="text-xs font-mono text-slate-400 bg-slate-950 p-3 rounded border border-slate-800 break-all text-left">
          {error?.message || 'An unknown rendering error occurred.'}
        </p>
        <button
          onClick={() => reset()}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium transition"
        >
          Try Again
        </button>
      </div>
    </div>
  );
}
