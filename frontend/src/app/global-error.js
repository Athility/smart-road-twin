'use client';

export default function GlobalError({ error, reset }) {
  return (
    <html>
      <body style={{ backgroundColor: '#020617', color: '#f8fafc', padding: '2rem', fontFamily: 'sans-serif' }}>
        <h2>System Rendering Failure</h2>
        <pre style={{ color: '#ef4444', fontSize: '12px' }}>{error?.stack || error?.message}</pre>
        <button onClick={() => reset()} style={{ padding: '8px 16px', cursor: 'pointer' }}>Reset Application</button>
      </body>
    </html>
  );
}
