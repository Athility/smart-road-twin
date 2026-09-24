'use client';

import React from 'react';
import { RefreshCw, Radio, Map, Database } from 'lucide-react';

interface HeaderProps {
  isLive: boolean;
  isRefreshing: boolean;
  lastUpdated: string | null;
  onRefresh: () => void;
  autoRefresh: boolean;
  onToggleAutoRefresh: () => void;
  totalDefects: number;
  activeView?: 'twin' | 'explorer';
  onViewChange?: (view: 'twin' | 'explorer') => void;
}

function formatRelativeTime(iso: string | null): string {
  if (!iso) return '';
  const diff = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 5) return 'just now';
  if (diff < 60) return `${diff}s ago`;
  return `${Math.round(diff / 60)}m ago`;
}

export const Header: React.FC<HeaderProps> = ({
  isLive,
  isRefreshing,
  lastUpdated,
  onRefresh,
  autoRefresh,
  onToggleAutoRefresh,
  activeView = 'twin',
  onViewChange,
}) => {
  const [mounted, setMounted] = React.useState(false);
  const [tick, setTick] = React.useState(0);

  React.useEffect(() => {
    setMounted(true);
    const id = setInterval(() => setTick((t) => t + 1), 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <header
      className="sticky top-0 z-40 border-b"
      style={{
        background: 'rgba(11, 14, 20, 0.96)',
        backdropFilter: 'blur(12px)',
        borderColor: '#1E2435',
      }}
    >
      <div
        className="mx-auto flex items-center justify-between gap-4 px-5"
        style={{ maxWidth: '1600px', height: '52px' }}
      >
        {/* ── Brand ────────────────────────────────────────── */}
        <div className="flex items-center gap-3 min-w-0">
          {/* Logo mark */}
          <div
            className="flex-shrink-0 flex items-center justify-center rounded"
            style={{
              width: 32,
              height: 32,
              background: 'linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%)',
              border: '1px solid #2563EB40',
            }}
          >
            {/* Simple road/network icon as SVG */}
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <rect x="7" y="1" width="4" height="16" rx="1" fill="#93C5FD" opacity="0.9"/>
              <rect x="1" y="7" width="16" height="4" rx="1" fill="#93C5FD" opacity="0.9"/>
              <circle cx="9" cy="9" r="2" fill="#DBEAFE"/>
            </svg>
          </div>

          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1
                className="font-semibold tracking-tight truncate"
                style={{ fontSize: '15px', color: '#EEF2FF' }}
              >
                Smart Road Digital Twin
              </h1>
              <span
                className="hidden sm:inline-flex items-center gap-1 flex-shrink-0"
                style={{
                  fontSize: '10px',
                  fontWeight: 600,
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  color: '#8892A4',
                }}
              >
                <span
                  style={{
                    display: 'inline-block',
                    width: 4,
                    height: 4,
                    borderRadius: 2,
                    background: '#374163',
                  }}
                />
                Mumbai Western Corridor
              </span>
            </div>
            <p style={{ fontSize: '10px', color: '#4D5A72', marginTop: 1 }}>
              IoT telemetry · AI depth estimation · TOPSIS prioritization
            </p>
          </div>
        </div>

        {/* ── View Switcher Tabs ──────────────────────────────── */}
        {onViewChange && (
          <div
            className="flex items-center p-0.5 rounded-lg"
            style={{
              background: '#111520',
              border: '1px solid #1E2435',
            }}
          >
            <button
              onClick={() => onViewChange('twin')}
              className="flex items-center gap-1.5 px-3 py-1 rounded text-xs font-semibold transition-all"
              style={{
                background: activeView === 'twin' ? '#1D4ED8' : 'transparent',
                color: activeView === 'twin' ? '#EFF6FF' : '#8892A4',
                border: activeView === 'twin' ? '1px solid #3B82F6' : '1px solid transparent',
                cursor: 'pointer',
              }}
            >
              <Map style={{ width: 12, height: 12 }} />
              <span>Spatial Twin Map</span>
            </button>
            <button
              onClick={() => onViewChange('explorer')}
              className="flex items-center gap-1.5 px-3 py-1 rounded text-xs font-semibold transition-all"
              style={{
                background: activeView === 'explorer' ? '#1D4ED8' : 'transparent',
                color: activeView === 'explorer' ? '#EFF6FF' : '#8892A4',
                border: activeView === 'explorer' ? '1px solid #3B82F6' : '1px solid transparent',
                cursor: 'pointer',
              }}
            >
              <Database style={{ width: 12, height: 12 }} />
              <span>Dataset Explorer</span>
            </button>
          </div>
        )}

        {/* ── Controls ─────────────────────────────────────── */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {/* Live / Standby status */}
          <div
            className="flex items-center gap-1.5 px-2.5 py-1 rounded"
            style={{
              background: isLive ? 'rgba(16, 185, 129, 0.08)' : 'rgba(245, 158, 11, 0.08)',
              border: `1px solid ${isLive ? 'rgba(16, 185, 129, 0.25)' : 'rgba(245, 158, 11, 0.25)'}`,
              fontSize: '11px',
              fontWeight: 500,
              color: isLive ? '#10B981' : '#F59E0B',
            }}
          >
            <span className="relative flex h-1.5 w-1.5">
              <span
                className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                style={{ background: isLive ? '#10B981' : '#F59E0B' }}
              />
              <span
                className="relative inline-flex rounded-full h-1.5 w-1.5"
                style={{ background: isLive ? '#10B981' : '#F59E0B' }}
              />
            </span>
            <span>{isLive ? 'Live' : 'Simulation'}</span>
          </div>

          {/* Last updated — client only */}
          {mounted && lastUpdated ? (
            <span
              className="hidden lg:block"
              style={{ fontSize: '11px', color: '#4D5A72' }}
            >
              {formatRelativeTime(lastUpdated)}
            </span>
          ) : null}

          {/* Auto-refresh toggle */}
          <button
            onClick={onToggleAutoRefresh}
            title={autoRefresh ? 'Pause auto-refresh' : 'Enable auto-refresh (3s)'}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded transition-colors"
            style={{
              background: autoRefresh ? 'rgba(59, 130, 246, 0.1)' : '#111520',
              border: `1px solid ${autoRefresh ? 'rgba(59, 130, 246, 0.3)' : '#1E2435'}`,
              fontSize: '11px',
              fontWeight: 500,
              color: autoRefresh ? '#60A5FA' : '#4D5A72',
              cursor: 'pointer',
            }}
          >
            <Radio
              style={{
                width: 12,
                height: 12,
                animation: autoRefresh ? 'pulse 2s cubic-bezier(0.4,0,0.6,1) infinite' : 'none',
              }}
            />
            <span className="hidden sm:inline">
              {autoRefresh ? 'Auto' : 'Paused'}
            </span>
          </button>

          {/* Manual refresh */}
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            title="Refresh now"
            className="flex items-center gap-1.5 px-3 py-1 rounded transition-colors disabled:opacity-50"
            style={{
              background: '#1D4ED8',
              border: '1px solid #2563EB',
              fontSize: '11px',
              fontWeight: 600,
              color: '#fff',
              cursor: isRefreshing ? 'not-allowed' : 'pointer',
            }}
          >
            <RefreshCw
              style={{
                width: 12,
                height: 12,
                animation: isRefreshing ? 'spin 1s linear infinite' : 'none',
              }}
            />
            <span className="hidden sm:inline">Sync</span>
          </button>
        </div>
      </div>
    </header>
  );
};
