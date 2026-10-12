'use client';

import React, { useMemo } from 'react';
import { TelemetryRecord, PriorityLevel } from '@/types/telemetry';
import { AlertTriangle, Clock, ArrowRight } from 'lucide-react';

interface QueueOverviewPanelProps {
  records: TelemetryRecord[];
  onSelectRecord: (id: string) => void;
}

const PRIORITY_CONFIG: Record<
  PriorityLevel,
  { color: string; bg: string; border: string }
> = {
  Critical: {
    color: '#EF4444',
    bg: 'rgba(239,68,68,0.12)',
    border: 'rgba(239,68,68,0.3)',
  },
  High: {
    color: '#F97316',
    bg: 'rgba(249,115,22,0.12)',
    border: 'rgba(249,115,22,0.3)',
  },
  Medium: {
    color: '#F59E0B',
    bg: 'rgba(245,158,11,0.12)',
    border: 'rgba(245,158,11,0.3)',
  },
  Low: {
    color: '#22C55E',
    bg: 'rgba(34,197,94,0.12)',
    border: 'rgba(34,197,94,0.3)',
  },
};

export const QueueOverviewPanel: React.FC<QueueOverviewPanelProps> = ({
  records,
  onSelectRecord,
}) => {
  // Sort descending by topsis_score for Top-3 maintenance dispatch queue
  const top3 = useMemo(() => {
    const sorted = [...records].sort((a, b) => {
      const scoreA = typeof a.topsis_score === 'number' && !isNaN(a.topsis_score) ? a.topsis_score : -1;
      const scoreB = typeof b.topsis_score === 'number' && !isNaN(b.topsis_score) ? b.topsis_score : -1;
      return scoreB - scoreA;
    });
    return sorted.slice(0, 3);
  }, [records]);

  // Priority distribution counts
  const distribution = useMemo(() => {
    const counts: Record<PriorityLevel, number> = {
      Critical: 0,
      High: 0,
      Medium: 0,
      Low: 0,
    };
    records.forEach((r) => {
      const p = r.priority_level;
      if (counts[p] !== undefined) {
        counts[p] += 1;
      }
    });
    return counts;
  }, [records]);

  const tiers: { level: PriorityLevel; count: number; color: string; bg: string; border: string }[] = [
    { level: 'Critical', count: distribution.Critical, ...PRIORITY_CONFIG.Critical },
    { level: 'High', count: distribution.High, ...PRIORITY_CONFIG.High },
    { level: 'Medium', count: distribution.Medium, ...PRIORITY_CONFIG.Medium },
    { level: 'Low', count: distribution.Low, ...PRIORITY_CONFIG.Low },
  ];

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflowY: 'auto',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '8px 14px',
          borderBottom: '1px solid #1E2435',
          background: '#0F1420',
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <span
          style={{
            fontSize: '10px',
            fontWeight: 700,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: '#8892A4',
          }}
        >
          Dispatch Queue & Overview
        </span>
        <span style={{ fontSize: '10px', color: '#4D5A72', fontFamily: 'monospace' }}>
          {records.length} monitored assets
        </span>
      </div>

      <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 14 }}>
        {/* Top-3 Maintenance Queue */}
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: 8,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Clock style={{ width: 12, height: 12, color: '#38BDF8' }} />
              <span
                style={{
                  fontSize: '10px',
                  fontWeight: 700,
                  letterSpacing: '0.07em',
                  textTransform: 'uppercase',
                  color: '#94A3B8',
                }}
              >
                Top-3 Maintenance Queue
              </span>
            </div>
            <span style={{ fontSize: '9px', color: '#64748B' }}>By TOPSIS urgency</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {top3.map((rec, idx) => {
              const pc = PRIORITY_CONFIG[rec.priority_level] ?? PRIORITY_CONFIG.Low;
              const pt = rec.processed_telemetry;
              const isPothole =
                (rec.damage_class || 'D40') === 'D40' && pt?.calculated_volume_liters != null;

              return (
                <div
                  key={rec.id}
                  onClick={() => onSelectRecord(rec.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 10px',
                    background: '#141824',
                    border: '1px solid #1E2435',
                    borderRadius: 6,
                    cursor: 'pointer',
                    transition: 'all 0.12s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = '#3B82F6';
                    e.currentTarget.style.background = '#1A2234';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = '#1E2435';
                    e.currentTarget.style.background = '#141824';
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                    <span
                      style={{
                        fontFamily: 'ui-monospace, monospace',
                        fontWeight: 800,
                        fontSize: '11px',
                        color: idx === 0 ? '#EF4444' : idx === 1 ? '#F97316' : '#F59E0B',
                        width: 20,
                        flexShrink: 0,
                      }}
                    >
                      #{idx + 1}
                    </span>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                        <span
                          style={{
                            fontFamily: 'ui-monospace, monospace',
                            fontWeight: 700,
                            fontSize: '11px',
                            color: '#EEF2FF',
                          }}
                        >
                          {rec.id}
                        </span>
                        <span
                          style={{
                            fontSize: '8px',
                            fontWeight: 700,
                            letterSpacing: '0.06em',
                            textTransform: 'uppercase',
                            padding: '1px 5px',
                            borderRadius: 3,
                            background: pc.bg,
                            color: pc.color,
                            border: `1px solid ${pc.border}`,
                          }}
                        >
                          {rec.priority_level}
                        </span>
                      </div>
                      <div
                        style={{
                          fontSize: '10px',
                          color: '#64748B',
                          marginTop: 2,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {rec.damage_class || 'D40'} · {rec.dist_hospital_km.toFixed(2)} km trauma · {rec.traffic_pcu.toLocaleString()} PCU
                      </div>
                    </div>
                  </div>

                  <div style={{ textAlign: 'right', flexShrink: 0, marginLeft: 8 }}>
                    <div
                      style={{
                        fontFamily: 'ui-monospace, monospace',
                        fontWeight: 700,
                        fontSize: '12px',
                        color: pc.color,
                      }}
                    >
                      {rec.topsis_score != null ? rec.topsis_score.toFixed(3) : '—'}
                    </div>
                    <div
                      style={{
                        fontSize: '9px',
                        color: isPothole ? '#60A5FA' : '#A78BFA',
                        fontWeight: 600,
                      }}
                    >
                      {isPothole && pt?.calculated_volume_liters != null
                        ? `${pt.calculated_volume_liters.toFixed(1)} L`
                        : pt?.crack_severity_index != null
                        ? `Sev ${pt.crack_severity_index.toFixed(2)}`
                        : '—'}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Priority Distribution Breakdown */}
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: 8,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <AlertTriangle style={{ width: 12, height: 12, color: '#F59E0B' }} />
              <span
                style={{
                  fontSize: '10px',
                  fontWeight: 700,
                  letterSpacing: '0.07em',
                  textTransform: 'uppercase',
                  color: '#94A3B8',
                }}
              >
                Priority Distribution
              </span>
            </div>
            <span style={{ fontSize: '9px', color: '#64748B' }}>Quantile calibration</span>
          </div>

          {/* Stacked Bar */}
          <div
            style={{
              display: 'flex',
              height: 7,
              borderRadius: 4,
              overflow: 'hidden',
              background: '#1C2235',
              gap: 1,
              marginBottom: 8,
            }}
          >
            {tiers.map(({ level, count, color }) => {
              const pct = records.length > 0 ? (count / records.length) * 100 : 0;
              if (pct === 0) return null;
              return (
                <div
                  key={level}
                  style={{ width: `${pct}%`, background: color }}
                  title={`${level}: ${count} (${pct.toFixed(0)}%)`}
                />
              );
            })}
          </div>

          {/* Metric cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6 }}>
            {tiers.map(({ level, count, color, bg, border }) => (
              <div
                key={level}
                style={{
                  background: bg,
                  border: `1px solid ${border}`,
                  borderRadius: 5,
                  padding: '6px 4px',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '8px', fontWeight: 700, color, textTransform: 'uppercase' }}>
                  {level}
                </div>
                <div
                  style={{
                    fontSize: '13px',
                    fontWeight: 800,
                    fontFamily: 'ui-monospace, monospace',
                    color: '#EEF2FF',
                    marginTop: 2,
                  }}
                >
                  {count}
                </div>
                <div style={{ fontSize: '9px', color: '#64748B' }}>
                  {records.length > 0 ? `${((count / records.length) * 100).toFixed(0)}%` : '0%'}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Hint banner */}
        <div
          style={{
            padding: '8px 10px',
            background: 'rgba(59,130,246,0.06)',
            border: '1px solid rgba(59,130,246,0.2)',
            borderRadius: 6,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <span style={{ fontSize: '10px', color: '#93C5FD' }}>
            Click an incident in the list or map to inspect full telemetry.
          </span>
          <ArrowRight style={{ width: 12, height: 12, color: '#60A5FA', flexShrink: 0 }} />
        </div>
      </div>
    </div>
  );
};
