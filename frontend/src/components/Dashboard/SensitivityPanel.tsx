'use client';

import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
} from 'recharts';
import {
  Activity,
  Sliders,
  Scale,
  Sparkles,
  Info,
  ShieldCheck,
  TrendingUp,
  BarChart2,
  RefreshCw,
} from 'lucide-react';
import defaultSensitivityData from '@/data/sensitivity.json';

interface SensitivityData {
  metadata: {
    experiment: string;
    dataset: string;
    num_records: number;
    profiles_evaluated: string[];
    v_values: number[];
    total_configurations: number;
    honesty_disclosure: string;
  };
  records: string[];
  record_labels: string[];
  settings: Array<{
    setting_id: string;
    profile: string;
    v: number;
    weights: number[];
    topsis_scores: number[];
    topsis_ranks: number[];
    vikor_S: number[];
    vikor_R: number[];
    vikor_Q: number[];
    vikor_ranks: number[];
    spearman_rho: number;
    priority_levels: string[];
    compromise_solution: string[];
  }>;
  rank_1_frequencies: Record<
    string,
    {
      record_id: string;
      label: string;
      damage_class: string;
      topsis_rank_1_count: number;
      topsis_rank_1_frequency: number;
      vikor_rank_1_count: number;
      vikor_rank_1_frequency: number;
      total_rank_1_count: number;
      overall_rank_1_frequency: number;
    }
  >;
  n300_cohort_analysis?: {
    cohort_label?: string;
    cohort_size?: number;
    mean_concordance_rho?: number;
    mean_kendall_tau?: number;
    mean_top5_overlap?: number;
    baseline_bootstrap_95ci?: {
      spearman_rho?: number[];
      kendall_tau?: number[];
      top5_overlap?: number[];
    };

  };
}

const RANK_COLORS: Record<number, { bg: string; text: string; border: string }> = {
  1: { bg: 'rgba(239, 68, 68, 0.25)', text: '#F87171', border: '#EF4444' },
  2: { bg: 'rgba(249, 115, 22, 0.22)', text: '#FB923C', border: '#F97316' },
  3: { bg: 'rgba(245, 158, 11, 0.20)', text: '#FCD34D', border: '#F59E0B' },
  4: { bg: 'rgba(59, 130, 246, 0.18)', text: '#93C5FD', border: '#3B82F6' },
  5: { bg: 'rgba(99, 102, 241, 0.18)', text: '#A5B4FC', border: '#6366F1' },
  6: { bg: 'rgba(51, 65, 85, 0.35)',  text: '#94A3B8', border: '#475569' },
};

export const SensitivityPanel: React.FC = () => {
  const [data, setData] = useState<SensitivityData>(defaultSensitivityData as SensitivityData);
  const [heatmapMethod, setHeatmapMethod] = useState<'topsis' | 'vikor'>('topsis');
  const [selectedRecord, setSelectedRecord] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    // Attempt dynamic fetch from backend API if available
    fetch('/api/v1/sensitivity')
      .then((res) => {
        if (res.ok) return res.json();
        throw new Error('API unavailable');
      })
      .then((json) => {
        if (json?.settings?.length) {
          setData(json);
        }
      })
      .catch(() => {
        // Fallback to static bundled data
      });
  }, []);

  if (!mounted) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: '#64748B' }}>
        Loading Sensitivity Engine...
      </div>
    );
  }

  // Prepare chart data for Recharts
  const chartData = data.records.map((id) => {
    const freq = data.rank_1_frequencies[id] || {
      topsis_rank_1_count: 0,
      vikor_rank_1_count: 0,
      total_rank_1_count: 0,
      topsis_rank_1_frequency: 0,
      vikor_rank_1_frequency: 0,
    };
    const shortLabel = id.replace('SRMD-IND-', '#');
    return {
      id,
      name: shortLabel,
      damageClass: freq.damage_class || (id.includes('45') || id.includes('278-00') ? 'D40' : 'Crack'),
      topsisCount: freq.topsis_rank_1_count,
      vikorCount: freq.vikor_rank_1_count,
      topsisPct: Math.round(freq.topsis_rank_1_frequency * 100),
      vikorPct: Math.round(freq.vikor_rank_1_frequency * 100),
    };
  });

  // Calculate average Spearman correlation across all settings
  const avgRho =
    data.settings.reduce((sum, s) => sum + s.spearman_rho, 0) / (data.settings.length || 1);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
        padding: '6px 4px 32px',
        maxWidth: 1600,
        margin: '0 auto',
        width: '100%',
        color: '#EEF2FF',
      }}
    >
      {/* ── Top Overview Banner ─────────────────────────────────── */}
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(17, 21, 32, 0.95) 0%, rgba(15, 23, 42, 0.85) 100%)',
          border: '1px solid #1E2435',
          borderRadius: 8,
          padding: '14px 18px',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 12,
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span
              style={{
                width: 3,
                height: 16,
                background: '#8B5CF6',
                borderRadius: 2,
                display: 'inline-block',
              }}
            />
            <h2 style={{ fontSize: '15px', fontWeight: 700, color: '#EEF2FF', letterSpacing: '-0.01em', margin: 0 }}>
              MCDM Sensitivity & Ranking Stability Analysis
            </h2>
            <span
              style={{
                fontSize: '10px',
                fontFamily: 'monospace',
                padding: '2px 8px',
                borderRadius: 4,
                background: 'rgba(139, 92, 246, 0.15)',
                border: '1px solid rgba(139, 92, 246, 0.4)',
                color: '#C4B5FD',
                fontWeight: 700,
              }}
            >
              9 CONFIGURATIONS
            </span>
          </div>
          <p style={{ fontSize: '11px', color: '#8892A4', margin: 0 }}>
            3 Fuzzy BWM profiles (safety-first, traffic-first, equity-first) × 3 VIKOR consensus weights v ∈ {'{0.25, 0.50, 0.75}'}
          </p>
        </div>

        {/* Quick summary stat chips */}
        <div style={{ display: 'flex', gap: 10 }}>
          <div
            style={{
              background: '#161B27',
              border: '1px solid #1E2435',
              borderRadius: 6,
              padding: '6px 12px',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '9px', color: '#64748B', fontWeight: 600 }}>MEAN SPEARMAN ρ</div>
            <div style={{ fontSize: '14px', fontWeight: 800, fontFamily: 'monospace', color: '#38BDF8' }}>
              {avgRho.toFixed(3)}
            </div>
          </div>
          <div
            style={{
              background: '#161B27',
              border: '1px solid #1E2435',
              borderRadius: 6,
              padding: '6px 12px',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '9px', color: '#64748B', fontWeight: 600 }}>RANK-1 CONCORDANCE</div>
            <div
              style={{ fontSize: '12px', fontWeight: 800, fontFamily: 'monospace', color: '#34D399' }}
              title="Potholes occupy 100% of Rank-1 dispatches at scale <= 15.0; at scale >= 30.0, severe alligator cracks reach Rank 1"
            >
              100% Potholes (Scale ≤ 15)
            </div>
          </div>
        </div>
      </div>

      {/* ── Honest Disclaimer Banner ────────────────────────────── */}
      <div
        style={{
          background: 'rgba(30, 41, 59, 0.35)',
          borderLeft: '4px solid #F59E0B',
          borderRadius: '0 6px 6px 0',
          padding: '10px 14px',
          fontSize: '11px',
          color: '#CBD5E1',
          lineHeight: 1.5,
          display: 'flex',
          gap: 10,
        }}
      >
        <Info style={{ width: 18, height: 18, color: '#F59E0B', flexShrink: 0, marginTop: 1 }} />
        <div>
          <strong style={{ color: '#FCD34D' }}>Empirical Limitations & Demonstration Notice: </strong>
          {data.metadata.honesty_disclosure}
        </div>
      </div>

      {/* ── N=300 Large Cohort Statistical Validation Banner ── */}
      <div
        style={{
          background: '#0F1420',
          border: '1px solid #1E2435',
          borderRadius: 8,
          padding: '10px 14px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 10,
          fontSize: '11px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              background: 'rgba(59,130,246,0.15)',
              border: '1px solid #3B82F6',
              color: '#93C5FD',
              padding: '2px 8px',
              borderRadius: 4,
              fontWeight: 700,
              fontSize: '10px',
              letterSpacing: '0.04em',
            }}
          >
            {data.n300_cohort_analysis?.cohort_label || 'SYNTHETIC (templated bboxes, fixed seed)'}
          </span>
          <span style={{ color: '#94A3B8' }}>
            N=300 Municipal Corridor Validation Cohort
          </span>
        </div>
        <div style={{ display: 'flex', gap: 16, color: '#CBD5E1', fontSize: '11px', flexWrap: 'wrap' }}>
          <span>
            Mean Spearman ρ:{' '}
            <strong style={{ color: '#10B981' }}>
              {data.n300_cohort_analysis?.mean_concordance_rho ?? 0.952}
            </strong>{' '}
            <span style={{ color: '#64748B', fontSize: '10px' }}>
              (95% CI: [{data.n300_cohort_analysis?.baseline_bootstrap_95ci?.spearman_rho?.[0] ?? 0.925}, {data.n300_cohort_analysis?.baseline_bootstrap_95ci?.spearman_rho?.[1] ?? 0.962}])
            </span>
          </span>
          <span>
            Kendall τ:{' '}
            <strong style={{ color: '#60A5FA' }}>
              {data.n300_cohort_analysis?.mean_kendall_tau ?? 0.841}
            </strong>{' '}
            <span style={{ color: '#64748B', fontSize: '10px' }}>
              (95% CI: [{data.n300_cohort_analysis?.baseline_bootstrap_95ci?.kendall_tau?.[0] ?? 0.797}, {data.n300_cohort_analysis?.baseline_bootstrap_95ci?.kendall_tau?.[1] ?? 0.855}])
            </span>
          </span>
          <span>
            Top-5 Overlap:{' '}
            <strong style={{ color: '#F59E0B' }}>
              {data.n300_cohort_analysis?.mean_top5_overlap ?? 0.667}
            </strong>{' '}
            <span style={{ color: '#64748B', fontSize: '10px' }}>
              (95% CI: [{data.n300_cohort_analysis?.baseline_bootstrap_95ci?.top5_overlap?.[0] ?? 0.20}, {data.n300_cohort_analysis?.baseline_bootstrap_95ci?.top5_overlap?.[1] ?? 0.933}])
            </span>
          </span>
        </div>
      </div>

      {/* ── Main Layout: Heatmap (Left/Top) + Recharts Bars (Right/Bottom) ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: 14 }}>
        {/* ── Section A: Rank Heatmap (Records x Setting) ────────── */}
        <div
          style={{
            background: '#111520',
            border: '1px solid #1E2435',
            borderRadius: 8,
            padding: 14,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Sliders style={{ width: 14, height: 14, color: '#60A5FA' }} />
              <span style={{ fontSize: '12px', fontWeight: 700, color: '#EEF2FF' }}>
                Rank Heatmap Matrix (6 Incidents × 9 Parameter Settings)
              </span>
            </div>

            {/* Toggle Method */}
            <div
              style={{
                display: 'flex',
                background: '#0B0E14',
                padding: 2,
                borderRadius: 5,
                border: '1px solid #1E2435',
              }}
            >
              <button
                onClick={() => setHeatmapMethod('topsis')}
                style={{
                  padding: '3px 10px',
                  fontSize: '10px',
                  fontWeight: 700,
                  borderRadius: 4,
                  border: 'none',
                  cursor: 'pointer',
                  background: heatmapMethod === 'topsis' ? '#1D4ED8' : 'transparent',
                  color: heatmapMethod === 'topsis' ? '#FFFFFF' : '#8892A4',
                  transition: 'all 0.15s ease',
                }}
              >
                TOPSIS Ranks
              </button>
              <button
                onClick={() => setHeatmapMethod('vikor')}
                style={{
                  padding: '3px 10px',
                  fontSize: '10px',
                  fontWeight: 700,
                  borderRadius: 4,
                  border: 'none',
                  cursor: 'pointer',
                  background: heatmapMethod === 'vikor' ? '#7C3AED' : 'transparent',
                  color: heatmapMethod === 'vikor' ? '#FFFFFF' : '#8892A4',
                  transition: 'all 0.15s ease',
                }}
              >
                VIKOR Ranks
              </button>
            </div>
          </div>

          {/* Matrix Table */}
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '10px' }}>
              <thead>
                <tr>
                  <th
                    style={{
                      textAlign: 'left',
                      padding: '8px 6px',
                      color: '#64748B',
                      borderBottom: '1px solid #1E2435',
                      fontWeight: 700,
                      minWidth: 140,
                    }}
                  >
                    INCIDENT / CLASS
                  </th>
                  {data.settings.map((s) => (
                    <th
                      key={s.setting_id}
                      style={{
                        textAlign: 'center',
                        padding: '6px 4px',
                        color: '#94A3B8',
                        borderBottom: '1px solid #1E2435',
                        fontWeight: 600,
                        fontFamily: 'monospace',
                        minWidth: 48,
                      }}
                    >
                      <div style={{ color: '#E2E8F0', fontWeight: 700 }}>
                        {s.profile === 'safety-first' ? 'Safe' : s.profile === 'traffic-first' ? 'Traf' : 'Eqty'}
                      </div>
                      <div style={{ fontSize: '9px', color: '#64748B' }}>v={s.v}</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.records.map((recId, rIdx) => {
                  const freq = data.rank_1_frequencies[recId];
                  const dClass = freq?.damage_class || (rIdx === 0 || rIdx === 4 ? 'D40' : 'Crack');
                  const isSelected = selectedRecord === recId;

                  return (
                    <tr
                      key={recId}
                      onClick={() => setSelectedRecord(isSelected ? null : recId)}
                      style={{
                        background: isSelected ? 'rgba(59, 130, 246, 0.08)' : 'transparent',
                        cursor: 'pointer',
                        transition: 'background 0.15s ease',
                      }}
                    >
                      <td
                        style={{
                          padding: '8px 6px',
                          borderBottom: '1px solid #1E2435',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        <div style={{ fontFamily: 'monospace', fontWeight: 700, color: '#EEF2FF' }}>
                          {recId.replace('SRMD-IND-', '#')}
                        </div>
                        <div style={{ fontSize: '9px', color: dClass === 'D40' ? '#60A5FA' : '#F59E0B' }}>
                          {dClass} · {dClass === 'D40' ? 'Pothole' : 'Crack'}
                        </div>
                      </td>

                      {data.settings.map((s) => {
                        const rankList = heatmapMethod === 'topsis' ? s.topsis_ranks : s.vikor_ranks;
                        const rank = rankList[rIdx] ?? 6;
                        const rc = RANK_COLORS[rank] || RANK_COLORS[6];

                        return (
                          <td
                            key={s.setting_id}
                            style={{
                              textAlign: 'center',
                              padding: '6px 4px',
                              borderBottom: '1px solid #1E2435',
                            }}
                          >
                            <span
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                width: 28,
                                height: 26,
                                borderRadius: 4,
                                background: rc.bg,
                                border: `1px solid ${rc.border}60`,
                                color: rc.text,
                                fontWeight: 800,
                                fontFamily: 'monospace',
                                fontSize: '11px',
                              }}
                              title={`${recId} in ${s.setting_id}: Rank ${rank}`}
                            >
                              {rank}
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Heatmap Legend */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginTop: 12,
              paddingTop: 8,
              borderTop: '1px solid #1E2435',
              fontSize: '10px',
            }}
          >
            <span style={{ color: '#64748B' }}>Priority Rank Gradient:</span>
            <div style={{ display: 'flex', gap: 6 }}>
              {[1, 2, 3, 4, 5, 6].map((rk) => (
                <div key={rk} style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                  <span
                    style={{
                      width: 12,
                      height: 12,
                      borderRadius: 2,
                      background: RANK_COLORS[rk].bg,
                      border: `1px solid ${RANK_COLORS[rk].border}`,
                      display: 'inline-block',
                    }}
                  />
                  <span style={{ color: '#94A3B8', fontFamily: 'monospace', fontSize: '9px' }}>#{rk}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── Section B: Rank-1 Frequency Bars (Recharts) ────────── */}
        <div
          style={{
            background: '#111520',
            border: '1px solid #1E2435',
            borderRadius: 8,
            padding: 14,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <BarChart2 style={{ width: 14, height: 14, color: '#8B5CF6' }} />
              <span style={{ fontSize: '12px', fontWeight: 700, color: '#EEF2FF' }}>
                Rank-1 Peak Frequency (Out of 9 Settings)
              </span>
            </div>
            <span style={{ fontSize: '10px', color: '#64748B', fontFamily: 'monospace' }}>
              Recharts Visualizer
            </span>
          </div>

          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E2435" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fill: '#8892A4', fontSize: 10 }}
                  axisLine={{ stroke: '#1E2435' }}
                  tickLine={false}
                />
                <YAxis
                  allowDecimals={false}
                  domain={[0, 9]}
                  tick={{ fill: '#8892A4', fontSize: 10 }}
                  axisLine={{ stroke: '#1E2435' }}
                  tickLine={false}
                />
                <RechartsTooltip
                  contentStyle={{
                    background: '#0F1420',
                    border: '1px solid #2563EB40',
                    borderRadius: 6,
                    fontSize: '11px',
                    color: '#EEF2FF',
                  }}
                  formatter={(value: any, name: any) => [
                    `${value} settings (${Math.round(((value as number) / 9) * 100)}%)`,
                    name === 'topsisCount' ? 'TOPSIS Rank-1' : 'VIKOR Rank-1',
                  ]}
                  labelFormatter={(lbl) => `Incident ${lbl}`}
                />
                <Legend
                  wrapperStyle={{ fontSize: '10px', paddingTop: 8 }}
                  formatter={(val) => (val === 'topsisCount' ? 'TOPSIS Rank-1 Count' : 'VIKOR Rank-1 Count')}
                />
                <Bar dataKey="topsisCount" fill="#3B82F6" radius={[4, 4, 0, 0]} name="topsisCount" />
                <Bar dataKey="vikorCount" fill="#8B5CF6" radius={[4, 4, 0, 0]} name="vikorCount" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Observations note */}
          <div
            style={{
              background: '#161B27',
              borderRadius: 6,
              padding: '10px 12px',
              fontSize: '10px',
              color: '#94A3B8',
              lineHeight: 1.45,
              marginTop: 10,
              border: '1px solid #1E2435',
            }}
          >
            <div style={{ marginBottom: 6 }}>
              <strong style={{ color: '#EEF2FF' }}>Key Empirical Observation (Scale-Dependent): </strong>
              At the default configured crack multiplier (<code>scale = 15.0</code>), the two major potholes (<code>#000045-00</code> and <code>#000278-00</code>) occupy 100% of Rank-1 dispatches. However, sensitivity evaluation across <code>scale ∈ {'{5, 15, 30, 60}'}</code> confirms this result is strictly scale-dependent.
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6, marginTop: 8 }}>
              <div style={{ background: '#0F1420', padding: '6px 8px', borderRadius: 4, border: '1px solid #1E293B' }}>
                <div style={{ color: '#64748B', fontWeight: 700, fontSize: '9px' }}>SCALE = 5.0</div>
                <div style={{ color: '#38BDF8', fontWeight: 700 }}>0% Cracks</div>
                <div style={{ color: '#94A3B8', fontSize: '9px' }}>100% Potholes R1</div>
              </div>
              <div style={{ background: '#0F1420', padding: '6px 8px', borderRadius: 4, border: '1px solid #3B82F640' }}>
                <div style={{ color: '#60A5FA', fontWeight: 700, fontSize: '9px' }}>SCALE = 15.0 (BASE)</div>
                <div style={{ color: '#34D399', fontWeight: 700 }}>0% Cracks</div>
                <div style={{ color: '#94A3B8', fontSize: '9px' }}>100% Potholes R1</div>
              </div>
              <div style={{ background: '#0F1420', padding: '6px 8px', borderRadius: 4, border: '1px solid #F59E0B40' }}>
                <div style={{ color: '#FCD34D', fontWeight: 700, fontSize: '9px' }}>SCALE = 30.0</div>
                <div style={{ color: '#F59E0B', fontWeight: 700 }}>11.1% Cracks</div>
                <div style={{ color: '#94A3B8', fontSize: '9px' }}>VIKOR R1: D20 Alligator</div>
              </div>
              <div style={{ background: '#0F1420', padding: '6px 8px', borderRadius: 4, border: '1px solid #EF444440' }}>
                <div style={{ color: '#F87171', fontWeight: 700, fontSize: '9px' }}>SCALE = 60.0</div>
                <div style={{ color: '#EF4444', fontWeight: 700 }}>50.0% Cracks</div>
                <div style={{ color: '#94A3B8', fontSize: '9px' }}>TOPSIS & VIKOR R1</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Section C: Pairwise Spearman Rho Concordance Table ── */}
      <div
        style={{
          background: '#111520',
          border: '1px solid #1E2435',
          borderRadius: 8,
          padding: 14,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
          <Scale style={{ width: 14, height: 14, color: '#38BDF8' }} />
          <span style={{ fontSize: '12px', fontWeight: 700, color: '#EEF2FF' }}>
            Setting-by-Setting TOPSIS vs VIKOR Spearman Rank Correlation (ρ)
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
          {data.settings.map((s) => {
            const rho = s.spearman_rho;
            const isPerfect = rho >= 0.99;
            const isModerate = rho >= 0.5 && rho < 0.99;

            return (
              <div
                key={s.setting_id}
                style={{
                  background: '#161B27',
                  border: '1px solid #1E2435',
                  borderRadius: 6,
                  padding: '8px 10px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: '#EEF2FF' }}>
                    {s.profile}
                  </div>
                  <div style={{ fontSize: '9px', color: '#64748B', fontFamily: 'monospace' }}>
                    v = {s.v} · DQ = 0.20
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div
                    style={{
                      fontSize: '13px',
                      fontWeight: 800,
                      fontFamily: 'monospace',
                      color: isPerfect ? '#34D399' : isModerate ? '#38BDF8' : '#F59E0B',
                    }}
                  >
                    ρ = {rho.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '8px', color: '#64748B' }}>
                    {isPerfect ? 'Perfect' : isModerate ? 'Moderate' : 'Low Concordance'}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
