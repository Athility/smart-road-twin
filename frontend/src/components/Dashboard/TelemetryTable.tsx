'use client';

import React, { useState } from 'react';
import { TelemetryRecord, PriorityLevel, WeightProfile } from '@/types/telemetry';
import { Search, ChevronUp, ChevronDown, ChevronsUpDown, Waves, Zap } from 'lucide-react';

interface TelemetryTableProps {
  records: TelemetryRecord[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  filteredPriority?: PriorityLevel | 'All';
  weightProfile?: WeightProfile;
  onSelectWeightProfile?: (profile: WeightProfile) => void;
}

type SortField = 'topsis' | 'volume' | 'depth' | 'time';

const PRIORITY_COLOR: Record<string, { color: string; bg: string; border: string }> = {
  Critical: { color: '#EF4444', bg: 'rgba(239,68,68,0.15)', border: 'rgba(239,68,68,0.4)' },
  High:     { color: '#F97316', bg: 'rgba(249,115,22,0.15)', border: 'rgba(249,115,22,0.4)' },
  Medium:   { color: '#F59E0B', bg: 'rgba(245,158,11,0.15)', border: 'rgba(245,158,11,0.4)' },
  Low:      { color: '#22C55E', bg: 'rgba(34,197,94,0.15)',  border: 'rgba(34,197,94,0.4)' },
};

function SortIcon({ field, active, asc }: { field: SortField; active: SortField; asc: boolean }) {
  if (field !== active) return <ChevronsUpDown style={{ width: 11, height: 11, color: '#374163' }} />;
  return asc
    ? <ChevronUp style={{ width: 11, height: 11, color: '#60A5FA' }} />
    : <ChevronDown style={{ width: 11, height: 11, color: '#60A5FA' }} />;
}

export const TelemetryTable: React.FC<TelemetryTableProps> = ({
  records,
  selectedId,
  onSelect,
  filteredPriority = 'All',
  weightProfile,
  onSelectWeightProfile,
}) => {
  const [search, setSearch] = useState('');
  const [sortField, setSortField] = useState<SortField>('topsis');
  const [sortAsc, setSortAsc] = useState(false);

  // Fallback rank calculation if topsis_rank is not directly on the record
  const computedTopsisRanks = React.useMemo(() => {
    const sorted = [...records].sort((a, b) => (b.topsis_score ?? 0) - (a.topsis_score ?? 0));
    const map = new Map<string, number>();
    sorted.forEach((r, i) => map.set(r.id, i + 1));
    return map;
  }, [records]);

  const filtered = React.useMemo(() => {
    return records
      .filter((r) => {
        const matchPriority =
          filteredPriority === 'All' || r.priority_level === filteredPriority;
        const q = search.toLowerCase();
        const matchSearch =
          !q ||
          r.id.toLowerCase().includes(q) ||
          r.vehicle_id.toLowerCase().includes(q) ||
          r.priority_level.toLowerCase().includes(q);
        return matchPriority && matchSearch;
      })
      .sort((a, b) => {
        let va = 0, vb = 0;
        if (sortField === 'topsis') {
          va = typeof a.topsis_score === 'number' && !isNaN(a.topsis_score) ? a.topsis_score : 0;
          vb = typeof b.topsis_score === 'number' && !isNaN(b.topsis_score) ? b.topsis_score : 0;
        } else if (sortField === 'volume') {
          va =
            a.processed_telemetry?.defect_severity_magnitude ??
            a.processed_telemetry?.calculated_volume_liters ??
            (a.processed_telemetry?.crack_severity_index != null
              ? a.processed_telemetry.crack_severity_index * 15
              : 0);
          vb =
            b.processed_telemetry?.defect_severity_magnitude ??
            b.processed_telemetry?.calculated_volume_liters ??
            (b.processed_telemetry?.crack_severity_index != null
              ? b.processed_telemetry.crack_severity_index * 15
              : 0);
        } else if (sortField === 'depth') {
          va = a.processed_telemetry?.estimated_depth_cm ?? 0;
          vb = b.processed_telemetry?.estimated_depth_cm ?? 0;
        } else {
          va = new Date(a.timestamp).getTime() || 0;
          vb = new Date(b.timestamp).getTime() || 0;
        }
        return sortAsc ? va - vb : vb - va;
      });
  }, [records, filteredPriority, search, sortField, sortAsc]);

  const toggleSort = (field: SortField) => {
    if (sortField === field) setSortAsc(!sortAsc);
    else { setSortField(field); setSortAsc(false); }
  };

  const thStyle = (field: SortField): React.CSSProperties => ({
    padding: '8px 12px',
    fontSize: '10px',
    fontWeight: 600,
    letterSpacing: '0.07em',
    textTransform: 'uppercase',
    color: sortField === field ? '#60A5FA' : '#4D5A72',
    cursor: 'pointer',
    userSelect: 'none',
    whiteSpace: 'nowrap',
    borderBottom: '1px solid #1E2435',
    background: '#111520',
    position: 'sticky' as const,
    top: 0,
    zIndex: 1,
  });

  const thPlainStyle: React.CSSProperties = {
    padding: '8px 12px',
    fontSize: '10px',
    fontWeight: 600,
    letterSpacing: '0.07em',
    textTransform: 'uppercase',
    color: '#4D5A72',
    whiteSpace: 'nowrap',
    borderBottom: '1px solid #1E2435',
    background: '#111520',
    position: 'sticky',
    top: 0,
    zIndex: 1,
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        background: '#111520',
        border: '1px solid #1E2435',
        borderRadius: 8,
        overflow: 'hidden',
        minHeight: 0,
        flex: 1,
      }}
    >
      {/* ── Toolbar ──────────────────────────────────────── */}
      <div
        className="flex items-center justify-between gap-3"
        style={{
          padding: '10px 12px',
          borderBottom: '1px solid #1E2435',
          background: '#0F1420',
          flexShrink: 0,
        }}
      >
        <div className="flex items-center gap-2">
          <span
            style={{
              fontSize: '11px',
              fontWeight: 700,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              color: '#8892A4',
            }}
          >
            Incidents
          </span>
          <span
            style={{
              fontSize: '10px',
              fontFamily: 'ui-monospace, SFMono-Regular, monospace',
              color: '#4D5A72',
              background: '#1C2235',
              border: '1px solid #252D40',
              padding: '1px 6px',
              borderRadius: 3,
            }}
          >
            {filtered.length} / {records.length}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {onSelectWeightProfile && (
            <select
              value={weightProfile || 'safety-first'}
              onChange={(e) => onSelectWeightProfile(e.target.value as WeightProfile)}
              style={{
                padding: '4px 8px',
                fontSize: '11px',
                fontWeight: 600,
                background: '#161B27',
                border: '1px solid #1E2435',
                borderRadius: 4,
                color: '#93C5FD',
                outline: 'none',
                cursor: 'pointer',
              }}
              title="MCDM Weight Profile: Re-ranks live via /api/v1/prioritization"
              aria-label="Prioritization weight profile"
            >
              <option value="safety-first">Profile: Safety-First</option>
              <option value="traffic-first">Profile: Traffic-First</option>
              <option value="equity-first">Profile: Equity-First</option>
            </select>
          )}

          <div style={{ position: 'relative', flexShrink: 0 }}>
            <Search
              style={{
                width: 12,
                height: 12,
                position: 'absolute',
                left: 8,
                top: '50%',
                transform: 'translateY(-50%)',
                color: '#4D5A72',
                pointerEvents: 'none',
              }}
            />
            <input
              id="incident-search"
              name="incident-search"
              type="text"
              placeholder="Search…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{
                paddingLeft: 26,
                paddingRight: 8,
                paddingTop: 5,
                paddingBottom: 5,
                fontSize: '11px',
                background: '#161B27',
                border: '1px solid #1E2435',
                borderRadius: 4,
                color: '#EEF2FF',
                outline: 'none',
                width: 120,
              }}
              onFocus={(e) => (e.target.style.borderColor = '#2563EB')}
              onBlur={(e) => (e.target.style.borderColor = '#1E2435')}
            />
          </div>
        </div>
      </div>

      {/* ── Table ────────────────────────────────────────── */}
      <div style={{ overflowY: 'auto', overflowX: 'hidden', flex: 1 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
          <colgroup>
            <col style={{ width: '28%' }} />
            <col style={{ width: '26%' }} />
            <col style={{ width: '30%' }} />
            <col style={{ width: '16%' }} />
          </colgroup>
          <thead>
            <tr>
              <th style={thPlainStyle}>Incident</th>
              <th
                style={thStyle('topsis')}
                onClick={() => toggleSort('topsis')}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleSort('topsis'); } }}
                tabIndex={0}
                role="button"
                aria-label="Sort by TOPSIS score"
              >
                <span className="flex items-center gap-1">
                  Priority <SortIcon field="topsis" active={sortField} asc={sortAsc} />
                </span>
              </th>
              <th
                style={{ ...thStyle('volume'), textAlign: 'right' }}
                onClick={() => toggleSort('volume')}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleSort('volume'); } }}
                tabIndex={0}
                role="button"
                aria-label="Sort by volume or crack severity"
                title="Volume (L) for potholes / Crack severity for surface cracks"
              >
                <span className="flex items-center justify-end gap-1">
                  Volume (L) / Severity <SortIcon field="volume" active={sortField} asc={sortAsc} />
                </span>
              </th>
              <th
                style={thStyle('depth')}
                onClick={() => toggleSort('depth')}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleSort('depth'); } }}
                tabIndex={0}
                role="button"
                aria-label="Sort by depth"
              >
                <span className="flex items-center gap-1">
                  Depth <SortIcon field="depth" active={sortField} asc={sortAsc} />
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td
                  colSpan={4}
                  style={{
                    padding: '32px 16px',
                    textAlign: 'center',
                    color: '#4D5A72',
                    fontSize: '12px',
                  }}
                >
                  {search
                    ? `No incidents match "${search}"`
                    : 'No incidents match the current filter'}
                </td>
              </tr>
            ) : (
              filtered.map((record) => {
                const pc = PRIORITY_COLOR[record.priority_level] ?? PRIORITY_COLOR.Low;
                const isSelected = record.id === selectedId;
                const pt = record.processed_telemetry;
                const isSonar = pt?.detection_modality === 'sonar_submerged_acoustic';
                const hasImpact = pt?.dynamic_impact_confirmed;
                const isPothole = (record.damage_class || 'D40') === 'D40' && pt?.calculated_volume_liters != null;

                return (
                  <tr
                    key={record.id}
                    onClick={() => onSelect(record.id)}
                    style={{
                      cursor: 'pointer',
                      background: isSelected ? 'rgba(59,130,246,0.06)' : 'transparent',
                      boxShadow: isSelected ? 'inset 3px 0 0 #3B82F6' : 'none',
                      borderBottom: '1px solid #1A2030',
                      transition: 'background 0.12s',
                    }}
                    onMouseOver={(e) => {
                      if (!isSelected) (e.currentTarget as HTMLTableRowElement).style.background = '#161B27';
                    }}
                    onMouseOut={(e) => {
                      if (!isSelected) (e.currentTarget as HTMLTableRowElement).style.background = 'transparent';
                    }}
                  >
                    {/* ID + vehicle */}
                    <td style={{ padding: '8px 12px', verticalAlign: 'middle' }}>
                      <div
                        style={{
                          fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                          fontWeight: 700,
                          fontSize: '12px',
                          color: '#EEF2FF',
                          marginBottom: 2,
                        }}
                      >
                        {record.id}
                      </div>
                      <div className="flex items-center gap-1.5" style={{ flexWrap: 'wrap' }}>
                        {isSonar && (
                          <span style={{ color: '#38BDF8', lineHeight: 1 }} title="Acoustic Sonar">
                            <Waves style={{ width: 9, height: 9 }} />
                          </span>
                        )}
                        {hasImpact && (
                          <span style={{ color: '#EF4444', lineHeight: 1 }} title="Dynamic Impact >1.5g">
                            <Zap style={{ width: 9, height: 9 }} />
                          </span>
                        )}
                        <span
                          style={{
                            fontSize: '10px',
                            color: '#4D5A72',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            maxWidth: 100,
                          }}
                        >
                          {record.vehicle_id}
                        </span>
                      </div>
                    </td>

                    {/* Priority badge + TOPSIS & VIKOR ranks */}
                    <td style={{ padding: '8px 10px', verticalAlign: 'middle' }}>
                      {(() => {
                        const tRank = record.topsis_rank ?? computedTopsisRanks.get(record.id);
                        const vRank = record.vikor_rank;
                        const methodsDisagree =
                          tRank !== undefined && vRank !== undefined && Math.abs(tRank - vRank) >= 2;

                        return (
                          <>
                            <div
                              style={{
                                marginBottom: 3,
                                display: 'flex',
                                alignItems: 'center',
                                gap: 4,
                                flexWrap: 'wrap',
                              }}
                            >
                              <span
                                style={{
                                  display: 'inline-block',
                                  fontSize: '9px',
                                  fontWeight: 700,
                                  letterSpacing: '0.08em',
                                  textTransform: 'uppercase',
                                  background: pc.bg,
                                  border: `1px solid ${pc.border}`,
                                  color: pc.color,
                                  padding: '2px 5px',
                                  borderRadius: 3,
                                }}
                              >
                                {record.priority_level}
                              </span>
                              {methodsDisagree && (
                                <span
                                  style={{
                                    fontSize: '8px',
                                    fontWeight: 700,
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.04em',
                                    background: 'rgba(239,68,68,0.2)',
                                    color: '#F87171',
                                    border: '1px solid rgba(239,68,68,0.45)',
                                    padding: '1px 4px',
                                    borderRadius: 2,
                                    whiteSpace: 'nowrap',
                                  }}
                                  title={`Methods disagree: TOPSIS rank #${tRank} vs VIKOR rank #${vRank} (difference >= 2)`}
                                >
                                  ≠ Disagree (Δ{Math.abs(tRank - (vRank as number))})
                                </span>
                              )}
                            </div>
                            <div
                              className="flex items-center gap-1.5"
                              style={{ fontSize: '10px', fontFamily: 'ui-monospace, monospace' }}
                            >
                              <span
                                title={`TOPSIS score: ${
                                  typeof record.topsis_score === 'number' && !isNaN(record.topsis_score)
                                    ? record.topsis_score.toFixed(3)
                                    : 'N/A'
                                }`}
                              >
                                <strong style={{ color: pc.color }}>T#{tRank ?? '—'}</strong>{' '}
                                <span style={{ color: '#64748B' }}>
                                  ({typeof record.topsis_score === 'number' && !isNaN(record.topsis_score)
                                    ? record.topsis_score.toFixed(3)
                                    : 'N/A'})
                                </span>
                              </span>
                              {vRank !== undefined && (
                                <>
                                  <span style={{ color: '#334155' }}>·</span>
                                  <span style={{ color: '#A78BFA' }} title={`VIKOR rank #${vRank}`}>
                                    <strong>V#{vRank}</strong>
                                  </span>
                                </>
                              )}
                            </div>
                          </>
                        );
                      })()}
                    </td>

                    {/* Volume (L) for potholes, Crack severity for others */}
                    <td
                      style={{
                        padding: '8px 12px',
                        verticalAlign: 'middle',
                        fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                        fontSize: '12px',
                        color: isPothole ? '#60A5FA' : '#A78BFA',
                        fontWeight: 600,
                        textAlign: 'right',
                      }}
                    >
                      {isPothole && pt?.calculated_volume_liters != null ? (
                        <div>
                          <div className="flex items-center justify-end gap-1">
                            <span>{pt.calculated_volume_liters.toFixed(1)} L</span>
                            {record.area_clamped && (
                              <span
                                style={{
                                  fontSize: '8px',
                                  fontWeight: 700,
                                  textTransform: 'uppercase',
                                  letterSpacing: '0.04em',
                                  background: 'rgba(234,179,8,0.18)',
                                  color: '#FBBF24',
                                  border: '1px solid rgba(234,179,8,0.4)',
                                  padding: '1px 3px',
                                  borderRadius: 2,
                                }}
                                title={
                                  record.area_unclamped_m2 != null
                                    ? `Area clamped from ${record.area_unclamped_m2.toFixed(3)} m² to physical bounds`
                                    : 'Area clamped to physical bounds'
                                }
                              >
                                clamped
                              </span>
                            )}
                          </div>
                          <div style={{ fontSize: '9px', color: '#4D5A72', fontWeight: 500 }}>Volume (L)</div>
                        </div>
                      ) : pt?.crack_severity_index != null ? (
                        <div>
                          <div className="flex items-center justify-end gap-1">
                            <span>{pt.crack_severity_index.toFixed(2)}</span>
                            {record.area_clamped && (
                              <span
                                style={{
                                  fontSize: '8px',
                                  fontWeight: 700,
                                  textTransform: 'uppercase',
                                  letterSpacing: '0.04em',
                                  background: 'rgba(234,179,8,0.18)',
                                  color: '#FBBF24',
                                  border: '1px solid rgba(234,179,8,0.4)',
                                  padding: '1px 3px',
                                  borderRadius: 2,
                                }}
                                title={
                                  record.area_unclamped_m2 != null
                                    ? `Area clamped from ${record.area_unclamped_m2.toFixed(3)} m² to physical bounds`
                                    : 'Area clamped to physical bounds'
                                }
                              >
                                clamped
                              </span>
                            )}
                          </div>
                          <div style={{ fontSize: '9px', color: '#4D5A72', fontWeight: 500 }}>Crack severity</div>
                        </div>
                      ) : (
                        '—'
                      )}
                    </td>

                    {/* Depth */}
                    <td
                      style={{
                        padding: '8px 12px',
                        verticalAlign: 'middle',
                        fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                        fontSize: '12px',
                        color: '#8892A4',
                        textAlign: 'right',
                      }}
                    >
                      {pt?.estimated_depth_cm != null
                        ? `${pt.estimated_depth_cm.toFixed(1)} cm`
                        : '—'}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
