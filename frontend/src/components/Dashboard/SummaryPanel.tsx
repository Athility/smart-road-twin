'use client';

import React from 'react';
import { TelemetrySummary, PriorityLevel } from '@/types/telemetry';
import { CloudRain, Sun, Eye, Droplets } from 'lucide-react';

interface SummaryPanelProps {
  summary: TelemetrySummary;
  selectedPriority: PriorityLevel | 'All';
  onSelectPriority: (priority: PriorityLevel | 'All') => void;
}

// Minimal KPI cell
function KpiCell({
  label,
  value,
  unit,
  valueColor,
  accent = false,
}: {
  label: string;
  value: string | number;
  unit?: string;
  valueColor?: string;
  accent?: boolean;
}) {
  return (
    <div
      className="flex flex-col justify-between"
      style={{ minWidth: 0, padding: '10px 16px' }}
    >
      <span
        style={{
          fontSize: '10px',
          fontWeight: 600,
          letterSpacing: '0.07em',
          textTransform: 'uppercase',
          color: '#4D5A72',
          display: 'block',
          marginBottom: 4,
          whiteSpace: 'nowrap',
        }}
      >
        {label}
      </span>
      <div className="flex items-baseline gap-1.5">
        <span
          style={{
            fontSize: '22px',
            fontWeight: 700,
            fontFamily: 'ui-monospace, SFMono-Regular, monospace',
            color: valueColor ?? '#EEF2FF',
            lineHeight: 1,
            letterSpacing: '-0.02em',
          }}
        >
          {value}
        </span>
        {unit && (
          <span
            style={{
              fontSize: '11px',
              fontWeight: 500,
              color: '#8892A4',
            }}
          >
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}

// Priority filter button
function PriorityButton({
  level,
  count,
  color,
  bgColor,
  borderColor,
  isActive,
  onClick,
}: {
  level: string;
  count: number;
  color: string;
  bgColor: string;
  borderColor: string;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 px-3 py-1.5 rounded transition-all focus:outline-none"
      style={{
        background: isActive ? bgColor : 'transparent',
        border: `1px solid ${isActive ? color : borderColor}`,
        cursor: 'pointer',
        opacity: count === 0 ? 0.4 : 1,
      }}
      title={`Filter: ${level} priority`}
      aria-pressed={isActive}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: '50%',
          background: color,
          flexShrink: 0,
          boxShadow: isActive ? `0 0 6px ${color}` : 'none',
        }}
      />
      <span
        style={{
          fontSize: '11px',
          fontWeight: 600,
          color: isActive ? color : '#8892A4',
          letterSpacing: '0.04em',
        }}
      >
        {level}
      </span>
      <span
        style={{
          fontSize: '12px',
          fontWeight: 700,
          fontFamily: 'ui-monospace, SFMono-Regular, monospace',
          color: isActive ? color : '#EEF2FF',
          minWidth: 14,
          textAlign: 'right',
        }}
      >
        {count}
      </span>
    </button>
  );
}

const DIVIDER = (
  <div
    style={{
      width: 1,
      background: '#1E2435',
      alignSelf: 'stretch',
      margin: '8px 0',
      flexShrink: 0,
    }}
  />
);

export const SummaryPanel: React.FC<SummaryPanelProps> = ({
  summary,
  selectedPriority,
  onSelectPriority,
}) => {
  const {
    total_defects,
    total_volume_liters,
    average_volume_liters,
    weather_state,
    is_raining,
    submerged_defects,
    average_luminance,
    priority_counts,
  } = summary;

  // Fix: weather description only uses active sensor logic
  const usesSonar = is_raining || (average_luminance < 40);
  const weatherLabel = is_raining ? 'Rain / Wet' : 'Clear / Dry';

  const critical = priority_counts.Critical ?? 0;
  const high = priority_counts.High ?? 0;
  const medium = priority_counts.Medium ?? 0;
  const low = priority_counts.Low ?? 0;

  const urgentCount = critical + high;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* ── KPI Strip ──────────────────────────────────────── */}
      <div
        className="flex items-stretch overflow-x-auto"
        style={{
          background: '#111520',
          border: '1px solid #1E2435',
          borderRadius: 8,
        }}
      >
        {/* Total defects */}
        <KpiCell
          label="Active Defects"
          value={total_defects}
          valueColor="#EEF2FF"
        />

        {DIVIDER}

        {/* Urgent */}
        <KpiCell
          label="Critical + High"
          value={urgentCount}
          valueColor={urgentCount > 0 ? '#EF4444' : '#22C55E'}
        />

        {DIVIDER}

        {/* Volume */}
        <KpiCell
          label="Total Volume"
          value={total_volume_liters.toFixed(1)}
          unit="L"
          valueColor="#60A5FA"
        />

        {DIVIDER}

        {/* Mean volume */}
        <KpiCell
          label="Mean / Defect"
          value={average_volume_liters.toFixed(1)}
          unit="L"
        />

        {DIVIDER}

        {/* Submerged */}
        <KpiCell
          label="Submerged"
          value={submerged_defects}
          unit="defects"
          valueColor={submerged_defects > 0 ? '#38BDF8' : '#4D5A72'}
        />

        {DIVIDER}

        {/* Weather / Road State */}
        <div
          className="flex items-center gap-2"
          style={{ padding: '10px 16px', minWidth: 0 }}
        >
          <div>
            <span
              style={{
                fontSize: '10px',
                fontWeight: 600,
                letterSpacing: '0.07em',
                textTransform: 'uppercase',
                color: '#4D5A72',
                display: 'block',
                marginBottom: 4,
                whiteSpace: 'nowrap',
              }}
            >
              Road Condition
            </span>
            <div className="flex items-center gap-2">
              {is_raining ? (
                <CloudRain style={{ width: 14, height: 14, color: '#38BDF8' }} />
              ) : (
                <Sun style={{ width: 14, height: 14, color: '#F59E0B' }} />
              )}
              <span
                style={{
                  fontSize: '13px',
                  fontWeight: 600,
                  color: is_raining ? '#38BDF8' : '#F59E0B',
                  whiteSpace: 'nowrap',
                }}
              >
                {weatherLabel}
              </span>
            </div>
          </div>
        </div>

        {DIVIDER}

        {/* Luminance + Modality */}
        <div
          className="flex items-center gap-2"
          style={{ padding: '10px 16px', minWidth: 0 }}
        >
          <div>
            <span
              style={{
                fontSize: '10px',
                fontWeight: 600,
                letterSpacing: '0.07em',
                textTransform: 'uppercase',
                color: '#4D5A72',
                display: 'block',
                marginBottom: 4,
                whiteSpace: 'nowrap',
              }}
            >
              Sensor Modality
            </span>
            <div className="flex items-center gap-2">
              {usesSonar ? (
                <Droplets style={{ width: 13, height: 13, color: '#38BDF8' }} />
              ) : (
                <Eye style={{ width: 13, height: 13, color: '#A78BFA' }} />
              )}
              <span
                style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  color: usesSonar ? '#38BDF8' : '#A78BFA',
                  whiteSpace: 'nowrap',
                }}
              >
                {usesSonar ? 'Acoustic Sonar' : 'Optical LiDAR'}
              </span>
              <span
                style={{
                  fontSize: '10px',
                  color: '#4D5A72',
                  fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                  whiteSpace: 'nowrap',
                }}
              >
                {average_luminance} lux
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Priority Filter Row ────────────────────────────── */}
      <div
        className="flex items-center gap-2 flex-wrap"
        style={{ paddingLeft: 2 }}
      >
        <span
          style={{
            fontSize: '10px',
            fontWeight: 600,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: '#4D5A72',
            marginRight: 4,
            flexShrink: 0,
          }}
        >
          Filter
        </span>

        <PriorityButton
          level="Critical"
          count={critical}
          color="#EF4444"
          bgColor="rgba(239,68,68,0.1)"
          borderColor="rgba(239,68,68,0.2)"
          isActive={selectedPriority === 'Critical'}
          onClick={() => onSelectPriority(selectedPriority === 'Critical' ? 'All' : 'Critical')}
        />
        <PriorityButton
          level="High"
          count={high}
          color="#F97316"
          bgColor="rgba(249,115,22,0.1)"
          borderColor="rgba(249,115,22,0.2)"
          isActive={selectedPriority === 'High'}
          onClick={() => onSelectPriority(selectedPriority === 'High' ? 'All' : 'High')}
        />
        <PriorityButton
          level="Medium"
          count={medium}
          color="#F59E0B"
          bgColor="rgba(245,158,11,0.1)"
          borderColor="rgba(245,158,11,0.2)"
          isActive={selectedPriority === 'Medium'}
          onClick={() => onSelectPriority(selectedPriority === 'Medium' ? 'All' : 'Medium')}
        />
        <PriorityButton
          level="Low"
          count={low}
          color="#22C55E"
          bgColor="rgba(34,197,94,0.1)"
          borderColor="rgba(34,197,94,0.2)"
          isActive={selectedPriority === 'Low'}
          onClick={() => onSelectPriority(selectedPriority === 'Low' ? 'All' : 'Low')}
        />

        {selectedPriority !== 'All' && (
          <>
            <span style={{ color: '#1E2435', fontSize: 14 }}>|</span>
            <button
              onClick={() => onSelectPriority('All')}
              style={{
                fontSize: '11px',
                fontWeight: 500,
                color: '#3B82F6',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: '4px 0',
              }}
            >
              Clear filter
            </button>
          </>
        )}
      </div>
    </div>
  );
};
