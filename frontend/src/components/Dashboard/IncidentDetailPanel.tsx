'use client';

import React from 'react';
import { TelemetryRecord } from '@/types/telemetry';
import {
  X,
  MapPin,
  Ruler,
  Droplets,
  Car,
  Hospital,
  Zap,
  Eye,
  Waves,
  Clock,
  Activity,
  Navigation,
} from 'lucide-react';

interface IncidentDetailPanelProps {
  record: TelemetryRecord | null;
  onClose: () => void;
}

// Priority config map
const PRIORITY_CONFIG = {
  Critical: { color: '#EF4444', bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.2)' },
  High:     { color: '#F97316', bg: 'rgba(249,115,22,0.08)', border: 'rgba(249,115,22,0.2)' },
  Medium:   { color: '#F59E0B', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.2)' },
  Low:      { color: '#22C55E', bg: 'rgba(34,197,94,0.08)',  border: 'rgba(34,197,94,0.2)' },
};

function Row({
  icon,
  label,
  value,
  valueColor,
  mono = false,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  valueColor?: string;
  mono?: boolean;
}) {
  return (
    <div
      className="flex items-center justify-between"
      style={{ padding: '7px 0', borderBottom: '1px solid #1E2435' }}
    >
      <div className="flex items-center gap-2" style={{ color: '#4D5A72' }}>
        <span style={{ flexShrink: 0 }}>{icon}</span>
        <span style={{ fontSize: '11px', fontWeight: 500, color: '#8892A4' }}>{label}</span>
      </div>
      <span
        style={{
          fontSize: '12px',
          fontWeight: 600,
          color: valueColor ?? '#EEF2FF',
          fontFamily: mono ? 'ui-monospace, SFMono-Regular, monospace' : undefined,
          textAlign: 'right',
          maxWidth: 160,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {value}
      </span>
    </div>
  );
}

function SectionHeader({ title }: { title: string }) {
  return (
    <div style={{ padding: '12px 0 4px' }}>
      <span
        style={{
          fontSize: '10px',
          fontWeight: 700,
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          color: '#4D5A72',
        }}
      >
        {title}
      </span>
    </div>
  );
}

function TopsisBar({ score, priority }: { score: number; priority: string }) {
  const config = PRIORITY_CONFIG[priority as keyof typeof PRIORITY_CONFIG] ?? PRIORITY_CONFIG.Low;
  const pct = Math.round(score * 100);

  return (
    <div style={{ marginTop: 4 }}>
      <div className="flex justify-between items-center" style={{ marginBottom: 6 }}>
        <span style={{ fontSize: '11px', color: '#8892A4', fontWeight: 500 }}>TOPSIS Score</span>
        <span
          style={{
            fontSize: '20px',
            fontWeight: 700,
            fontFamily: 'ui-monospace, SFMono-Regular, monospace',
            color: config.color,
            letterSpacing: '-0.03em',
          }}
        >
          {score.toFixed(4)}
        </span>
      </div>
      <div
        style={{
          width: '100%',
          height: 6,
          background: '#1E2435',
          borderRadius: 3,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: config.color,
            borderRadius: 3,
            transition: 'width 0.4s ease',
            boxShadow: `0 0 8px ${config.color}60`,
          }}
        />
      </div>
      <div className="flex justify-between" style={{ marginTop: 3 }}>
        <span style={{ fontSize: '9px', color: '#4D5A72' }}>0.00</span>
        <span style={{ fontSize: '9px', color: '#4D5A72' }}>1.00</span>
      </div>
    </div>
  );
}

export const IncidentDetailPanel: React.FC<IncidentDetailPanelProps> = ({
  record,
  onClose,
}) => {
  if (!record) {
    return (
      <div
        className="flex flex-col items-center justify-center text-center"
        style={{
          flex: 1,
          padding: '32px 20px',
          color: '#4D5A72',
        }}
      >
        <Activity style={{ width: 32, height: 32, marginBottom: 12, opacity: 0.4 }} />
        <p style={{ fontSize: '13px', fontWeight: 500, color: '#4D5A72', marginBottom: 6 }}>
          No incident selected
        </p>
        <p style={{ fontSize: '11px', color: '#374163', lineHeight: 1.5 }}>
          Click a marker on the map or<br />a row in the incident list
        </p>
      </div>
    );
  }

  const pt = record.processed_telemetry;
  const config = PRIORITY_CONFIG[record.priority_level as keyof typeof PRIORITY_CONFIG] ?? PRIORITY_CONFIG.Low;
  const isSubmerged = pt?.is_submerged;
  const hasImpact = pt?.dynamic_impact_confirmed;
  const isSonar = pt?.detection_modality === 'sonar_submerged_acoustic';

  const formattedTime = (() => {
    try {
      return new Date(record.timestamp).toLocaleString('en-IN', {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false,
      });
    } catch {
      return record.timestamp;
    }
  })();

  return (
    <div
      className="slide-in-right flex flex-col overflow-hidden"
      style={{
        flex: 1,
        overflowY: 'auto',
        padding: '0 16px 16px',
      }}
    >
      {/* ── Header ─────────────────────────────────────────── */}
      <div
        className="flex items-start justify-between"
        style={{
          paddingTop: 14,
          paddingBottom: 12,
          borderBottom: '1px solid #1E2435',
          marginBottom: 2,
        }}
      >
        <div>
          <div className="flex items-center gap-2" style={{ marginBottom: 4 }}>
            <span
              style={{
                fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                fontWeight: 700,
                fontSize: '15px',
                color: '#EEF2FF',
                letterSpacing: '-0.02em',
              }}
            >
              {record.id}
            </span>
            <span
              style={{
                fontSize: '10px',
                fontWeight: 700,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                background: config.bg,
                border: `1px solid ${config.border}`,
                color: config.color,
                padding: '2px 8px',
                borderRadius: 4,
              }}
            >
              {record.priority_level}
            </span>
          </div>
          <div className="flex items-center gap-1.5" style={{ color: '#4D5A72', fontSize: '10px' }}>
            <Clock style={{ width: 10, height: 10 }} />
            <span>{formattedTime}</span>
          </div>
        </div>
        <button
          onClick={onClose}
          title="Deselect incident"
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: '#4D5A72',
            padding: 4,
            borderRadius: 4,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onMouseOver={(e) => (e.currentTarget.style.color = '#8892A4')}
          onMouseOut={(e) => (e.currentTarget.style.color = '#4D5A72')}
        >
          <X style={{ width: 16, height: 16 }} />
        </button>
      </div>

      {/* ── TOPSIS Score Bar ───────────────────────────────── */}
      <div
        style={{
          background: config.bg,
          border: `1px solid ${config.border}`,
          borderRadius: 6,
          padding: '10px 12px',
          marginBottom: 4,
        }}
      >
        <TopsisBar score={record.topsis_score} priority={record.priority_level} />
      </div>

      {/* ── Status Badges ──────────────────────────────────── */}
      <div className="flex flex-wrap gap-1.5" style={{ marginTop: 8, marginBottom: 4 }}>
        <span
          style={{
            fontSize: '10px',
            fontWeight: 600,
            padding: '3px 8px',
            borderRadius: 3,
            background: isSonar ? 'rgba(56,189,248,0.1)' : 'rgba(167,139,250,0.1)',
            border: `1px solid ${isSonar ? 'rgba(56,189,248,0.25)' : 'rgba(167,139,250,0.25)'}`,
            color: isSonar ? '#38BDF8' : '#A78BFA',
          }}
        >
          {isSonar ? '⦿ Acoustic Sonar' : '◉ Optical LiDAR'}
        </span>
        {isSubmerged && (
          <span
            style={{
              fontSize: '10px',
              fontWeight: 600,
              padding: '3px 8px',
              borderRadius: 3,
              background: 'rgba(56,189,248,0.1)',
              border: '1px solid rgba(56,189,248,0.25)',
              color: '#38BDF8',
            }}
          >
            ≋ Submerged
          </span>
        )}
        {hasImpact && (
          <span
            style={{
              fontSize: '10px',
              fontWeight: 600,
              padding: '3px 8px',
              borderRadius: 3,
              background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.25)',
              color: '#EF4444',
            }}
          >
            ⚡ Impact &gt;1.5g
          </span>
        )}
      </div>

      {/* ── Dimensions ────────────────────────────────────── */}
      <SectionHeader title="Dimensions" />
      <Row
        icon={<Ruler style={{ width: 13, height: 13 }} />}
        label="Est. Depth"
        value={`${pt?.estimated_depth_cm ?? '—'} cm`}
        mono
        valueColor="#EEF2FF"
      />
      <Row
        icon={<Droplets style={{ width: 13, height: 13 }} />}
        label="Defect Volume"
        value={`${pt?.calculated_volume_liters ?? '—'} L`}
        mono
        valueColor="#60A5FA"
      />
      <Row
        icon={<Activity style={{ width: 13, height: 13 }} />}
        label="Surface Area"
        value={`${record.surface_area_sqm} m²`}
        mono
      />

      {/* ── Environment ───────────────────────────────────── */}
      <SectionHeader title="Environment" />
      <Row
        icon={<Eye style={{ width: 13, height: 13 }} />}
        label="Luminance"
        value={`${record.mean_luminance} lux`}
        mono
      />
      <Row
        icon={<Waves style={{ width: 13, height: 13 }} />}
        label="Rain Detected"
        value={record.rain_detected ? 'Yes' : 'No'}
        valueColor={record.rain_detected ? '#38BDF8' : '#4D5A72'}
      />
      <Row
        icon={<Zap style={{ width: 13, height: 13 }} />}
        label="Z-Accel"
        value={`${record.z_accel_g} g`}
        mono
        valueColor={hasImpact ? '#EF4444' : '#EEF2FF'}
      />

      {/* ── Traffic & Context ─────────────────────────────── */}
      <SectionHeader title="Traffic & Context" />
      <Row
        icon={<Car style={{ width: 13, height: 13 }} />}
        label="Traffic Volume"
        value={`${record.traffic_pcu.toLocaleString()} PCU`}
        mono
        valueColor="#EEF2FF"
      />
      <Row
        icon={<Navigation style={{ width: 13, height: 13 }} />}
        label="Vehicle Speed"
        value={`${record.speed_kmh} km/h`}
        mono
      />
      <Row
        icon={<Hospital style={{ width: 13, height: 13 }} />}
        label="Hospital Dist."
        value={`${record.dist_hospital_km} km`}
        mono
        valueColor={record.dist_hospital_km < 1.0 ? '#EF4444' : '#EEF2FF'}
      />

      {/* ── Location ──────────────────────────────────────── */}
      <SectionHeader title="Location" />
      <Row
        icon={<MapPin style={{ width: 13, height: 13 }} />}
        label="Coordinates"
        value={`${record.latitude.toFixed(4)}, ${record.longitude.toFixed(4)}`}
        mono
      />
      <Row
        icon={<Car style={{ width: 13, height: 13 }} />}
        label="Detected By"
        value={record.vehicle_id}
      />

      {/* Raw sensor readings (collapsed info row) */}
      <div style={{ marginTop: 10, paddingTop: 8, borderTop: '1px solid #1E2435' }}>
        <span style={{ fontSize: '10px', color: '#374163', lineHeight: 1.6, display: 'block' }}>
          <span style={{ color: '#4D5A72' }}>LiDAR raw:</span>{' '}
          <span style={{ fontFamily: 'monospace' }}>{record.lidar_depth_cm} cm</span>
          {'  '}
          <span style={{ color: '#4D5A72' }}>Sonar raw:</span>{' '}
          <span style={{ fontFamily: 'monospace' }}>{record.sonar_depth_cm} cm</span>
          {'  '}
          <span style={{ color: '#4D5A72' }}>Baseline:</span>{' '}
          <span style={{ fontFamily: 'monospace' }}>30 cm</span>
        </span>
      </div>
    </div>
  );
};
