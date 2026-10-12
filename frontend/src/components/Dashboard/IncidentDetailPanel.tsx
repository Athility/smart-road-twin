'use client';

import React, { useState } from 'react';
import { TelemetryRecord, WeightProfile } from '@/types/telemetry';
import { getInspectionImageUrl } from '@/lib/api';
import {
  X,
  MapPin,
  Ruler,
  Car,
  Hospital,
  Zap,
  Eye,
  Waves,
  Clock,
  Activity,
  Layers,
  Sparkles,
} from 'lucide-react';
import { PipelineFlowPanel } from './PipelineFlowPanel';

interface IncidentDetailPanelProps {
  record: TelemetryRecord | null;
  onClose: () => void;
  weightProfile?: WeightProfile;
}

// Priority config map
const PRIORITY_CONFIG = {
  Critical: { color: '#EF4444', bg: 'rgba(239,68,68,0.12)', border: 'rgba(239,68,68,0.3)' },
  High:     { color: '#F97316', bg: 'rgba(249,115,22,0.12)', border: 'rgba(249,115,22,0.3)' },
  Medium:   { color: '#F59E0B', bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.3)' },
  Low:      { color: '#22C55E', bg: 'rgba(34,197,94,0.12)',  border: 'rgba(34,197,94,0.3)' },
};

function Row({
  icon,
  label,
  value,
  subvalue,
  valueColor,
  mono = false,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  subvalue?: React.ReactNode;
  valueColor?: string;
  mono?: boolean;
}) {
  return (
    <div
      className="flex items-center justify-between"
      style={{ padding: '8px 0', borderBottom: '1px solid #1E2435' }}
    >
      <div className="flex items-center gap-2" style={{ color: '#64748B' }}>
        <span style={{ flexShrink: 0 }}>{icon}</span>
        <span style={{ fontSize: '11px', fontWeight: 600, color: '#94A3B8' }}>{label}</span>
      </div>
      <div className="flex flex-col items-end">
        <span
          style={{
            fontSize: '12px',
            fontWeight: 700,
            color: valueColor ?? '#EEF2FF',
            fontFamily: mono ? 'ui-monospace, SFMono-Regular, monospace' : undefined,
            textAlign: 'right',
            maxWidth: 180,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {value}
        </span>
        {subvalue && (
          <span style={{ fontSize: '9px', color: '#64748B', fontFamily: 'monospace' }}>
            {subvalue}
          </span>
        )}
      </div>
    </div>
  );
}

function SectionBadge({ number, title }: { number: number; title: string }) {
  return (
    <div className="flex items-center gap-1.5" style={{ padding: '10px 0 3px' }}>
      <span
        style={{
          fontSize: '9px',
          fontFamily: 'monospace',
          fontWeight: 700,
          background: '#1E293B',
          color: '#60A5FA',
          padding: '1px 5px',
          borderRadius: 3,
        }}
      >
        {number}
      </span>
      <span
        style={{
          fontSize: '10px',
          fontWeight: 700,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: '#8892A4',
        }}
      >
        {title}
      </span>
    </div>
  );
}

function TopsisBar({ score, priority }: { score: number; priority: string }) {
  const config = PRIORITY_CONFIG[priority as keyof typeof PRIORITY_CONFIG] ?? PRIORITY_CONFIG.Low;
  const pct = Math.min(100, Math.max(0, Math.round(score * 100)));

  return (
    <div style={{ marginTop: 2 }}>
      <div className="flex justify-between items-center" style={{ marginBottom: 4 }}>
        <span style={{ fontSize: '11px', color: '#94A3B8', fontWeight: 600 }}>Urgency Index</span>
        <span
          style={{
            fontSize: '18px',
            fontWeight: 700,
            fontFamily: 'ui-monospace, SFMono-Regular, monospace',
            color: config.color,
            letterSpacing: '-0.02em',
          }}
        >
          {score.toFixed(4)}
        </span>
      </div>
      <div
        style={{
          width: '100%',
          height: 7,
          background: '#161B27',
          borderRadius: 4,
          overflow: 'hidden',
          border: '1px solid #1E2435',
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: `linear-gradient(90deg, #3B82F6 0%, ${config.color} 100%)`,
            borderRadius: 4,
            transition: 'width 0.4s ease',
            boxShadow: `0 0 10px ${config.color}60`,
          }}
        />
      </div>
      <div className="flex justify-between" style={{ marginTop: 3 }}>
        <span style={{ fontSize: '9px', color: '#475569', fontFamily: 'monospace' }}>0.00 (Low)</span>
        <span style={{ fontSize: '9px', color: '#475569', fontFamily: 'monospace' }}>1.00 (Critical)</span>
      </div>
    </div>
  );
}

export const IncidentDetailPanel: React.FC<IncidentDetailPanelProps> = ({
  record,
  onClose,
  weightProfile,
}) => {
  const [imageError, setImageError] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'pipeline'>('overview');
  const panelRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!record) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    // Autofocus panel on open for immediate keyboard interaction
    panelRef.current?.focus();
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [record, onClose]);

  if (!record) {
    return (
      <div
        className="flex flex-col items-center justify-center text-center"
        style={{
          flex: 1,
          padding: '32px 20px',
          color: '#8892A4',
        }}
      >
        <Activity style={{ width: 32, height: 32, marginBottom: 12, opacity: 0.4 }} />
        <p style={{ fontSize: '13px', fontWeight: 500, color: '#94A3B8', marginBottom: 6 }}>
          No incident selected
        </p>
        <p style={{ fontSize: '11px', color: '#8892A4', lineHeight: 1.5 }}>
          Click a marker on the map or<br />a row in the incident list
        </p>
      </div>
    );
  }

  const pt = record.processed_telemetry;
  const config = PRIORITY_CONFIG[record.priority_level as keyof typeof PRIORITY_CONFIG] ?? PRIORITY_CONFIG.Low;
  const hasImpact = pt?.dynamic_impact_confirmed || record.z_accel_g > 1.5;
  const isSonar = (pt?.detection_modality || '').includes('sonar');

  // Format data source label
  const dataSource = record.data_source || 'MULTIMODAL';

  // Format image URL
  const imgUrl = getInspectionImageUrl(record.image_url, record.image_id);

  const damageClassCode = record.damage_class || 'D40';
  const damageClassName = record.damage_label || (
    damageClassCode === 'D40' ? 'Pothole' :
    damageClassCode === 'D20' ? 'Alligator Crack' :
    damageClassCode === 'D00' ? 'Longitudinal Crack' : 'Transverse Crack'
  );

  const confidencePct = record.confidence != null ? `${(record.confidence * 100).toFixed(1)}%` : '96.2%';

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
      ref={panelRef}
      role="dialog"
      aria-modal="false"
      aria-label={`Incident detail for ${record.id}`}
      tabIndex={-1}
      className="slide-in-right flex flex-col focus:outline-none"
      style={{
        flex: 1,
        overflowY: 'auto',
        padding: '0 16px 20px',
        outline: 'none',
      }}
    >
      {/* ── Top Header with ID & Deselect ───────────────────────────────── */}
      <div
        className="flex items-start justify-between"
        style={{
          paddingTop: 12,
          paddingBottom: 10,
          borderBottom: '1px solid #1E2435',
          marginBottom: 8,
        }}
      >
        <div>
          <div className="flex items-center gap-2" style={{ marginBottom: 4 }}>
            <span
              style={{
                fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                fontWeight: 700,
                fontSize: '14px',
                color: '#EEF2FF',
                letterSpacing: '-0.02em',
              }}
            >
              {record.id}
            </span>
          </div>
          <div className="flex items-center gap-1.5" style={{ color: '#64748B', fontSize: '10px' }}>
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
            color: '#64748B',
            padding: 4,
            borderRadius: 4,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onMouseOver={(e) => (e.currentTarget.style.color = '#EEF2FF')}
          onMouseOut={(e) => (e.currentTarget.style.color = '#64748B')}
        >
          <X style={{ width: 16, height: 16 }} />
        </button>
      </div>

      {/* ── Sub-navigation Tabs: Overview vs Pipeline Flow ───────────── */}
      <div
        style={{
          display: 'flex',
          gap: 4,
          background: '#0B0E14',
          padding: 3,
          borderRadius: 6,
          border: '1px solid #1E2435',
          marginBottom: 12,
        }}
      >
        <button
          onClick={() => setActiveTab('overview')}
          style={{
            flex: 1,
            padding: '5px 8px',
            fontSize: '11px',
            fontWeight: 700,
            borderRadius: 4,
            border: 'none',
            cursor: 'pointer',
            background: activeTab === 'overview' ? '#1E293B' : 'transparent',
            color: activeTab === 'overview' ? '#60A5FA' : '#64748B',
            transition: 'all 0.15s ease',
          }}
        >
          Incident Overview
        </button>
        <button
          onClick={() => setActiveTab('pipeline')}
          style={{
            flex: 1,
            padding: '5px 8px',
            fontSize: '11px',
            fontWeight: 700,
            borderRadius: 4,
            border: 'none',
            cursor: 'pointer',
            background: activeTab === 'pipeline' ? '#1E293B' : 'transparent',
            color: activeTab === 'pipeline' ? '#60A5FA' : '#64748B',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 5,
            transition: 'all 0.15s ease',
          }}
        >
          <Layers style={{ width: 12, height: 12 }} />
          Pipeline Flow (L1-L5)
        </button>
      </div>

      {activeTab === 'pipeline' ? (
        <PipelineFlowPanel record={record} />
      ) : (
        <>
          {/* ── DATA SOURCE PROVENANCE BADGE ──────────────────────────────────── */}
          <div
            style={{
              background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%)',
              border: '1px solid #2563EB40',
              borderRadius: 6,
              padding: '8px 10px',
              marginBottom: 12,
            }}
      >
        <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
          <div className="flex items-center gap-1.5">
            <Layers style={{ width: 12, height: 12, color: '#60A5FA' }} />
            <span
              style={{
                fontSize: '9px',
                fontWeight: 700,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: '#94A3B8',
              }}
            >
              DATA SOURCE
            </span>
          </div>
          <span
            style={{
              fontSize: '10px',
              fontWeight: 800,
              letterSpacing: '0.06em',
              padding: '2px 8px',
              borderRadius: 4,
              background: dataSource === 'MULTIMODAL' ? '#1D4ED825' : '#05966925',
              border: `1px solid ${dataSource === 'MULTIMODAL' ? '#3B82F660' : '#10B98160'}`,
              color: dataSource === 'MULTIMODAL' ? '#93C5FD' : '#34D399',
              fontFamily: 'ui-monospace, monospace',
            }}
          >
            {dataSource}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-1.5" style={{ fontSize: '9px', color: '#64748B' }}>
          <span className="flex items-center gap-1" style={{ color: '#93C5FD' }}>
            ● RDD2022 Visual
          </span>
          <span>·</span>
          <span className="flex items-center gap-1" style={{ color: '#A78BFA' }}>
            ● Synthetic Sensor
          </span>
          <span>·</span>
          <span className="flex items-center gap-1" style={{ color: '#34D399' }}>
            ● Synthetic GIS
          </span>
        </div>
      </div>

      {/* ────────────────────────────────────────────────────────────────────── */}
      {/* ── MANDATORY SEQUENCE (PHASE 19) ───────────────────────────────────── */}
      {/* 1. IMAGE */}
      {/* 2. Damage classification */}
      {/* 3. Confidence */}
      {/* 4. LiDAR depth */}
      {/* 5. Sonar depth */}
      {/* 6. Selected modality */}
      {/* 7. Accelerometer */}
      {/* 8. Traffic PCU */}
      {/* 9. Hospital distance */}
      {/* 10. TOPSIS score */}
      {/* 11. Priority */}
      {/* ────────────────────────────────────────────────────────────────────── */}

      {/* ── 1. IMAGE ──────────────────────────────────────────────────────── */}
      <SectionBadge number={1} title="Inspection Image (RDD2022 Camera)" />
      <div
        style={{
          position: 'relative',
          width: '100%',
          height: 140,
          background: '#0F1420',
          border: '1px solid #1E2435',
          borderRadius: 6,
          overflow: 'hidden',
          marginBottom: 10,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {!imageError ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={imgUrl}
            alt={`Road defect ${damageClassCode} - ${record.id}`}
            onError={() => setImageError(true)}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
            }}
          />
        ) : (
          <div
            className="flex flex-col items-center justify-center p-3 text-center"
            style={{ width: '100%', height: '100%', background: '#111520' }}
          >
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: '50%',
                background: '#1E2435',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: 4,
              }}
            >
              <Eye style={{ width: 16, height: 16, color: '#3B82F6' }} />
            </div>
            <span style={{ fontSize: '11px', fontWeight: 600, color: '#93C5FD' }}>
              RDD2022 Inspection Frame
            </span>
            <span style={{ fontSize: '10px', color: '#64748B', fontFamily: 'monospace' }}>
              {record.image_id || 'India_000045'}.jpg [{damageClassCode}]
            </span>
          </div>
        )}

        {/* Visual bounding box callout overlay */}
        <div
          style={{
            position: 'absolute',
            bottom: 6,
            left: 6,
            background: 'rgba(15, 23, 42, 0.85)',
            backdropFilter: 'blur(4px)',
            border: '1px solid rgba(59, 130, 246, 0.4)',
            padding: '2px 6px',
            borderRadius: 3,
            fontSize: '9px',
            fontFamily: 'monospace',
            color: '#93C5FD',
          }}
        >
          {damageClassCode} · {record.image_id || 'India_000045'}
        </div>
      </div>

      {/* ── 2. Damage classification ───────────────────────────────────────── */}
      <SectionBadge number={2} title="Damage Classification" />
      <Row
        icon={<Activity style={{ width: 13, height: 13 }} />}
        label="Defect Class"
        value={`${damageClassCode} — ${damageClassName}`}
        subvalue={`detector: ${record.detector || 'ground-truth lookup'}`}
        valueColor="#60A5FA"
        mono
      />

      {/* ── 3. Confidence ──────────────────────────────────────────────────── */}
      <SectionBadge number={3} title="Detection Confidence" />
      <div style={{ padding: '6px 0 10px', borderBottom: '1px solid #1E2435' }}>
        <div className="flex justify-between items-center" style={{ marginBottom: 4 }}>
          <span style={{ fontSize: '11px', fontWeight: 600, color: '#94A3B8' }}>
            Detection Source ({record.detector || 'ground-truth lookup'})
          </span>
          <span
            style={{
              fontSize: (record.detector || 'ground-truth lookup') === 'ground-truth lookup' ? '11px' : '13px',
              fontWeight: 700,
              fontFamily: (record.detector || 'ground-truth lookup') === 'ground-truth lookup' ? 'sans-serif' : 'monospace',
              color: (record.detector || 'ground-truth lookup') === 'ground-truth lookup' ? '#94A3B8' : '#34D399',
            }}
          >
            {(record.detector || 'ground-truth lookup') === 'ground-truth lookup'
              ? 'N/A (annotation, not a model prediction)'
              : confidencePct}
          </span>
        </div>
        {(record.detector || 'ground-truth lookup') !== 'ground-truth lookup' && (
          <div
            style={{
              width: '100%',
              height: 5,
              background: '#161B27',
              borderRadius: 3,
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: confidencePct,
                height: '100%',
                background: '#10B981',
                borderRadius: 3,
              }}
            />
          </div>
        )}
      </div>

      {/* ── 4. LiDAR depth ─────────────────────────────────────────────────── */}
      <SectionBadge number={4} title="LiDAR Depth" />
      <Row
        icon={<Ruler style={{ width: 13, height: 13 }} />}
        label="LiDAR Sensor Reading"
        value={`${record.lidar_depth_cm.toFixed(2)} cm`}
        subvalue={`Cavity: ${(record.lidar_depth_cm - 30.0).toFixed(2)} cm (30 cm baseline)`}
        mono
        valueColor="#EEF2FF"
      />

      {/* ── 5. Sonar depth ─────────────────────────────────────────────────── */}
      <SectionBadge number={5} title="Sonar Depth" />
      <Row
        icon={<Waves style={{ width: 13, height: 13 }} />}
        label="Acoustic Sonar Reading"
        value={`${record.sonar_depth_cm.toFixed(2)} cm`}
        subvalue={`Cavity: ${(record.sonar_depth_cm - 30.0).toFixed(2)} cm`}
        mono
        valueColor="#38BDF8"
      />

      {/* ── 6. Selected modality & Volume / Crack Severity ─────────────────── */}
      <SectionBadge number={6} title="Selected Modality" />
      <div
        className="flex items-center justify-between"
        style={{ padding: '8px 0', borderBottom: '1px solid #1E2435' }}
      >
        <span style={{ fontSize: '11px', fontWeight: 600, color: '#94A3B8' }}>Active Depth Sensor</span>
        <span
          style={{
            fontSize: '11px',
            fontWeight: 700,
            padding: '3px 8px',
            borderRadius: 4,
            background: isSonar ? 'rgba(56,189,248,0.15)' : 'rgba(167,139,250,0.15)',
            border: `1px solid ${isSonar ? 'rgba(56,189,248,0.35)' : 'rgba(167,139,250,0.35)'}`,
            color: isSonar ? '#38BDF8' : '#A78BFA',
            fontFamily: 'monospace',
          }}
        >
          {isSonar ? '⦿ Acoustic Sonar' : '◉ Optical LiDAR'}
        </span>
      </div>
      <Row
        icon={<Ruler style={{ width: 13, height: 13 }} />}
        label={damageClassCode === 'D40' ? 'Volume (L)' : 'Crack severity'}
        value={
          damageClassCode === 'D40' && pt?.calculated_volume_liters != null
            ? `${pt.calculated_volume_liters.toFixed(2)} L`
            : pt?.crack_severity_index != null
            ? `${pt.crack_severity_index.toFixed(3)}`
            : '—'
        }
        subvalue={
          <span className="flex items-center gap-1.5 justify-end">
            <span>
              {damageClassCode === 'D40'
                ? `Footprint: ${record.surface_area_sqm.toFixed(3)} m²`
                : `Normalized [0–1] severity · Footprint: ${record.surface_area_sqm.toFixed(3)} m²`}
            </span>
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
                  padding: '1px 4px',
                  borderRadius: 3,
                }}
                title={
                  record.area_unclamped_m2 != null
                    ? `Area clamped from ${record.area_unclamped_m2.toFixed(3)} m² to physical bounds`
                    : 'Area clamped to physical ground-plane bounds'
                }
              >
                clamped
              </span>
            )}
          </span>
        }
        mono
        valueColor={damageClassCode === 'D40' ? '#60A5FA' : '#A78BFA'}
      />

      {/* ── 7. Accelerometer ───────────────────────────────────────────────── */}
      <SectionBadge number={7} title="Accelerometer (Z-Axis)" />
      <Row
        icon={<Zap style={{ width: 13, height: 13 }} />}
        label="Z-Axis Dynamic Impact"
        value={`${record.z_accel_g.toFixed(2)} g`}
        subvalue={hasImpact ? '⚡ Dynamic Impact Confirmed (>1.5g)' : 'Nominal Vibration (<=1.35g)'}
        mono
        valueColor={hasImpact ? '#EF4444' : '#EEF2FF'}
      />

      {/* ── 8. Traffic PCU ─────────────────────────────────────────────────── */}
      <SectionBadge number={8} title="Traffic PCU" />
      <Row
        icon={<Car style={{ width: 13, height: 13 }} />}
        label="Traffic Volume"
        value={`${record.traffic_pcu.toLocaleString()} PCU`}
        subvalue={record.road_class || 'HIGH_TRAFFIC_CORRIDOR'}
        mono
        valueColor="#EEF2FF"
      />

      {/* ── 9. Hospital distance ───────────────────────────────────────────── */}
      <SectionBadge number={9} title="Hospital Distance" />
      <Row
        icon={<Hospital style={{ width: 13, height: 13 }} />}
        label="Nearest Trauma Facility"
        value={`${record.dist_hospital_km.toFixed(2)} km`}
        subvalue={record.dist_hospital_km < 1.0 ? 'Critical Emergency Corridor Proximity' : 'Standard GIS buffer'}
        mono
        valueColor={record.dist_hospital_km < 1.0 ? '#EF4444' : '#EEF2FF'}
      />

      {/* ── 10. TOPSIS score & Multi-Criteria Ranks ───────────────────────── */}
      <SectionBadge number={10} title="MCDM Prioritization (TOPSIS vs VIKOR)" />
      <div
        style={{
          background: config.bg,
          border: `1px solid ${config.border}`,
          borderRadius: 6,
          padding: '10px 12px',
          margin: '4px 0 10px',
        }}
      >
        <TopsisBar score={record.topsis_score} priority={record.priority_level} />

        {/* Side-by-side Ranks & Disagree marker */}
        <div
          style={{
            marginTop: 8,
            paddingTop: 8,
            borderTop: '1px solid rgba(255,255,255,0.08)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '11px',
          }}
        >
          <div className="flex items-center gap-3">
            <span style={{ color: '#94A3B8' }}>
              TOPSIS Rank: <strong style={{ color: config.color, fontFamily: 'monospace' }}>#{record.topsis_rank ?? '—'}</strong>
            </span>
            {record.vikor_rank !== undefined && (
              <span style={{ color: '#94A3B8' }}>
                VIKOR Rank: <strong style={{ color: '#A78BFA', fontFamily: 'monospace' }}>#{record.vikor_rank}</strong>
              </span>
            )}
          </div>

          {record.topsis_rank !== undefined &&
            record.vikor_rank !== undefined &&
            Math.abs(record.topsis_rank - record.vikor_rank) >= 2 && (
              <span
                style={{
                  fontSize: '8px',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                  background: 'rgba(239,68,68,0.2)',
                  color: '#F87171',
                  border: '1px solid rgba(239,68,68,0.45)',
                  padding: '1px 5px',
                  borderRadius: 3,
                }}
                title={`Methods disagree: TOPSIS rank #${record.topsis_rank} vs VIKOR rank #${record.vikor_rank} (Δ >= 2)`}
              >
                ≠ Methods Disagree (Δ{Math.abs(record.topsis_rank - record.vikor_rank)})
              </span>
            )}
        </div>
      </div>

      {/* ── 11. Priority ───────────────────────────────────────────────────── */}
      <SectionBadge number={11} title="Hazard Priority" />
      <div
        className="flex items-center justify-between"
        style={{
          padding: '8px 12px',
          background: config.bg,
          border: `1px solid ${config.border}`,
          borderRadius: 6,
          marginBottom: 12,
        }}
      >
        <div className="flex items-center gap-2">
          <Sparkles style={{ width: 14, height: 14, color: config.color }} />
          <span style={{ fontSize: '11px', fontWeight: 600, color: '#EEF2FF' }}>Dispatch Urgency</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            style={{
              fontSize: '11px',
              fontWeight: 800,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: config.color,
              fontFamily: 'monospace',
            }}
          >
            {record.priority_level} Priority
          </span>
          <span
            style={{
              fontSize: '9px',
              fontWeight: 700,
              background: 'rgba(255,255,255,0.08)',
              border: '1px solid rgba(255,255,255,0.18)',
              color: '#CBD5E1',
              padding: '2px 6px',
              borderRadius: 3,
              fontFamily: 'sans-serif',
              letterSpacing: '0.02em',
            }}
            title={`Active weight profile: ${weightProfile || 'safety-first'}`}
          >
            {weightProfile === 'traffic-first'
              ? 'Traffic-First'
              : weightProfile === 'equity-first'
              ? 'Equity-First'
              : 'Safety-First'}
          </span>
        </div>
      </div>

          {/* ── Geographic Context Footer ──────────────────────────────────────── */}
          <div style={{ marginTop: 4, paddingTop: 8, borderTop: '1px solid #1E2435' }}>
            <div className="flex items-center justify-between" style={{ fontSize: '10px', color: '#64748B' }}>
              <span className="flex items-center gap-1">
                <MapPin style={{ width: 10, height: 10 }} />
                {record.latitude.toFixed(4)}, {record.longitude.toFixed(4)}
              </span>
              <span style={{ fontFamily: 'monospace' }}>
                {record.road_name || 'Mumbai Metropolitan Corridor'}
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

