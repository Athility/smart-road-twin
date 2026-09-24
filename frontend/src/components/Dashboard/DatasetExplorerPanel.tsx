'use client';

import React, { useState, useEffect } from 'react';
import { DatasetExplorerStats, TelemetryRecord } from '@/types/telemetry';
import { fetchDatasetExplorerStats, FALLBACK_EXPLORER_STATS } from '@/lib/api';
import {
  Database,
  Layers,
  Ruler,
  Waves,
  Zap,
  Car,
  Hospital,
  CloudRain,
  Sparkles,
  RefreshCw,
  ExternalLink,
} from 'lucide-react';

interface DatasetExplorerPanelProps {
  onSelectRecord?: (recordId: string) => void;
  onSwitchToMap?: () => void;
}

const CLASS_INFO: Record<string, { name: string; color: string; desc: string }> = {
  D00: { name: 'Longitudinal Crack', color: '#60A5FA', desc: 'Parallel pavement distress along wheelpath' },
  D10: { name: 'Transverse Crack',   color: '#38BDF8', desc: 'Perpendicular thermal contraction crack' },
  D20: { name: 'Alligator Crack',    color: '#F59E0B', desc: 'Interconnected structural fatigue cracking' },
  D40: { name: 'Pothole',            color: '#EF4444', desc: 'Structural cavity causing dynamic vehicular shock' },
};

export const DatasetExplorerPanel: React.FC<DatasetExplorerPanelProps> = ({
  onSelectRecord,
  onSwitchToMap,
}) => {
  const [stats, setStats] = useState<DatasetExplorerStats>(FALLBACK_EXPLORER_STATS);
  const [loading, setLoading] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null);

  const loadStats = async () => {
    setLoading(true);
    try {
      const { stats: resStats, isLive: live } = await fetchDatasetExplorerStats();
      setStats(resStats);
      setIsLive(live);
    } catch (err) {
      console.error('Failed to load explorer stats:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStats();
  }, []);

  const total = stats.total_records || 1;
  const pDist = stats.priority_distribution || { Critical: 0, High: 0, Medium: 0, Low: 0 };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
        width: '100%',
      }}
    >
      {/* ── Top Header Strip ──────────────────────────────────────────────── */}
      <div
        className="flex flex-wrap items-center justify-between gap-3"
        style={{
          background: 'linear-gradient(135deg, #111520 0%, #161B27 100%)',
          border: '1px solid #1E2435',
          borderRadius: 8,
          padding: '14px 18px',
        }}
      >
        <div className="flex items-center gap-3">
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: 6,
              background: 'linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 12px rgba(37,99,235,0.4)',
            }}
          >
            <Database style={{ width: 18, height: 18, color: '#EFF6FF' }} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 style={{ fontSize: '15px', fontWeight: 700, color: '#EEF2FF', margin: 0 }}>
                SRMD Research Dataset Explorer
              </h2>
              <span
                style={{
                  fontSize: '9px',
                  fontWeight: 800,
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  padding: '2px 7px',
                  borderRadius: 3,
                  background: isLive ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)',
                  border: `1px solid ${isLive ? 'rgba(16,185,129,0.3)' : 'rgba(245,158,11,0.3)'}`,
                  color: isLive ? '#34D399' : '#FBBF24',
                  fontFamily: 'monospace',
                }}
              >
                {isLive ? 'LIVE DATASET' : 'SRMD SYNTHETIC BASELINE'}
              </span>
            </div>
            <p style={{ fontSize: '11px', color: '#64748B', margin: 0, marginTop: 2 }}>
              Smart Road Multimodal Dataset (SRMD) — RDD2022 Visual Ground Truth with Controlled Physical & GIS Layers
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={loadStats}
            disabled={loading}
            style={{
              background: '#1A2030',
              border: '1px solid #252D40',
              color: '#93C5FD',
              fontSize: '11px',
              fontWeight: 600,
              padding: '6px 12px',
              borderRadius: 5,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <RefreshCw style={{ width: 12, height: 12, animation: loading ? 'spin 1s linear infinite' : undefined }} />
            Refresh
          </button>
          {onSwitchToMap && (
            <button
              onClick={onSwitchToMap}
              style={{
                background: '#2563EB',
                border: '1px solid #3B82F6',
                color: '#EEF2FF',
                fontSize: '11px',
                fontWeight: 600,
                padding: '6px 12px',
                borderRadius: 5,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              <ExternalLink style={{ width: 12, height: 12 }} />
              Open Map Twin
            </button>
          )}
        </div>
      </div>

      {/* ── ROW 1: TOTAL RECORDS + RDD2022 DAMAGE CLASS DISTRIBUTION ───────── */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 12,
        }}
      >
        {/* Total Records Card */}
        <div
          style={{
            background: '#111520',
            border: '1px solid #1E2435',
            borderRadius: 8,
            padding: '14px 16px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
        >
          <div className="flex items-center justify-between" style={{ marginBottom: 6 }}>
            <span style={{ fontSize: '11px', fontWeight: 600, color: '#8892A4', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Total Records
            </span>
            <Database style={{ width: 14, height: 14, color: '#60A5FA' }} />
          </div>
          <div className="flex items-baseline gap-2">
            <span style={{ fontSize: '26px', fontWeight: 800, color: '#EEF2FF', fontFamily: 'monospace' }}>
              {stats.total_records}
            </span>
            <span style={{ fontSize: '11px', color: '#64748B' }}>Multimodal Events</span>
          </div>
          <div style={{ marginTop: 8, fontSize: '10px', color: '#64748B', display: 'flex', alignItems: 'center', gap: 4 }}>
            <Layers style={{ width: 10, height: 10, color: '#3B82F6' }} />
            <span>RDD2022 Ground Truth Verified</span>
          </div>
        </div>

        {/* D00 Longitudinal Crack */}
        <div
          style={{
            background: '#111520',
            border: '1px solid #1E2435',
            borderRadius: 8,
            padding: '14px 16px',
          }}
        >
          <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
            <span style={{ fontSize: '10px', fontWeight: 700, color: CLASS_INFO.D00.color, letterSpacing: '0.06em' }}>
              [D00] Longitudinal
            </span>
            <span style={{ fontSize: '10px', color: '#64748B' }}>Crack</span>
          </div>
          <div className="flex items-baseline gap-2">
            <span style={{ fontSize: '24px', fontWeight: 800, color: '#EEF2FF', fontFamily: 'monospace' }}>
              {stats.d00_count}
            </span>
            <span style={{ fontSize: '11px', color: '#64748B' }}>
              ({Math.round((stats.d00_count / total) * 100)}%)
            </span>
          </div>
          <p style={{ fontSize: '10px', color: '#475569', margin: '6px 0 0', lineHeight: 1.3 }}>
            {CLASS_INFO.D00.desc}
          </p>
        </div>

        {/* D10 Transverse Crack */}
        <div
          style={{
            background: '#111520',
            border: '1px solid #1E2435',
            borderRadius: 8,
            padding: '14px 16px',
          }}
        >
          <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
            <span style={{ fontSize: '10px', fontWeight: 700, color: CLASS_INFO.D10.color, letterSpacing: '0.06em' }}>
              [D10] Transverse
            </span>
            <span style={{ fontSize: '10px', color: '#64748B' }}>Crack</span>
          </div>
          <div className="flex items-baseline gap-2">
            <span style={{ fontSize: '24px', fontWeight: 800, color: '#EEF2FF', fontFamily: 'monospace' }}>
              {stats.d10_count}
            </span>
            <span style={{ fontSize: '11px', color: '#64748B' }}>
              ({Math.round((stats.d10_count / total) * 100)}%)
            </span>
          </div>
          <p style={{ fontSize: '10px', color: '#475569', margin: '6px 0 0', lineHeight: 1.3 }}>
            {CLASS_INFO.D10.desc}
          </p>
        </div>

        {/* D20 Alligator Crack */}
        <div
          style={{
            background: '#111520',
            border: '1px solid #1E2435',
            borderRadius: 8,
            padding: '14px 16px',
          }}
        >
          <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
            <span style={{ fontSize: '10px', fontWeight: 700, color: CLASS_INFO.D20.color, letterSpacing: '0.06em' }}>
              [D20] Alligator
            </span>
            <span style={{ fontSize: '10px', color: '#64748B' }}>Crack</span>
          </div>
          <div className="flex items-baseline gap-2">
            <span style={{ fontSize: '24px', fontWeight: 800, color: '#EEF2FF', fontFamily: 'monospace' }}>
              {stats.d20_count}
            </span>
            <span style={{ fontSize: '11px', color: '#64748B' }}>
              ({Math.round((stats.d20_count / total) * 100)}%)
            </span>
          </div>
          <p style={{ fontSize: '10px', color: '#475569', margin: '6px 0 0', lineHeight: 1.3 }}>
            {CLASS_INFO.D20.desc}
          </p>
        </div>

        {/* D40 Pothole */}
        <div
          style={{
            background: '#111520',
            border: '1px solid #1E2435',
            borderRadius: 8,
            padding: '14px 16px',
          }}
        >
          <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
            <span style={{ fontSize: '10px', fontWeight: 700, color: CLASS_INFO.D40.color, letterSpacing: '0.06em' }}>
              [D40] Pothole
            </span>
            <span style={{ fontSize: '10px', color: '#EF4444', fontWeight: 700 }}>High Urgency</span>
          </div>
          <div className="flex items-baseline gap-2">
            <span style={{ fontSize: '24px', fontWeight: 800, color: '#EEF2FF', fontFamily: 'monospace' }}>
              {stats.d40_count}
            </span>
            <span style={{ fontSize: '11px', color: '#64748B' }}>
              ({Math.round((stats.d40_count / total) * 100)}%)
            </span>
          </div>
          <p style={{ fontSize: '10px', color: '#475569', margin: '6px 0 0', lineHeight: 1.3 }}>
            {CLASS_INFO.D40.desc}
          </p>
        </div>
      </div>

      {/* ── ROW 2: SENSOR & URBAN CONTEXT AVERAGES ──────────────────────────── */}
      <div
        style={{
          background: '#111520',
          border: '1px solid #1E2435',
          borderRadius: 8,
          padding: '16px',
        }}
      >
        <div style={{ marginBottom: 12 }}>
          <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#8892A4' }}>
            Multi-Sensor & Context Averages
          </span>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: 12,
          }}
        >
          {/* Average LiDAR depth */}
          <div style={{ background: '#161B27', border: '1px solid #1E2435', borderRadius: 6, padding: '12px' }}>
            <div className="flex items-center gap-1.5" style={{ color: '#60A5FA', marginBottom: 4 }}>
              <Ruler style={{ width: 13, height: 13 }} />
              <span style={{ fontSize: '11px', fontWeight: 600 }}>Avg LiDAR Depth</span>
            </div>
            <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'monospace', color: '#EEF2FF' }}>
              {stats.avg_lidar_depth.toFixed(2)} cm
            </div>
            <span style={{ fontSize: '9px', color: '#64748B' }}>Baseline 30 cm chassis height</span>
          </div>

          {/* Average Sonar depth */}
          <div style={{ background: '#161B27', border: '1px solid #1E2435', borderRadius: 6, padding: '12px' }}>
            <div className="flex items-center gap-1.5" style={{ color: '#38BDF8', marginBottom: 4 }}>
              <Waves style={{ width: 13, height: 13 }} />
              <span style={{ fontSize: '11px', fontWeight: 600 }}>Avg Sonar Depth</span>
            </div>
            <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'monospace', color: '#EEF2FF' }}>
              {stats.avg_sonar_depth.toFixed(2)} cm
            </div>
            <span style={{ fontSize: '9px', color: '#64748B' }}>Acoustic submerged echoes</span>
          </div>

          {/* Average Acceleration */}
          <div style={{ background: '#161B27', border: '1px solid #1E2435', borderRadius: 6, padding: '12px' }}>
            <div className="flex items-center gap-1.5" style={{ color: '#F87171', marginBottom: 4 }}>
              <Zap style={{ width: 13, height: 13 }} />
              <span style={{ fontSize: '11px', fontWeight: 600 }}>Avg Acceleration</span>
            </div>
            <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'monospace', color: '#EEF2FF' }}>
              {stats.avg_acceleration.toFixed(2)} g
            </div>
            <span style={{ fontSize: '9px', color: '#64748B' }}>Threshold &gt; 1.5g confirms impact</span>
          </div>

          {/* Average Traffic PCU */}
          <div style={{ background: '#161B27', border: '1px solid #1E2435', borderRadius: 6, padding: '12px' }}>
            <div className="flex items-center gap-1.5" style={{ color: '#A78BFA', marginBottom: 4 }}>
              <Car style={{ width: 13, height: 13 }} />
              <span style={{ fontSize: '11px', fontWeight: 600 }}>Avg Traffic PCU</span>
            </div>
            <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'monospace', color: '#EEF2FF' }}>
              {stats.avg_traffic_pcu.toLocaleString()}
            </div>
            <span style={{ fontSize: '9px', color: '#64748B' }}>Passenger Car Units / hour</span>
          </div>

          {/* Average Hospital Distance */}
          <div style={{ background: '#161B27', border: '1px solid #1E2435', borderRadius: 6, padding: '12px' }}>
            <div className="flex items-center gap-1.5" style={{ color: '#34D399', marginBottom: 4 }}>
              <Hospital style={{ width: 13, height: 13 }} />
              <span style={{ fontSize: '11px', fontWeight: 600 }}>Avg Hospital Dist.</span>
            </div>
            <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'monospace', color: '#EEF2FF' }}>
              {stats.avg_hospital_distance.toFixed(2)} km
            </div>
            <span style={{ fontSize: '9px', color: '#64748B' }}>Geodesic GIS trauma proximity</span>
          </div>
        </div>
      </div>

      {/* ── ROW 3: ENVIRONMENTAL SWITCHING & PRIORITY DISTRIBUTION ─────────── */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: 12,
        }}
      >
        {/* Environmental Sensor Switching Breakdown */}
        <div
          style={{
            background: '#111520',
            border: '1px solid #1E2435',
            borderRadius: 8,
            padding: '16px',
          }}
        >
          <div className="flex items-center justify-between" style={{ marginBottom: 12 }}>
            <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#8892A4' }}>
              Environmental Modality Selection
            </span>
            <CloudRain style={{ width: 14, height: 14, color: '#38BDF8' }} />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {/* Rain Percentage */}
            <div>
              <div className="flex justify-between" style={{ fontSize: '11px', marginBottom: 4 }}>
                <span style={{ color: '#94A3B8' }}>Rain / Precipitation Percentage</span>
                <span style={{ color: '#38BDF8', fontFamily: 'monospace', fontWeight: 700 }}>
                  {stats.rain_percentage.toFixed(1)}%
                </span>
              </div>
              <div style={{ width: '100%', height: 6, background: '#161B27', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ width: `${stats.rain_percentage}%`, height: '100%', background: '#38BDF8', borderRadius: 3 }} />
              </div>
            </div>

            {/* LiDAR Selected Percentage */}
            <div>
              <div className="flex justify-between" style={{ fontSize: '11px', marginBottom: 4 }}>
                <span style={{ color: '#94A3B8' }}>LiDAR Modality Selected (Dry / Clear)</span>
                <span style={{ color: '#A78BFA', fontFamily: 'monospace', fontWeight: 700 }}>
                  {stats.lidar_selected_percentage.toFixed(1)}%
                </span>
              </div>
              <div style={{ width: '100%', height: 6, background: '#161B27', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ width: `${stats.lidar_selected_percentage}%`, height: '100%', background: '#A78BFA', borderRadius: 3 }} />
              </div>
            </div>

            {/* Sonar Selected Percentage */}
            <div>
              <div className="flex justify-between" style={{ fontSize: '11px', marginBottom: 4 }}>
                <span style={{ color: '#94A3B8' }}>Sonar Modality Selected (Rain / Night)</span>
                <span style={{ color: '#38BDF8', fontFamily: 'monospace', fontWeight: 700 }}>
                  {stats.sonar_selected_percentage.toFixed(1)}%
                </span>
              </div>
              <div style={{ width: '100%', height: 6, background: '#161B27', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ width: `${stats.sonar_selected_percentage}%`, height: '100%', background: '#0284C7', borderRadius: 3 }} />
              </div>
            </div>
          </div>
        </div>

        {/* Critical/High/Medium/Low Priority Distribution */}
        <div
          style={{
            background: '#111520',
            border: '1px solid #1E2435',
            borderRadius: 8,
            padding: '16px',
          }}
        >
          <div className="flex items-center justify-between" style={{ marginBottom: 12 }}>
            <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#8892A4' }}>
              TOPSIS Priority Distribution
            </span>
            <Sparkles style={{ width: 14, height: 14, color: '#F59E0B' }} />
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 8,
              marginBottom: 12,
            }}
          >
            {/* Critical */}
            <div style={{ background: '#161B27', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 6, padding: '8px', textAlign: 'center' }}>
              <span style={{ fontSize: '10px', color: '#EF4444', fontWeight: 700 }}>Critical</span>
              <div style={{ fontSize: '16px', fontWeight: 800, color: '#EEF2FF', fontFamily: 'monospace', marginTop: 2 }}>
                {pDist.Critical}
              </div>
            </div>

            {/* High */}
            <div style={{ background: '#161B27', border: '1px solid rgba(249,115,22,0.3)', borderRadius: 6, padding: '8px', textAlign: 'center' }}>
              <span style={{ fontSize: '10px', color: '#F97316', fontWeight: 700 }}>High</span>
              <div style={{ fontSize: '16px', fontWeight: 800, color: '#EEF2FF', fontFamily: 'monospace', marginTop: 2 }}>
                {pDist.High}
              </div>
            </div>

            {/* Medium */}
            <div style={{ background: '#161B27', border: '1px solid rgba(245,158,11,0.3)', borderRadius: 6, padding: '8px', textAlign: 'center' }}>
              <span style={{ fontSize: '10px', color: '#F59E0B', fontWeight: 700 }}>Medium</span>
              <div style={{ fontSize: '16px', fontWeight: 800, color: '#EEF2FF', fontFamily: 'monospace', marginTop: 2 }}>
                {pDist.Medium}
              </div>
            </div>

            {/* Low */}
            <div style={{ background: '#161B27', border: '1px solid rgba(34,197,94,0.3)', borderRadius: 6, padding: '8px', textAlign: 'center' }}>
              <span style={{ fontSize: '10px', color: '#22C55E', fontWeight: 700 }}>Low</span>
              <div style={{ fontSize: '16px', fontWeight: 800, color: '#EEF2FF', fontFamily: 'monospace', marginTop: 2 }}>
                {pDist.Low}
              </div>
            </div>
          </div>

          <div style={{ width: '100%', height: 8, display: 'flex', borderRadius: 4, overflow: 'hidden', background: '#1E2435' }}>
            <div style={{ width: `${(pDist.Critical / total) * 100}%`, background: '#EF4444' }} title={`Critical: ${pDist.Critical}`} />
            <div style={{ width: `${(pDist.High / total) * 100}%`, background: '#F97316' }} title={`High: ${pDist.High}`} />
            <div style={{ width: `${(pDist.Medium / total) * 100}%`, background: '#F59E0B' }} title={`Medium: ${pDist.Medium}`} />
            <div style={{ width: `${(pDist.Low / total) * 100}%`, background: '#22C55E' }} title={`Low: ${pDist.Low}`} />
          </div>
        </div>
      </div>

      {/* ── ROW 4: DATASET RECORDS INSPECTION TABLE ────────────────────────── */}
      {stats.records && stats.records.length > 0 && (
        <div
          style={{
            background: '#111520',
            border: '1px solid #1E2435',
            borderRadius: 8,
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              padding: '12px 16px',
              borderBottom: '1px solid #1E2435',
              background: '#0F1420',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div className="flex items-center gap-2">
              <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#8892A4' }}>
                SRMD Multimodal Ground Truth Records ({stats.records.length})
              </span>
            </div>
            <span style={{ fontSize: '10px', color: '#64748B', fontFamily: 'monospace' }}>
              Click any record to inspect full provenance
            </span>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
              <thead>
                <tr style={{ background: '#161B27', borderBottom: '1px solid #1E2435', color: '#8892A4', textTransform: 'uppercase', fontSize: '10px' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Event ID</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Class / Distress</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Data Source</th>
                  <th style={{ padding: '8px 12px', textAlign: 'right' }}>LiDAR (cm)</th>
                  <th style={{ padding: '8px 12px', textAlign: 'right' }}>Sonar (cm)</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Active Modality</th>
                  <th style={{ padding: '8px 12px', textAlign: 'right' }}>Z-Accel (g)</th>
                  <th style={{ padding: '8px 12px', textAlign: 'right' }}>Traffic PCU</th>
                  <th style={{ padding: '8px 12px', textAlign: 'right' }}>Hosp Dist</th>
                  <th style={{ padding: '8px 12px', textAlign: 'right' }}>TOPSIS</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center' }}>Priority</th>
                </tr>
              </thead>
              <tbody>
                {stats.records.map((r, i) => {
                  const isSel = r.id === selectedRecordId;
                  const isSonarMod = (r.processed_telemetry?.detection_modality || '').includes('sonar');
                  const pColor = r.priority_level === 'Critical' ? '#EF4444' :
                                 r.priority_level === 'High' ? '#F97316' :
                                 r.priority_level === 'Medium' ? '#F59E0B' : '#22C55E';
                  return (
                    <tr
                      key={r.id}
                      onClick={() => {
                        setSelectedRecordId(r.id);
                        if (onSelectRecord) onSelectRecord(r.id);
                      }}
                      style={{
                        background: isSel ? '#1E293B' : (i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)'),
                        borderBottom: '1px solid #1E2435',
                        cursor: 'pointer',
                        transition: 'background 0.15s',
                      }}
                      onMouseEnter={(e) => { if (!isSel) e.currentTarget.style.background = '#182030'; }}
                      onMouseLeave={(e) => { if (!isSel) e.currentTarget.style.background = i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)'; }}
                    >
                      <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontWeight: 700, color: '#EEF2FF' }}>
                        {r.id}
                      </td>
                      <td style={{ padding: '8px 12px' }}>
                        <span
                          style={{
                            fontWeight: 700,
                            fontFamily: 'monospace',
                            color: CLASS_INFO[r.damage_class || 'D40']?.color || '#EEF2FF',
                          }}
                        >
                          [{r.damage_class || 'D40'}]
                        </span>{' '}
                        <span style={{ color: '#CBD5E1' }}>{r.damage_label || 'Pothole'}</span>
                      </td>
                      <td style={{ padding: '8px 12px' }}>
                        <span
                          style={{
                            fontSize: '9px',
                            fontWeight: 700,
                            padding: '2px 6px',
                            borderRadius: 3,
                            background: '#1D4ED820',
                            border: '1px solid #3B82F650',
                            color: '#93C5FD',
                            fontFamily: 'monospace',
                          }}
                        >
                          {r.data_source || 'MULTIMODAL'}
                        </span>
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'monospace', color: '#E2E8F0' }}>
                        {r.lidar_depth_cm.toFixed(2)}
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'monospace', color: '#38BDF8' }}>
                        {r.sonar_depth_cm.toFixed(2)}
                      </td>
                      <td style={{ padding: '8px 12px' }}>
                        <span
                          style={{
                            fontSize: '10px',
                            color: isSonarMod ? '#38BDF8' : '#A78BFA',
                            fontFamily: 'monospace',
                          }}
                        >
                          {isSonarMod ? '⦿ Sonar' : '◉ LiDAR'}
                        </span>
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'monospace', color: r.z_accel_g > 1.5 ? '#EF4444' : '#E2E8F0', fontWeight: r.z_accel_g > 1.5 ? 700 : 400 }}>
                        {r.z_accel_g.toFixed(2)}g
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'monospace', color: '#E2E8F0' }}>
                        {r.traffic_pcu.toLocaleString()}
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'monospace', color: r.dist_hospital_km < 1.0 ? '#EF4444' : '#E2E8F0' }}>
                        {r.dist_hospital_km.toFixed(2)}km
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'monospace', fontWeight: 700, color: pColor }}>
                        {r.topsis_score.toFixed(4)}
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                        <span
                          style={{
                            fontSize: '9px',
                            fontWeight: 700,
                            padding: '2px 7px',
                            borderRadius: 3,
                            background: `${pColor}20`,
                            border: `1px solid ${pColor}50`,
                            color: pColor,
                            fontFamily: 'monospace',
                            textTransform: 'uppercase',
                          }}
                        >
                          {r.priority_level}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
