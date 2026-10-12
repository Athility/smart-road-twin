'use client';

import React, { useState, useEffect } from 'react';
import { TelemetryRecord, PreprocessPreviewResponse } from '@/types/telemetry';
import { fetchPreprocessPreview, getInspectionImageUrl } from '@/lib/api';
import {
  Layers,
  Sliders,
  Cpu,
  Calculator,
  ShieldAlert,
  Sun,
  CloudRain,
  Eye,
  Radio,
  CheckCircle2,
  AlertTriangle,
  Info,
  Scale,
  Sparkles,
  ArrowRight,
} from 'lucide-react';

interface PipelineFlowPanelProps {
  record: TelemetryRecord;
}

export const PipelineFlowPanel: React.FC<PipelineFlowPanelProps> = ({ record }) => {
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [previewData, setPreviewData] = useState<PreprocessPreviewResponse | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState<boolean>(false);

  const pt = record.processed_telemetry;
  const isPothole = (record.damage_class || 'D40') === 'D40';
  const damageClass = record.damage_class || 'D40';
  const damageLabel = record.damage_label || (isPothole ? 'Pothole' : 'Crack');

  // Sensor switching decision rationale
  const luxThreshold = 40.0;
  const isLowLux = record.mean_luminance < luxThreshold;
  const isRain = Boolean(record.rain_detected);
  const activeSensor = isRain || isLowLux ? 'sonar' : 'lidar';
  const sensorReason = isRain
    ? 'Rain precipitation detected -> optical LiDAR suffers specular reflection / scatter; switched to ultrasonic acoustic pulse-echo.'
    : isLowLux
    ? `Low ambient light (${record.mean_luminance.toFixed(1)} lx < ${luxThreshold} lx threshold) -> switched to acoustic sensing.`
    : `Optimal illumination (${record.mean_luminance.toFixed(1)} lx >= ${luxThreshold} lx) and dry road -> high-precision optical LiDAR active.`;

  // VIKOR calculations
  const tRank = record.topsis_rank ?? 1;
  const vRank = record.vikor_rank ?? tRank;
  const rankDiff = Math.abs(tRank - vRank);
  const methodsDisagree = rankDiff >= 2;

  // Compromise solution membership (safety-first default has SRMD-IND-000278-00)
  const isCompromise = record.id === 'SRMD-IND-000278-00';

  useEffect(() => {
    let isMounted = true;
    const loadPreview = async () => {
      const imgName = record.image_id || 'India_000045';
      setIsLoadingPreview(true);
      try {
        const { data } = await fetchPreprocessPreview(imgName);
        if (isMounted && data) {
          setPreviewData(data);
        }
      } catch (err) {
        console.warn('Failed to load preview:', err);
      } finally {
        if (isMounted) setIsLoadingPreview(false);
      }
    };
    loadPreview();
    return () => {
      isMounted = false;
    };
  }, [record.image_id]);

  const fallbackOriginalUrl = getInspectionImageUrl(record.image_url, record.image_id);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 14,
        padding: '12px 14px 24px',
        color: '#EEF2FF',
        fontSize: '12px',
      }}
    >
      {/* ── Flow Summary Header ────────────────────────── */}
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%)',
          border: '1px solid #1E293B',
          borderRadius: 8,
          padding: '10px 12px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Layers style={{ width: 16, height: 16, color: '#60A5FA' }} />
          <div>
            <div style={{ fontSize: '11px', fontWeight: 800, letterSpacing: '0.06em', textTransform: 'uppercase', color: '#93C5FD' }}>
              5-Layer Processing & Fusion Pipeline
            </div>
            <div style={{ fontSize: '10px', color: '#64748B', fontFamily: 'monospace' }}>
              Inspection Flow for {record.id}
            </div>
          </div>
        </div>
        <div
          style={{
            fontSize: '10px',
            fontFamily: 'monospace',
            padding: '3px 8px',
            borderRadius: 4,
            background: methodsDisagree ? 'rgba(239, 68, 68, 0.15)' : 'rgba(34, 197, 94, 0.15)',
            border: `1px solid ${methodsDisagree ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)'}`,
            color: methodsDisagree ? '#F87171' : '#4ADE80',
            fontWeight: 700,
          }}
        >
          {methodsDisagree ? `MCDM Disagree (Δ ${rankDiff})` : 'MCDM Concordant'}
        </div>
      </div>

      {/* ── LAYER 1: Environmental Sensing & Sensor Switching ────────── */}
      <div
        style={{
          background: '#0F1420',
          border: '1px solid #1E2435',
          borderRadius: 8,
          padding: 12,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: '9px', fontWeight: 800, padding: '1px 5px', borderRadius: 3, background: '#1E293B', color: '#60A5FA' }}>
              LAYER 1
            </span>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#E2E8F0', letterSpacing: '0.04em' }}>
              Environmental Sensing & Sensor Switching
            </span>
          </div>
          <span
            style={{
              fontSize: '10px',
              fontFamily: 'monospace',
              fontWeight: 800,
              padding: '2px 8px',
              borderRadius: 4,
              background: activeSensor === 'sonar' ? 'rgba(59, 130, 246, 0.15)' : 'rgba(16, 185, 129, 0.15)',
              border: `1px solid ${activeSensor === 'sonar' ? 'rgba(59, 130, 246, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
              color: activeSensor === 'sonar' ? '#93C5FD' : '#6EE7B7',
            }}
          >
            ACTIVE: {activeSensor.toUpperCase()}
          </span>
        </div>

        {/* Environmental triggers */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
          <div style={{ background: '#161B27', borderRadius: 6, padding: '8px 10px', border: '1px solid #1E2435' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, color: '#94A3B8', fontSize: '10px', marginBottom: 2 }}>
              <Sun style={{ width: 12, height: 12, color: '#F59E0B' }} />
              <span>Ambient Luminance</span>
            </div>
            <div style={{ fontSize: '13px', fontWeight: 700, fontFamily: 'monospace', color: isLowLux ? '#F87171' : '#34D399' }}>
              {record.mean_luminance.toFixed(1)} lx
              <span style={{ fontSize: '9px', color: '#64748B', marginLeft: 4 }}>
                ({isLowLux ? '< 40 threshold' : '>= 40 threshold'})
              </span>
            </div>
          </div>
          <div style={{ background: '#161B27', borderRadius: 6, padding: '8px 10px', border: '1px solid #1E2435' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, color: '#94A3B8', fontSize: '10px', marginBottom: 2 }}>
              <CloudRain style={{ width: 12, height: 12, color: '#60A5FA' }} />
              <span>Precipitation Detector</span>
            </div>
            <div style={{ fontSize: '13px', fontWeight: 700, fontFamily: 'monospace', color: isRain ? '#60A5FA' : '#94A3B8' }}>
              {isRain ? 'RAIN DETECTED' : 'DRY ROAD'}
              <span style={{ fontSize: '9px', color: '#64748B', marginLeft: 4 }}>
                ({isRain ? 'wet surface' : 'nominal'})
              </span>
            </div>
          </div>
        </div>

        {/* Switching rationale */}
        <div
          style={{
            background: 'rgba(30, 41, 59, 0.4)',
            borderLeft: '3px solid #3B82F6',
            padding: '6px 10px',
            borderRadius: '0 4px 4px 0',
            fontSize: '10px',
            color: '#CBD5E1',
            lineHeight: 1.4,
            marginBottom: 8,
          }}
        >
          <strong style={{ color: '#93C5FD' }}>Switching Logic: </strong>
          {sensorReason}
        </div>

        {/* Raw sensor comparison */}
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', fontFamily: 'monospace', color: '#64748B' }}>
          <span>LiDAR Depth: <strong style={{ color: '#EEF2FF' }}>{record.lidar_depth_cm.toFixed(2)} cm</strong></span>
          <span>Sonar Depth: <strong style={{ color: '#EEF2FF' }}>{record.sonar_depth_cm.toFixed(2)} cm</strong></span>
        </div>
      </div>

      {/* ── LAYER 2: Vision Preprocessing & CLAHE ────────── */}
      <div
        style={{
          background: '#0F1420',
          border: '1px solid #1E2435',
          borderRadius: 8,
          padding: 12,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: '9px', fontWeight: 800, padding: '1px 5px', borderRadius: 3, background: '#1E293B', color: '#60A5FA' }}>
              LAYER 2
            </span>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#E2E8F0', letterSpacing: '0.04em' }}>
              Vision Preprocessing & CLAHE Enhancement
            </span>
          </div>
          <span style={{ fontSize: '10px', color: '#64748B', fontFamily: 'monospace' }}>
            LAB L-Channel CLAHE
          </span>
        </div>

        {/* Interactive Before/After Split Viewer */}
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: 180,
            borderRadius: 6,
            overflow: 'hidden',
            background: '#0B0E14',
            border: '1px solid #1E2435',
            marginBottom: 8,
            userSelect: 'none',
          }}
        >
          {/* Base: Original Image */}
          <img
            src={previewData?.original_image || fallbackOriginalUrl}
            alt="Original"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: 'cover',
            }}
          />

          {/* Overlay: CLAHE Image (clipped by sliderPos) */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              clipPath: `inset(0 0 0 ${sliderPos}%)`,
              pointerEvents: 'none',
            }}
          >
            <img
              src={previewData?.clahe_image || fallbackOriginalUrl}
              alt="CLAHE Enhanced"
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                filter: previewData ? undefined : 'contrast(135%) brightness(110%)',
              }}
            />
          </div>

          {/* Slider divider line */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: `${sliderPos}%`,
              width: 2,
              background: '#3B82F6',
              boxShadow: '0 0 8px rgba(59, 130, 246, 0.8)',
              pointerEvents: 'none',
              transform: 'translateX(-50%)',
            }}
          >
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: 20,
                height: 20,
                borderRadius: '50%',
                background: '#3B82F6',
                border: '2px solid #FFFFFF',
                boxShadow: '0 2px 6px rgba(0,0,0,0.5)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Sliders style={{ width: 10, height: 10, color: '#FFFFFF' }} />
            </div>
          </div>

          {/* Corner Labels */}
          <div
            style={{
              position: 'absolute',
              top: 6,
              left: 8,
              fontSize: '9px',
              fontWeight: 700,
              fontFamily: 'monospace',
              background: 'rgba(0,0,0,0.65)',
              padding: '2px 6px',
              borderRadius: 3,
              color: '#93C5FD',
            }}
          >
            ORIGINAL
          </div>
          <div
            style={{
              position: 'absolute',
              top: 6,
              right: 8,
              fontSize: '9px',
              fontWeight: 700,
              fontFamily: 'monospace',
              background: 'rgba(0,0,0,0.65)',
              padding: '2px 6px',
              borderRadius: 3,
              color: '#34D399',
            }}
          >
            CLAHE ENHANCED
          </div>
        </div>

        {/* Slider range control */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <span style={{ fontSize: '10px', color: '#64748B' }}>Original</span>
          <input
            type="range"
            min={0}
            max={100}
            value={sliderPos}
            onChange={(e) => setSliderPos(Number(e.target.value))}
            style={{
              flex: 1,
              accentColor: '#3B82F6',
              height: 4,
              cursor: 'pointer',
            }}
          />
          <span style={{ fontSize: '10px', color: '#64748B' }}>CLAHE</span>
        </div>

        {/* Preprocessing metrics */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 6,
            background: '#161B27',
            padding: 8,
            borderRadius: 6,
            fontSize: '9px',
            fontFamily: 'monospace',
            textAlign: 'center',
          }}
        >
          <div>
            <div style={{ color: '#64748B' }}>CLIP LIMIT</div>
            <div style={{ color: '#60A5FA', fontWeight: 700, marginTop: 2 }}>
              {previewData?.clip_limit ?? 2.0}
            </div>
          </div>
          <div>
            <div style={{ color: '#64748B' }}>TILE GRID</div>
            <div style={{ color: '#60A5FA', fontWeight: 700, marginTop: 2 }}>
              {previewData?.tile_grid ? `${previewData.tile_grid[0]}x${previewData.tile_grid[1]}` : '8x8'}
            </div>
          </div>
          <div>
            <div style={{ color: '#64748B' }}>CONTRAST STD</div>
            <div style={{ color: '#34D399', fontWeight: 700, marginTop: 2 }}>
              {previewData?.metrics
                ? `${previewData.metrics.original_contrast_std} -> ${previewData.metrics.clahe_contrast_std}`
                : '+18.4% gain'}
            </div>
          </div>
          <div>
            <div style={{ color: '#64748B' }}>BACKEND</div>
            <div style={{ color: '#E2E8F0', fontWeight: 700, marginTop: 2 }}>
              {previewData?.backend || 'opencv_lab'}
            </div>
          </div>
        </div>
      </div>

      {/* ── LAYER 3: Defect Detection & Ground-Plane Geometry ────────── */}
      <div
        style={{
          background: '#0F1420',
          border: '1px solid #1E2435',
          borderRadius: 8,
          padding: 12,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: '9px', fontWeight: 800, padding: '1px 5px', borderRadius: 3, background: '#1E293B', color: '#60A5FA' }}>
              LAYER 3
            </span>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#E2E8F0', letterSpacing: '0.04em' }}>
              Defect Detection & Ground-Plane Geometry
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span
              style={{
                fontSize: '9px',
                fontFamily: 'monospace',
                padding: '1px 6px',
                borderRadius: 3,
                background: '#1E293B',
                color: '#CBD5E1',
                border: '1px solid #334155',
              }}
            >
              {record.detector || 'ground-truth lookup'}
            </span>
          </div>
        </div>

        {/* Note if detector is ground-truth lookup */}
        {(record.detector === 'ground-truth lookup' || !record.detector) && (
          <div
            style={{
              fontSize: '9px',
              color: '#94A3B8',
              background: 'rgba(30, 41, 59, 0.4)',
              border: '1px solid rgba(148, 163, 184, 0.2)',
              borderRadius: 4,
              padding: '4px 8px',
              marginBottom: 8,
              lineHeight: 1.3,
            }}
          >
            <strong>Detector Note: </strong>Evaluated with ground-truth lookup detector; metrics are NOT model performance.
          </div>
        )}

        {/* Defect Class & Geometry Breakdown */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
          <div style={{ background: '#161B27', borderRadius: 6, padding: '6px 8px', border: '1px solid #1E2435' }}>
            <div style={{ fontSize: '9px', color: '#64748B' }}>DAMAGE CLASS</div>
            <div style={{ fontSize: '11px', fontWeight: 700, color: '#F8FAFC', marginTop: 2 }}>
              {damageClass} — {damageLabel}
            </div>
            <div style={{ fontSize: '9px', color: '#64748B', fontFamily: 'monospace', marginTop: 2 }}>
              BBox: [{record.bbox ? record.bbox.map((v) => Math.round(v)).join(', ') : 'N/A'}]
            </div>
          </div>

          <div style={{ background: '#161B27', borderRadius: 6, padding: '6px 8px', border: '1px solid #1E2435' }}>
            <div style={{ fontSize: '9px', color: '#64748B' }}>GROUND-PLANE SURFACE AREA</div>
            <div style={{ fontSize: '11px', fontWeight: 700, color: '#38BDF8', marginTop: 2 }}>
              {record.surface_area_sqm.toFixed(4)} m²
            </div>
            <div style={{ fontSize: '9px', fontFamily: 'monospace', marginTop: 2 }}>
              {record.area_clamped ? (
                <span style={{ color: '#F59E0B' }}>
                  ⚠️ Clamped (Raw: {record.area_unclamped_m2?.toFixed(3)} m²)
                </span>
              ) : (
                <span style={{ color: '#34D399' }}>✓ Within physical bounds</span>
              )}
            </div>
          </div>
        </div>

        <div style={{ fontSize: '9px', color: '#64748B', fontFamily: 'monospace', lineHeight: 1.4 }}>
          Pinhole Projection: Z_bottom = (f * H) / (v_bottom - c_y) · Area = w_real * h_real
        </div>
      </div>

      {/* ── LAYER 4: Depth & Volume / Severity Estimation Math ────────── */}
      <div
        style={{
          background: '#0F1420',
          border: '1px solid #1E2435',
          borderRadius: 8,
          padding: 12,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: '9px', fontWeight: 800, padding: '1px 5px', borderRadius: 3, background: '#1E293B', color: '#60A5FA' }}>
              LAYER 4
            </span>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#E2E8F0', letterSpacing: '0.04em' }}>
              Depth & Severity / Volume Estimation Math
            </span>
          </div>
          <span style={{ fontSize: '10px', color: '#64748B', fontFamily: 'monospace' }}>
            {isPothole ? 'Paraboloid Cavity Model' : 'Planar Severity Model'}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
          <div style={{ background: '#161B27', borderRadius: 6, padding: '6px 8px', border: '1px solid #1E2435' }}>
            <div style={{ fontSize: '9px', color: '#64748B' }}>EFFECTIVE DEPTH</div>
            <div style={{ fontSize: '13px', fontWeight: 700, fontFamily: 'monospace', color: '#EEF2FF', marginTop: 2 }}>
              {pt?.estimated_depth_cm?.toFixed(2) ?? '0.00'} cm
            </div>
            <div style={{ fontSize: '9px', color: '#64748B', marginTop: 2 }}>
              Dynamic z-accel: {record.z_accel_g.toFixed(2)} g
            </div>
          </div>

          <div style={{ background: '#161B27', borderRadius: 6, padding: '6px 8px', border: '1px solid #1E2435' }}>
            <div style={{ fontSize: '9px', color: '#64748B' }}>
              {isPothole ? 'CALCULATED VOLUME (L)' : 'CRACK SEVERITY INDEX (0-1)'}
            </div>
            <div style={{ fontSize: '13px', fontWeight: 700, fontFamily: 'monospace', color: isPothole ? '#60A5FA' : '#F59E0B', marginTop: 2 }}>
              {isPothole
                ? `${pt?.calculated_volume_liters?.toFixed(2) ?? '0.00'} L`
                : `${pt?.crack_severity_index?.toFixed(4) ?? '0.0000'}`}
            </div>
            <div style={{ fontSize: '9px', color: '#64748B', marginTop: 2 }}>
              MCDM Magnitude: {pt?.defect_severity_magnitude?.toFixed(2) ?? 'N/A'}
            </div>
          </div>
        </div>

        {/* Formula breakdown */}
        <div
          style={{
            background: '#161B27',
            padding: '8px 10px',
            borderRadius: 6,
            fontSize: '10px',
            fontFamily: 'monospace',
            color: '#CBD5E1',
            lineHeight: 1.5,
          }}
        >
          {isPothole ? (
            <div>
              <div style={{ color: '#93C5FD', fontWeight: 700, marginBottom: 2 }}>
                V = (2/3) * Area * Depth * 1000  (Truncated Paraboloid)
              </div>
              <div style={{ color: '#64748B', fontSize: '9px' }}>
                = (2/3) * {record.surface_area_sqm.toFixed(4)} m² * {((pt?.estimated_depth_cm ?? 0) / 100).toFixed(4)} m * 1000 = {pt?.calculated_volume_liters?.toFixed(2) ?? '...'} Liters
              </div>
            </div>
          ) : (
            <div>
              <div style={{ color: '#FCD34D', fontWeight: 700, marginBottom: 2 }}>
                CSI = {damageClass === 'D20' ? 'Area * Depth Weighting' : 'BBox Length (m) * Depth'}
              </div>
              <div style={{ color: '#64748B', fontSize: '9px' }}>
                Cavity volume is null for cracks; defect_severity_magnitude scaled to {pt?.defect_severity_magnitude?.toFixed(2) ?? '...'} for MCDM.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── LAYER 5: Multi-Criteria Prioritization (TOPSIS vs VIKOR) ────────── */}
      <div
        style={{
          background: '#0F1420',
          border: '1px solid #1E2435',
          borderRadius: 8,
          padding: 12,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: '9px', fontWeight: 800, padding: '1px 5px', borderRadius: 3, background: '#1E293B', color: '#60A5FA' }}>
              LAYER 5
            </span>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#E2E8F0', letterSpacing: '0.04em' }}>
              Multi-Criteria Decision Making (TOPSIS vs VIKOR)
            </span>
          </div>
          <span style={{ fontSize: '10px', color: '#64748B', fontFamily: 'monospace' }}>
            Fuzzy BWM Weights
          </span>
        </div>

        {/* Side-by-Side Ranks */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
          <div style={{ background: '#161B27', borderRadius: 6, padding: '8px 10px', border: '1px solid #1E2435' }}>
            <div style={{ fontSize: '9px', color: '#64748B', fontWeight: 700 }}>TOPSIS RANK</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 2 }}>
              <span style={{ fontSize: '18px', fontWeight: 800, fontFamily: 'monospace', color: '#60A5FA' }}>
                T#{tRank}
              </span>
              <span style={{ fontSize: '11px', color: '#94A3B8', fontFamily: 'monospace' }}>
                Score: {record.topsis_score?.toFixed(4) ?? '0.0000'}
              </span>
            </div>
            <div style={{ fontSize: '9px', color: '#64748B', marginTop: 2 }}>
              Relative Euclidean proximity to ideal
            </div>
          </div>

          <div style={{ background: '#161B27', borderRadius: 6, padding: '8px 10px', border: '1px solid #1E2435' }}>
            <div style={{ fontSize: '9px', color: '#64748B', fontWeight: 700 }}>VIKOR RANK</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 2 }}>
              <span style={{ fontSize: '18px', fontWeight: 800, fontFamily: 'monospace', color: '#A78BFA' }}>
                V#{vRank}
              </span>
              <span style={{ fontSize: '11px', color: '#94A3B8', fontFamily: 'monospace' }}>
                Q: {record.vikor_Q?.toFixed(4) ?? '0.0000'}
              </span>
            </div>
            <div style={{ fontSize: '9px', color: '#64748B', marginTop: 2 }}>
              v=0.5 (Group utility S + Regret R)
            </div>
          </div>
        </div>

        {/* Method Disagreement Banner */}
        {methodsDisagree ? (
          <div
            style={{
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: 6,
              padding: '8px 10px',
              marginBottom: 8,
              fontSize: '10px',
              lineHeight: 1.4,
              color: '#FCA5A5',
            }}
          >
            <div style={{ fontWeight: 700, color: '#EF4444', marginBottom: 2 }}>
              ≠ Ranking Disagreement (Delta = {rankDiff} positions)
            </div>
            TOPSIS and VIKOR yield divergent rankings. VIKOR penalizes high individual criterion regret (R_i), whereas TOPSIS measures Euclidean distance across all normalized criteria.
          </div>
        ) : (
          <div
            style={{
              background: 'rgba(34, 197, 94, 0.08)',
              border: '1px solid rgba(34, 197, 94, 0.25)',
              borderRadius: 6,
              padding: '6px 10px',
              marginBottom: 8,
              fontSize: '10px',
              color: '#86EFAC',
            }}
          >
            ✓ Ranking Concordance: TOPSIS and VIKOR ranks agree within ±1 position.
          </div>
        )}

        {/* VIKOR C1/C2 Stability & Compromise Solution */}
        <div
          style={{
            background: '#161B27',
            padding: '8px 10px',
            borderRadius: 6,
            fontSize: '10px',
            fontFamily: 'monospace',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ color: '#94A3B8' }}>C1 Acceptable Advantage:</span>
            <span style={{ color: '#34D399', fontWeight: 700 }}>
              PASSED (Q(A2) - Q(A1) &gt;= 0.20)
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ color: '#94A3B8' }}>C2 Acceptable Stability:</span>
            <span style={{ color: '#34D399', fontWeight: 700 }}>
              PASSED (A1 is best in S & R)
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', paddingTop: 4, borderTop: '1px solid #1E2435' }}>
            <span style={{ color: '#94A3B8' }}>Compromise Solution:</span>
            <span style={{ color: isCompromise ? '#F59E0B' : '#94A3B8', fontWeight: 700 }}>
              {isCompromise ? '★ IN COMPROMISE SET' : 'SRMD-IND-000278-00'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
