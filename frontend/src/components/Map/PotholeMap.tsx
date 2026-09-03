'use client';

import React, { useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap, Circle } from 'react-leaflet';
import L from 'leaflet';
import { TelemetryRecord, PriorityLevel } from '@/types/telemetry';
import {
  Ruler,
  Droplets,
  Hospital,
  Navigation,
  Zap,
  Car,
} from 'lucide-react';

interface PotholeMapProps {
  potholes: TelemetryRecord[];
  selectedPotholeId: string | null;
  onSelectPothole: (id: string) => void;
  filteredPriority: PriorityLevel | 'All';
}

const PRIORITY_CONFIG = {
  Critical: { color: '#EF4444', dot: '#FF6B6B', size: 36 },
  High:     { color: '#F97316', dot: '#FB923C', size: 30 },
  Medium:   { color: '#F59E0B', dot: '#FBBF24', size: 26 },
  Low:      { color: '#22C55E', dot: '#4ADE80', size: 22 },
};

// Controller: smooth pan on selection, initial bounds fit only once
function MapController({
  selectedRecord,
  potholes,
}: {
  selectedRecord: TelemetryRecord | null;
  potholes: TelemetryRecord[];
}) {
  const map = useMap();
  const initialFitDone = React.useRef(false);

  useEffect(() => {
    if (selectedRecord) {
      map.flyTo([selectedRecord.latitude, selectedRecord.longitude], 16, {
        animate: true,
        duration: 1.0,
      });
    } else if (potholes.length > 0 && !initialFitDone.current) {
      const bounds = L.latLngBounds(potholes.map((p) => [p.latitude, p.longitude]));
      map.fitBounds(bounds, { padding: [48, 48], maxZoom: 15 });
      initialFitDone.current = true;
    }
  }, [selectedRecord, map, potholes]);

  return null;
}

// Create professional DivIcon markers
function createMarkerIcon(priority: PriorityLevel, isSelected: boolean): L.DivIcon {
  const cfg = PRIORITY_CONFIG[priority] ?? PRIORITY_CONFIG.Low;
  const size = isSelected ? cfg.size + 4 : cfg.size;
  const borderColor = isSelected ? '#fff' : 'rgba(255,255,255,0.55)';
  const borderWidth = isSelected ? 2.5 : 1.5;

  const pulseHtml =
    priority === 'Critical'
      ? `<div class="radar-pulse-critical" style="
            position:absolute; inset:-6px; border-radius:50%;
            background:${cfg.color}; opacity:0.35;
          "></div>`
      : '';

  const selectedRing = isSelected
    ? `<div style="
          position:absolute; inset:-5px; border-radius:50%;
          border:2px solid #60A5FA; opacity:0.8;
          animation:ping 1s cubic-bezier(0,0,0.2,1) infinite;
        "></div>`
    : '';

  const html = `
    <div style="position:relative; width:${size}px; height:${size}px; display:flex; align-items:center; justify-content:center;">
      ${pulseHtml}
      ${selectedRing}
      <div style="
        width:${size}px; height:${size}px; border-radius:50%;
        background:${cfg.color};
        border:${borderWidth}px solid ${borderColor};
        display:flex; align-items:center; justify-content:center;
        box-shadow: 0 0 12px ${cfg.color}60, 0 2px 6px rgba(0,0,0,0.5);
        font-family:ui-monospace,monospace;
        font-weight:700;
        font-size:${size >= 30 ? 12 : 10}px;
        color:#fff;
        letter-spacing:-0.02em;
        transition:transform 0.15s ease;
      ">
        ${priority.charAt(0)}
      </div>
      <div style="
        position:absolute; bottom:-5px; left:50%; transform:translateX(-50%);
        width:0; height:0;
        border-left:5px solid transparent;
        border-right:5px solid transparent;
        border-top:6px solid ${cfg.color};
      "></div>
    </div>
  `;

  return L.divIcon({
    html,
    className: 'custom-pothole-marker',
    iconSize: [size, size + 6],
    iconAnchor: [size / 2, size + 5],
    popupAnchor: [0, -(size + 4)],
  });
}

export const PotholeMap: React.FC<PotholeMapProps> = ({
  potholes,
  selectedPotholeId,
  onSelectPothole,
  filteredPriority,
}) => {
  const displayedPotholes = useMemo(() => {
    if (filteredPriority === 'All') return potholes;
    return potholes.filter((p) => p.priority_level === filteredPriority);
  }, [potholes, filteredPriority]);

  const selectedRecord = useMemo(
    () => potholes.find((p) => p.id === selectedPotholeId) || null,
    [potholes, selectedPotholeId]
  );

  const defaultCenter: [number, number] =
    potholes.length > 0
      ? [potholes[0].latitude, potholes[0].longitude]
      : [19.078, 72.879];

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <MapContainer
        center={defaultCenter}
        zoom={14}
        scrollWheelZoom={true}
        style={{ width: '100%', height: '100%' }}
      >
        {/*
          Esri World Street Map — free, no API key required, no watermarks.
          The Esri attribution is required as per their terms.
        */}
        <TileLayer
          attribution='Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom, 2012'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
          maxZoom={18}
        />

        <MapController selectedRecord={selectedRecord} potholes={displayedPotholes} />

        {displayedPotholes.map((pothole) => {
          const cfg = PRIORITY_CONFIG[pothole.priority_level] ?? PRIORITY_CONFIG.Low;
          const isSelected = pothole.id === selectedPotholeId;
          const icon = createMarkerIcon(pothole.priority_level, isSelected);
          const pt = pothole.processed_telemetry;

          return (
            <React.Fragment key={pothole.id}>
              {/* Hazard zone circle */}
              <Circle
                center={[pothole.latitude, pothole.longitude]}
                radius={Math.max(3, Math.min(12, Math.sqrt((pothole.surface_area_sqm || 0.1) / Math.PI) * 6))}
                pathOptions={{
                  color: cfg.color,
                  fillColor: cfg.color,
                  fillOpacity: pothole.priority_level === 'Critical' ? 0.18 : 0.08,
                  weight: 1,
                  dashArray: pt?.is_submerged ? '5, 4' : undefined,
                }}
              />

              <Marker
                position={[pothole.latitude, pothole.longitude]}
                icon={icon}
                eventHandlers={{ click: () => onSelectPothole(pothole.id) }}
              >
                <Popup className="pothole-custom-popup" minWidth={260} maxWidth={300}>
                  <div style={{ padding: '14px 14px 10px' }}>
                    {/* Popup header */}
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        marginBottom: 10,
                        paddingBottom: 8,
                        borderBottom: '1px solid #1E2435',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span
                          style={{
                            fontFamily: 'ui-monospace, monospace',
                            fontWeight: 700,
                            fontSize: 14,
                            color: '#EEF2FF',
                          }}
                        >
                          {pothole.id}
                        </span>
                        <span
                          style={{
                            fontSize: 9,
                            fontWeight: 700,
                            letterSpacing: '0.08em',
                            textTransform: 'uppercase',
                            background: `${cfg.color}20`,
                            border: `1px solid ${cfg.color}50`,
                            color: cfg.color,
                            padding: '2px 7px',
                            borderRadius: 3,
                          }}
                        >
                          {pothole.priority_level}
                        </span>
                      </div>
                      <span
                        style={{
                          fontFamily: 'ui-monospace, monospace',
                          fontSize: 11,
                          fontWeight: 700,
                          color: cfg.color,
                        }}
                      >
                        {typeof pothole.topsis_score === 'number' && !isNaN(pothole.topsis_score)
                          ? pothole.topsis_score.toFixed(3)
                          : 'N/A'}
                      </span>
                    </div>

                    {/* Key metrics 2-column */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
                      <div
                        style={{
                          background: '#1C2235',
                          border: '1px solid #252D40',
                          borderRadius: 5,
                          padding: '7px 9px',
                        }}
                      >
                        <div style={{ fontSize: 9, color: '#4D5A72', marginBottom: 3 }}>Est. Depth</div>
                        <div
                          style={{
                            fontFamily: 'monospace',
                            fontWeight: 700,
                            fontSize: 14,
                            color: '#EEF2FF',
                          }}
                        >
                          {pt?.estimated_depth_cm} cm
                        </div>
                      </div>
                      <div
                        style={{
                          background: '#1C2235',
                          border: '1px solid #252D40',
                          borderRadius: 5,
                          padding: '7px 9px',
                        }}
                      >
                        <div style={{ fontSize: 9, color: '#4D5A72', marginBottom: 3 }}>Volume</div>
                        <div
                          style={{
                            fontFamily: 'monospace',
                            fontWeight: 700,
                            fontSize: 14,
                            color: '#60A5FA',
                          }}
                        >
                          {pt?.calculated_volume_liters} L
                        </div>
                      </div>
                    </div>

                    {/* Details list */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                      {[
                        {
                          icon: <Ruler style={{ width: 11, height: 11 }} />,
                          label: 'Modality',
                          value:
                            pt?.detection_modality === 'sonar_submerged_acoustic'
                              ? 'Acoustic Sonar'
                              : 'Optical LiDAR',
                        },
                        {
                          icon: <Hospital style={{ width: 11, height: 11 }} />,
                          label: 'Hospital',
                          value: `${pothole.dist_hospital_km} km`,
                          valueColor: pothole.dist_hospital_km < 1.0 ? '#EF4444' : undefined,
                        },
                        {
                          icon: <Navigation style={{ width: 11, height: 11 }} />,
                          label: 'Traffic',
                          value: `${pothole.traffic_pcu.toLocaleString()} PCU`,
                        },
                        {
                          icon: <Zap style={{ width: 11, height: 11 }} />,
                          label: 'Impact',
                          value: pt?.dynamic_impact_confirmed ? 'Confirmed >1.5g' : 'None',
                          valueColor: pt?.dynamic_impact_confirmed ? '#EF4444' : '#4D5A72',
                        },
                        {
                          icon: <Droplets style={{ width: 11, height: 11 }} />,
                          label: 'Surface',
                          value: `${pothole.surface_area_sqm} m² · ${pothole.rain_detected ? 'Wet' : 'Dry'}`,
                        },
                        {
                          icon: <Car style={{ width: 11, height: 11 }} />,
                          label: 'Vehicle',
                          value: pothole.vehicle_id,
                        },
                      ].map(({ icon, label, value, valueColor }) => (
                        <div
                          key={label}
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            fontSize: 11,
                          }}
                        >
                          <span style={{ display: 'flex', alignItems: 'center', gap: 5, color: '#4D5A72' }}>
                            {icon}
                            <span style={{ color: '#8892A4' }}>{label}</span>
                          </span>
                          <span
                            style={{
                              color: valueColor ?? '#EEF2FF',
                              fontWeight: 500,
                              fontFamily: 'ui-monospace, monospace',
                              fontSize: 11,
                            }}
                          >
                            {value}
                          </span>
                        </div>
                      ))}
                    </div>

                    {/* Footer */}
                    <div
                      style={{
                        marginTop: 10,
                        paddingTop: 7,
                        borderTop: '1px solid #1E2435',
                        display: 'flex',
                        justifyContent: 'flex-end',
                      }}
                    >
                      <span
                        style={{
                          fontFamily: 'monospace',
                          fontSize: 9,
                          color: '#4D5A72',
                        }}
                      >
                        {pothole.latitude.toFixed(4)}, {pothole.longitude.toFixed(4)}
                      </span>
                    </div>
                  </div>
                </Popup>
              </Marker>
            </React.Fragment>
          );
        })}
      </MapContainer>

      {/* ── Compact Legend (bottom-left) ─────────────────── */}
      <div
        style={{
          position: 'absolute',
          bottom: 12,
          left: 12,
          zIndex: 1000,
          background: 'rgba(11, 14, 20, 0.88)',
          border: '1px solid #1E2435',
          borderRadius: 5,
          padding: '6px 10px',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          backdropFilter: 'blur(8px)',
        }}
      >
        {(['Critical', 'High', 'Medium', 'Low'] as PriorityLevel[]).map((p) => (
          <div key={p} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: PRIORITY_CONFIG[p].color,
                flexShrink: 0,
              }}
            />
            <span style={{ fontSize: '10px', color: '#8892A4', whiteSpace: 'nowrap' }}>
              {p}
            </span>
          </div>
        ))}
      </div>

      {/* ── Hazard Counter (top-right) ────────────────────── */}
      <div
        style={{
          position: 'absolute',
          top: 10,
          right: 10,
          zIndex: 1000,
          background: 'rgba(11, 14, 20, 0.88)',
          border: '1px solid #1E2435',
          borderRadius: 4,
          padding: '4px 10px',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          backdropFilter: 'blur(8px)',
          fontSize: '11px',
          color: '#8892A4',
        }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: '#10B981',
            animation: 'pulse 2s infinite',
          }}
        />
        <span>
          <strong style={{ color: '#EEF2FF', fontFamily: 'monospace' }}>
            {displayedPotholes.length}
          </strong>
          {' '}of{' '}
          <strong style={{ color: '#EEF2FF', fontFamily: 'monospace' }}>
            {potholes.length}
          </strong>
          {' '}hazards
        </span>
      </div>
    </div>
  );
};
