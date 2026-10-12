'use client';

import React, { useEffect, useMemo, useState, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap, Circle } from 'react-leaflet';
import L from 'leaflet';
import { TelemetryRecord, PriorityLevel } from '@/types/telemetry';
import {
  Layers,
  Flame,
  Maximize2,
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

// Controller: smooth pan on selection, fit bounds whenever the incident set changes or on trigger
function MapController({
  selectedRecord,
  potholes,
  fitTrigger,
}: {
  selectedRecord: TelemetryRecord | null;
  potholes: TelemetryRecord[];
  fitTrigger: number;
}) {
  const map = useMap();
  const lastFittedSignature = useRef<string>('');

  useEffect(() => {
    const signature = potholes.map((p) => p.id).join(',');
    if (selectedRecord) {
      map.flyTo([selectedRecord.latitude, selectedRecord.longitude], 16, {
        animate: true,
        duration: 1.0,
      });
    } else if (potholes.length > 0 && lastFittedSignature.current !== signature) {
      const bounds = L.latLngBounds(potholes.map((p) => [p.latitude, p.longitude]));
      map.fitBounds(bounds, { padding: [48, 48], maxZoom: 15 });
      lastFittedSignature.current = signature;
    }
  }, [selectedRecord, map, potholes]);

  useEffect(() => {
    if (fitTrigger > 0 && potholes.length > 0) {
      const bounds = L.latLngBounds(potholes.map((p) => [p.latitude, p.longitude]));
      map.fitBounds(bounds, { padding: [48, 48], maxZoom: 15 });
    }
  }, [fitTrigger, map, potholes]);

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

const markerIconCache = new Map<string, L.DivIcon>();

function getMemoizedMarkerIcon(priority: PriorityLevel, isSelected: boolean): L.DivIcon {
  const key = `${priority}_${isSelected}`;
  let icon = markerIconCache.get(key);
  if (!icon) {
    icon = createMarkerIcon(priority, isSelected);
    markerIconCache.set(key, icon);
  }
  return icon;
}

function createPopupHtml(pothole: TelemetryRecord): string {
  const cfg = PRIORITY_CONFIG[pothole.priority_level] ?? PRIORITY_CONFIG.Low;
  const pt = pothole.processed_telemetry;
  const isPothole = (pothole.damage_class || 'D40') === 'D40' && pt?.calculated_volume_liters != null;
  const volumeDisplay = isPothole && pt?.calculated_volume_liters != null
    ? `${pt.calculated_volume_liters.toFixed(1)} L`
    : pt?.crack_severity_index != null
    ? pt.crack_severity_index.toFixed(3)
    : '—';
  const volumeLabel = isPothole ? 'Volume (L)' : 'Crack severity';
  const volumeColor = isPothole ? '#60A5FA' : '#A78BFA';
  const topsisStr = typeof pothole.topsis_score === 'number' && !isNaN(pothole.topsis_score)
    ? pothole.topsis_score.toFixed(3)
    : 'N/A';
  const modality = pt?.detection_modality === 'sonar_submerged_acoustic' ? 'Acoustic Sonar' : 'Optical LiDAR';
  const hospColor = pothole.dist_hospital_km < 1.0 ? 'color:#EF4444;' : 'color:#EEF2FF;';
  const impactColor = pt?.dynamic_impact_confirmed ? 'color:#EF4444;' : 'color:#4D5A72;';
  const impactVal = pt?.dynamic_impact_confirmed ? 'Confirmed >1.5g' : 'None';
  const clampedBadge = pothole.area_clamped
    ? '<span style="font-size:8px;font-weight:700;color:#FBBF24;background:rgba(234,179,8,0.18);border:1px solid rgba(234,179,8,0.4);padding:1px 3px;border-radius:2px;margin-left:4px;">CLAMPED</span>'
    : '';

  return `
    <div style="padding: 12px 14px 10px; font-family: system-ui, -apple-system, sans-serif; min-width: 250px;">
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #1E2435;">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span style="font-family: ui-monospace, monospace; font-weight: 700; font-size: 13px; color: #EEF2FF;">${pothole.id}</span>
          <span style="font-size: 9px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; background: ${cfg.color}20; border: 1px solid ${cfg.color}50; color: ${cfg.color}; padding: 2px 7px; border-radius: 3px;">${pothole.priority_level}</span>
        </div>
        <span style="font-family: ui-monospace, monospace; font-size: 11px; font-weight: 700; color: ${cfg.color};">${topsisStr}</span>
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 10px;">
        <div style="background: #1C2235; border: 1px solid #252D40; border-radius: 5px; padding: 6px 8px;">
          <div style="font-size: 9px; color: #4D5A72; margin-bottom: 2px;">Est. Depth</div>
          <div style="font-family: monospace; font-weight: 700; font-size: 13px; color: #EEF2FF;">${pt?.estimated_depth_cm ?? '—'} cm</div>
        </div>
        <div style="background: #1C2235; border: 1px solid #252D40; border-radius: 5px; padding: 6px 8px;">
          <div style="font-size: 9px; color: #4D5A72; margin-bottom: 2px;">${volumeLabel}</div>
          <div style="font-family: monospace; font-weight: 700; font-size: 13px; color: ${volumeColor};">${volumeDisplay}</div>
        </div>
      </div>
      <div style="display: flex; flex-direction: column; gap: 5px; font-size: 11px;">
        <div style="display: flex; justify-content: space-between; align-items: center;"><span style="color: #8892A4;">Modality</span><span style="color: #EEF2FF; font-family: monospace;">${modality}</span></div>
        <div style="display: flex; justify-content: space-between; align-items: center;"><span style="color: #8892A4;">Hospital</span><span style="${hospColor} font-family: monospace;">${pothole.dist_hospital_km} km</span></div>
        <div style="display: flex; justify-content: space-between; align-items: center;"><span style="color: #8892A4;">Traffic</span><span style="color: #EEF2FF; font-family: monospace;">${pothole.traffic_pcu.toLocaleString()} PCU</span></div>
        <div style="display: flex; justify-content: space-between; align-items: center;"><span style="${impactColor}; color: #8892A4;">Impact</span><span style="${impactColor} font-family: monospace;">${impactVal}</span></div>
        <div style="display: flex; justify-content: space-between; align-items: center;"><span style="color: #8892A4;">Surface</span><span style="color: #EEF2FF; font-family: monospace;">${pothole.surface_area_sqm} m² ${clampedBadge} · ${pothole.rain_detected ? 'Wet' : 'Dry'}</span></div>
        <div style="display: flex; justify-content: space-between; align-items: center;"><span style="color: #8892A4;">Vehicle</span><span style="color: #EEF2FF; font-family: monospace;">${pothole.vehicle_id}</span></div>
      </div>
      <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #1E2435; display: flex; justify-content: flex-end;">
        <span style="font-family: monospace; font-size: 9px; color: #4D5A72;">${pothole.latitude.toFixed(4)}, ${pothole.longitude.toFixed(4)}</span>
      </div>
    </div>
  `;
}

// Marker Clustering Layer
function ClusteredLayer({
  potholes,
  selectedPotholeId,
  onSelectPothole,
}: {
  potholes: TelemetryRecord[];
  selectedPotholeId: string | null;
  onSelectPothole: (id: string) => void;
}) {
  const map = useMap();

  useEffect(() => {
    if (typeof (L as any).markerClusterGroup !== 'function') return;

    const clusterGroup = (L as any).markerClusterGroup({
      showCoverageOnHover: false,
      zoomToBoundsOnClick: true,
      spiderfyOnMaxZoom: true,
      maxClusterRadius: 45,
      iconCreateFunction: (cluster: any) => {
        const markers = cluster.getAllChildMarkers();
        const count = cluster.getChildCount();
        const rankMap: Record<string, number> = { Critical: 4, High: 3, Medium: 2, Low: 1 };
        let highest = 'Low';
        markers.forEach((m: any) => {
          const p = m.options.priorityLevel || 'Low';
          if ((rankMap[p] || 0) > (rankMap[highest] || 0)) highest = p;
        });
        const cfg = PRIORITY_CONFIG[highest as PriorityLevel] ?? PRIORITY_CONFIG.Low;

        return L.divIcon({
          html: `<div style="
            width: 36px; height: 36px; border-radius: 50%;
            background: ${cfg.color}E6;
            border: 2px solid #FFFFFF;
            display: flex; align-items: center; justify-content: center;
            color: #FFFFFF; font-weight: 700; font-family: ui-monospace, monospace;
            font-size: 13px;
            box-shadow: 0 0 14px ${cfg.color}90, 0 3px 8px rgba(0,0,0,0.6);
          ">${count}</div>`,
          className: 'custom-cluster-marker',
          iconSize: [36, 36],
        });
      },
    });

    potholes.forEach((pothole) => {
      const isSelected = pothole.id === selectedPotholeId;
      const icon = getMemoizedMarkerIcon(pothole.priority_level, isSelected);
      const marker = L.marker([pothole.latitude, pothole.longitude], {
        icon,
        priorityLevel: pothole.priority_level,
      } as any);

      marker.on('click', () => {
        onSelectPothole(pothole.id);
      });

      marker.bindPopup(createPopupHtml(pothole), {
        className: 'pothole-custom-popup',
        minWidth: 260,
        maxWidth: 300,
      });

      clusterGroup.addLayer(marker);
    });

    map.addLayer(clusterGroup);

    return () => {
      map.removeLayer(clusterGroup);
    };
  }, [map, potholes, selectedPotholeId, onSelectPothole]);

  return null;
}

// Heatmap Layer using leaflet.heat
function HeatmapLayer({ potholes }: { potholes: TelemetryRecord[] }) {
  const map = useMap();

  useEffect(() => {
    if (typeof (L as any).heatLayer !== 'function') return;

    const points = potholes.map((p) => {
      const intensity =
        typeof p.topsis_score === 'number' && !isNaN(p.topsis_score)
          ? Math.max(0.3, Math.min(1.0, p.topsis_score))
          : 0.5;
      return [p.latitude, p.longitude, intensity] as [number, number, number];
    });

    const heat = (L as any).heatLayer(points, {
      radius: 38,
      blur: 24,
      maxZoom: 17,
      max: 1.0,
      gradient: {
        0.2: '#3B82F6',
        0.4: '#10B981',
        0.6: '#F59E0B',
        0.8: '#F97316',
        1.0: '#EF4444',
      },
    });

    map.addLayer(heat);

    return () => {
      map.removeLayer(heat);
    };
  }, [map, potholes]);

  return null;
}

export const PotholeMap: React.FC<PotholeMapProps> = ({
  potholes,
  selectedPotholeId,
  onSelectPothole,
  filteredPriority,
}) => {
  const [enableCluster, setEnableCluster] = useState<boolean>(true);
  const [enableHeatmap, setEnableHeatmap] = useState<boolean>(false);
  const [fitTrigger, setFitTrigger] = useState<number>(0);
  const [pluginsReady, setPluginsReady] = useState<boolean>(false);

  useEffect(() => {
    let mounted = true;
    Promise.all([
      import('leaflet.markercluster'),
      import('leaflet.heat'),
    ]).then(() => {
      if (mounted) setPluginsReady(true);
    }).catch((err) => {
      console.warn('Failed to load leaflet extensions:', err);
    });
    return () => {
      mounted = false;
    };
  }, []);

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
          Esri World Dark Gray Base with Esri World Street fallback
        */}
        <TileLayer
          attribution='Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
          maxZoom={16}
          eventHandlers={{
            tileerror: (error) => {
              const tile = (error as any).tile as HTMLImageElement;
              const coords = (error as any).coords;
              if (tile && coords && !tile.dataset.fallback) {
                tile.dataset.fallback = 'true';
                tile.src = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/${coords.z}/${coords.y}/${coords.x}`;
              }
            },
          }}
        />

        <MapController
          selectedRecord={selectedRecord}
          potholes={displayedPotholes}
          fitTrigger={fitTrigger}
        />

        {/* Heatmap Layer */}
        {enableHeatmap && pluginsReady && (
          <HeatmapLayer potholes={displayedPotholes} />
        )}

        {/* Marker Clustering Layer or Unclustered Markers */}
        {enableCluster && pluginsReady ? (
          <ClusteredLayer
            potholes={displayedPotholes}
            selectedPotholeId={selectedPotholeId}
            onSelectPothole={onSelectPothole}
          />
        ) : (
          displayedPotholes.map((pothole) => {
            const cfg = PRIORITY_CONFIG[pothole.priority_level] ?? PRIORITY_CONFIG.Low;
            const isSelected = pothole.id === selectedPotholeId;
            const icon = getMemoizedMarkerIcon(pothole.priority_level, isSelected);
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
                    <div
                      dangerouslySetInnerHTML={{
                        __html: createPopupHtml(pothole),
                      }}
                    />
                  </Popup>
                </Marker>
              </React.Fragment>
            );
          })
        )}
      </MapContainer>

      {/* ── Map Controls & Counter (top-right) ────────────────── */}
      <div
        style={{
          position: 'absolute',
          top: 10,
          right: 10,
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        {/* Toggle Clustering */}
        <button
          onClick={() => setEnableCluster((prev) => !prev)}
          style={{
            background: enableCluster ? 'rgba(59, 130, 246, 0.25)' : 'rgba(11, 14, 20, 0.88)',
            border: `1px solid ${enableCluster ? '#3B82F6' : '#1E2435'}`,
            borderRadius: 5,
            padding: '5px 9px',
            color: enableCluster ? '#60A5FA' : '#8892A4',
            fontSize: 11,
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
            transition: 'all 0.15s ease',
          }}
          title={enableCluster ? 'Disable marker clustering' : 'Enable marker clustering'}
        >
          <Layers style={{ width: 12, height: 12 }} />
          <span>Clusters</span>
        </button>

        {/* Toggle Heatmap */}
        <button
          onClick={() => setEnableHeatmap((prev) => !prev)}
          style={{
            background: enableHeatmap ? 'rgba(239, 68, 68, 0.25)' : 'rgba(11, 14, 20, 0.88)',
            border: `1px solid ${enableHeatmap ? '#EF4444' : '#1E2435'}`,
            borderRadius: 5,
            padding: '5px 9px',
            color: enableHeatmap ? '#F87171' : '#8892A4',
            fontSize: 11,
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
            transition: 'all 0.15s ease',
          }}
          title={enableHeatmap ? 'Disable heatmap overlay' : 'Enable heatmap overlay'}
        >
          <Flame style={{ width: 12, height: 12 }} />
          <span>Heatmap</span>
        </button>

        {/* Fit Bounds Button */}
        <button
          onClick={() => setFitTrigger((prev) => prev + 1)}
          style={{
            background: 'rgba(11, 14, 20, 0.88)',
            border: '1px solid #1E2435',
            borderRadius: 5,
            padding: '5px 9px',
            color: '#8892A4',
            fontSize: 11,
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
            transition: 'all 0.15s ease',
          }}
          title="Fit view to all visible incidents"
        >
          <Maximize2 style={{ width: 12, height: 12 }} />
          <span>Fit</span>
        </button>

        {/* Hazard Counter */}
        <div
          style={{
            background: 'rgba(11, 14, 20, 0.88)',
            border: '1px solid #1E2435',
            borderRadius: 5,
            padding: '5px 10px',
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
          </span>
        </div>
      </div>

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
    </div>
  );
};
