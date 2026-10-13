import { TelemetryResponse, PriorityLevel, TelemetryRecord, DatasetExplorerStats, DetectorType, PrioritizationResponse, WeightProfile, PreprocessPreviewResponse } from '@/types/telemetry';

export const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1/telemetry';

export function getBackendOrigin(): string {
  const raw = process.env.NEXT_PUBLIC_API_URL || BACKEND_URL;
  return raw.replace(/\/api\/v1\/.*$/, '').replace(/\/+$/, '');
}

export function getInspectionImageUrl(imageUrl?: string, imageId?: string): string {
  const origin = getBackendOrigin();
  if (imageUrl) {
    return imageUrl.startsWith('http') ? imageUrl : `${origin}${imageUrl.startsWith('/') ? '' : '/'}${imageUrl}`;
  }
  return `${origin}/api/v1/images/${imageId || 'India_000045'}.jpg`;
}

import fallbackRecordsRaw from '@/data/fallback_records.json';

export const FALLBACK_RECORDS: TelemetryRecord[] = fallbackRecordsRaw as unknown as TelemetryRecord[];


export function computeExplorerStatsFromRecords(
  records: TelemetryRecord[],
  detector: DetectorType = 'ground-truth lookup'
): DatasetExplorerStats {
  const N = Math.max(1, records.length);
  const priority_distribution: Record<PriorityLevel, number> = {
    Critical: 0,
    High: 0,
    Medium: 0,
    Low: 0,
  };
  for (const r of records) {
    const p = r.priority_level || 'Low';
    priority_distribution[p] = (priority_distribution[p] || 0) + 1;
  }

  return {
    detector,
    total_records: records.length,
    d00_count: records.filter((r) => r.damage_class === 'D00').length,
    d10_count: records.filter((r) => r.damage_class === 'D10').length,
    d20_count: records.filter((r) => r.damage_class === 'D20').length,
    d40_count: records.filter((r) => (r.damage_class || 'D40') === 'D40').length,
    avg_lidar_depth: Number((records.reduce((s, r) => s + r.lidar_depth_cm, 0) / N).toFixed(2)),
    avg_sonar_depth: Number((records.reduce((s, r) => s + r.sonar_depth_cm, 0) / N).toFixed(2)),
    avg_acceleration: Number((records.reduce((s, r) => s + r.z_accel_g, 0) / N).toFixed(2)),
    avg_traffic_pcu: Number((records.reduce((s, r) => s + r.traffic_pcu, 0) / N).toFixed(1)),
    avg_hospital_distance: Number((records.reduce((s, r) => s + r.dist_hospital_km, 0) / N).toFixed(2)),
    rain_percentage: Number(((records.filter((r) => r.rain_detected).length / N) * 100).toFixed(1)),
    lidar_selected_percentage: Number(
      ((records.filter((r) => (r.processed_telemetry?.detection_modality || '').includes('lidar')).length / N) * 100).toFixed(1)
    ),
    sonar_selected_percentage: Number(
      ((records.filter((r) => (r.processed_telemetry?.detection_modality || '').includes('sonar')).length / N) * 100).toFixed(1)
    ),
    priority_distribution,
    records,
  };
}

export const FALLBACK_EXPLORER_STATS: DatasetExplorerStats = computeExplorerStatsFromRecords(
  FALLBACK_RECORDS,
  'ground-truth lookup'
);

export const FALLBACK_DATA: TelemetryResponse = {
  status: 'fallback',
  detector: 'ground-truth lookup',
  total: FALLBACK_RECORDS.length,
  summary: {
    detector: 'ground-truth lookup',
    total_defects: FALLBACK_RECORDS.length,
    total_volume_liters: 169.97,
    average_volume_liters: 84.99,
    weather_state: 'Clear (Dry Road)',
    is_raining: false,
    submerged_defects: 2,
    average_luminance: 51.3,
    priority_counts: {
      Critical: 0,
      High: 2,
      Medium: 3,
      Low: 1
    },
    latest_update: '2026-09-24T12:30:01.802066+00:00'
  },
  data: FALLBACK_RECORDS
};

export async function fetchTelemetry(): Promise<{ data: TelemetryResponse; isLive: boolean }> {
  try {
    const res = await fetch(BACKEND_URL, {
      cache: 'no-store',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }

    const json = await res.json();
    return { data: json, isLive: true };
  } catch (error) {
    console.warn('Backend unavailable, using simulated digital twin telemetry:', error);
    return { data: FALLBACK_DATA, isLive: false };
  }
}

export async function fetchDatasetExplorerStats(): Promise<{ stats: DatasetExplorerStats; isLive: boolean }> {
  try {
    const url = BACKEND_URL.replace(/\/telemetry.*$/, '/dataset/explorer');
    const res = await fetch(url, {
      cache: 'no-store',
      headers: {
        'Accept': 'application/json',
      },
    });
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    const json = await res.json();
    return { stats: json, isLive: true };
  } catch (error) {
    console.warn('Backend unavailable, using simulated SRMD dataset stats:', error);
    return { stats: FALLBACK_EXPLORER_STATS, isLive: false };
  }
}

export async function fetchPrioritization(
  profile: WeightProfile = 'safety-first'
): Promise<{ data: PrioritizationResponse | null; isLive: boolean }> {
  try {
    const origin = getBackendOrigin();
    const url = `${origin}/api/v1/prioritization?method=both&profile=${encodeURIComponent(profile)}`;
    const res = await fetch(url, {
      cache: 'no-store',
      headers: {
        Accept: 'application/json',
      },
    });
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    const json = await res.json();
    return { data: json, isLive: true };
  } catch (error) {
    console.warn('Backend prioritization API unavailable:', error);
    return { data: null, isLive: false };
  }
}

export async function fetchPreprocessPreview(
  imageName: string,
  clipLimit = 2.0,
  tileGridSize = 8
): Promise<{ data: PreprocessPreviewResponse | null; isLive: boolean }> {
  try {
    const origin = getBackendOrigin();
    const url = `${origin}/api/v1/preprocess/preview/${encodeURIComponent(imageName)}?clip_limit=${clipLimit}&tile_grid_size=${tileGridSize}`;
    const res = await fetch(url, {
      cache: 'no-store',
      headers: {
        Accept: 'application/json',
      },
    });
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    const json = await res.json();
    return { data: json, isLive: true };
  } catch (error) {
    console.warn('Preprocess preview API unavailable:', error);
    return { data: null, isLive: false };
  }
}

export function getPriorityColor(level: PriorityLevel): {
  hex: string;
  bgClass: string;
  borderClass: string;
  textClass: string;
  badgeClass: string;
  glowClass: string;
} {
  switch (level) {
    case 'Critical':
      return {
        hex: '#EF4444',
        bgClass: 'bg-red-500/10',
        borderClass: 'border-red-500/30',
        textClass: 'text-red-500',
        badgeClass: 'bg-red-500 text-white shadow-red-500/50',
        glowClass: 'shadow-[0_0_15px_rgba(239,68,68,0.5)]'
      };
    case 'High':
      return {
        hex: '#F97316',
        bgClass: 'bg-orange-500/10',
        borderClass: 'border-orange-500/30',
        textClass: 'text-orange-500',
        badgeClass: 'bg-orange-500 text-white shadow-orange-500/50',
        glowClass: 'shadow-[0_0_15px_rgba(249,115,22,0.5)]'
      };
    case 'Medium':
      return {
        hex: '#EAB308',
        bgClass: 'bg-yellow-500/10',
        borderClass: 'border-yellow-500/30',
        textClass: 'text-yellow-500',
        badgeClass: 'bg-yellow-500 text-black shadow-yellow-500/50',
        glowClass: 'shadow-[0_0_15px_rgba(234,179,8,0.5)]'
      };
    case 'Low':
    default:
      return {
        hex: '#22C55E',
        bgClass: 'bg-emerald-500/10',
        borderClass: 'border-emerald-500/30',
        textClass: 'text-emerald-500',
        badgeClass: 'bg-emerald-500 text-white shadow-emerald-500/50',
        glowClass: 'shadow-[0_0_15px_rgba(34,197,94,0.5)]'
      };
  }
}
