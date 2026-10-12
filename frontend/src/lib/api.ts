import { TelemetryResponse, PriorityLevel, TelemetryRecord, DatasetExplorerStats, DetectorType, PrioritizationResponse, WeightProfile } from '@/types/telemetry';

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

export const FALLBACK_RECORDS: TelemetryRecord[] = [
  {
    id: 'SRMD-IND-000045-00',
    timestamp: '2026-09-24T12:30:01.800067+00:00',
    vehicle_id: 'INSPECTION_FLEET_01',
    latitude: 19.05939,
    longitude: 72.8488,
    rain_detected: true,
    mean_luminance: 22.1,
    lidar_depth_cm: 45.94,
    sonar_depth_cm: 45.95,
    z_accel_g: 2.53,
    surface_area_sqm: 1.812,
    traffic_pcu: 27151,
    dist_hospital_km: 2.4,
    speed_kmh: 41.7,
    data_source: 'MULTIMODAL',
    detector: 'ground-truth lookup',
    damage_class: 'D40',
    damage_label: 'Pothole',
    confidence: 1.0,
    image_id: 'India_000045',
    image_url: '/api/v1/images/India_000045.jpg',
    bbox: [180.0, 260.0, 420.0, 480.0],
    road_name: 'Western Express Highway, Bandra-Kalanagar Junction',
    road_class: 'HIGH_TRAFFIC_CORRIDOR',
    processed_telemetry: {
      detection_modality: 'sonar_submerged_acoustic',
      is_submerged: true,
      estimated_depth_cm: 15.95,
      calculated_volume_liters: 144.51,
      crack_severity_index: null,
      defect_severity_magnitude: 144.51,
      dynamic_impact_confirmed: true
    },
    topsis_score: 0.6745,
    priority_level: 'High',
    topsis_rank: 1,
    vikor_rank: 4,
  },
  {
    id: 'SRMD-IND-000102-00',
    timestamp: '2026-09-24T12:30:01.800067+00:00',
    vehicle_id: 'INSPECTION_FLEET_01',
    latitude: 19.07801,
    longitude: 72.84119,
    rain_detected: false,
    mean_luminance: 20.1,
    lidar_depth_cm: 34.13,
    sonar_depth_cm: 34.27,
    z_accel_g: 1.16,
    surface_area_sqm: 3.587,
    traffic_pcu: 16381,
    dist_hospital_km: 1.31,
    speed_kmh: 36.5,
    data_source: 'MULTIMODAL',
    detector: 'ground-truth lookup',
    damage_class: 'D20',
    damage_label: 'Alligator Crack',
    confidence: 1.0,
    image_id: 'India_000102',
    image_url: '/api/v1/images/India_000102.jpg',
    bbox: [120.0, 300.0, 580.0, 520.0],
    road_name: 'Swami Vivekananda (SV) Road, Santacruz West',
    road_class: 'ARTERIAL',
    processed_telemetry: {
      detection_modality: 'sonar_submerged_acoustic',
      is_submerged: false,
      estimated_depth_cm: 4.27,
      calculated_volume_liters: null,
      crack_severity_index: 0.8318,
      defect_severity_magnitude: 12.48,
      dynamic_impact_confirmed: false
    },
    topsis_score: 0.5109,
    priority_level: 'High',
    topsis_rank: 3,
    vikor_rank: 3,
  },
  {
    id: 'SRMD-IND-000155-00',
    timestamp: '2026-09-24T12:30:01.801067+00:00',
    vehicle_id: 'INSPECTION_FLEET_01',
    latitude: 19.06179,
    longitude: 72.83285,
    rain_detected: true,
    mean_luminance: 24.0,
    lidar_depth_cm: 30.79,
    sonar_depth_cm: 30.92,
    z_accel_g: 1.03,
    surface_area_sqm: 0.918,
    traffic_pcu: 11391,
    dist_hospital_km: 0.88,
    speed_kmh: 25.0,
    data_source: 'MULTIMODAL',
    detector: 'ground-truth lookup',
    damage_class: 'D00',
    damage_label: 'Longitudinal Crack',
    confidence: 1.0,
    image_id: 'India_000155',
    image_url: '/api/v1/images/India_000155.jpg',
    bbox: [310.0, 150.0, 370.0, 640.0],
    road_name: 'Linking Road Commercial Sub-Arterial, Bandra West',
    road_class: 'COLLECTOR',
    processed_telemetry: {
      detection_modality: 'sonar_submerged_acoustic',
      is_submerged: true,
      estimated_depth_cm: 0.92,
      calculated_volume_liters: null,
      crack_severity_index: 0.3193,
      defect_severity_magnitude: 4.79,
      dynamic_impact_confirmed: false
    },
    topsis_score: 0.2667,
    priority_level: 'Medium',
    topsis_rank: 5,
    vikor_rank: 6,
  },
  {
    id: 'SRMD-IND-000210-00',
    timestamp: '2026-09-24T12:30:01.801067+00:00',
    vehicle_id: 'INSPECTION_FLEET_01',
    latitude: 19.06505,
    longitude: 72.82503,
    rain_detected: false,
    mean_luminance: 76.9,
    lidar_depth_cm: 30.99,
    sonar_depth_cm: 30.8,
    z_accel_g: 1.02,
    surface_area_sqm: 1.221,
    traffic_pcu: 5596,
    dist_hospital_km: 0.96,
    speed_kmh: 20.5,
    data_source: 'MULTIMODAL',
    detector: 'ground-truth lookup',
    damage_class: 'D10',
    damage_label: 'Transverse Crack',
    confidence: 1.0,
    image_id: 'India_000210',
    image_url: '/api/v1/images/India_000210.jpg',
    bbox: [80.0, 380.0, 620.0, 440.0],
    road_name: 'Pali Hill Residential Access Corridor, Bandra West',
    road_class: 'LOCAL',
    processed_telemetry: {
      detection_modality: 'optical_lidar',
      is_submerged: false,
      estimated_depth_cm: 0.99,
      calculated_volume_liters: null,
      crack_severity_index: 0.4151,
      defect_severity_magnitude: 6.23,
      dynamic_impact_confirmed: false
    },
    topsis_score: 0.2429,
    priority_level: 'Low',
    topsis_rank: 4,
    vikor_rank: 5,
  },
  {
    id: 'SRMD-IND-000278-00',
    timestamp: '2026-09-24T12:30:01.802066+00:00',
    vehicle_id: 'INSPECTION_FLEET_01',
    latitude: 19.13615,
    longitude: 72.86003,
    rain_detected: false,
    mean_luminance: 85.0,
    lidar_depth_cm: 36.09,
    sonar_depth_cm: 35.96,
    z_accel_g: 1.41,
    surface_area_sqm: 0.85,
    traffic_pcu: 29131,
    dist_hospital_km: 0.27,
    speed_kmh: 38.3,
    data_source: 'MULTIMODAL',
    detector: 'ground-truth lookup',
    damage_class: 'D40',
    damage_label: 'Pothole',
    confidence: 1.0,
    image_id: 'India_000278',
    image_url: '/api/v1/images/India_000278.jpg',
    bbox: [220.0, 310.0, 390.0, 450.0],
    road_name: 'Western Express Highway, Jogeshwari Link Flyover',
    road_class: 'HIGH_TRAFFIC_CORRIDOR',
    processed_telemetry: {
      detection_modality: 'optical_lidar',
      is_submerged: false,
      estimated_depth_cm: 6.09,
      calculated_volume_liters: 25.46,
      crack_severity_index: null,
      defect_severity_magnitude: 25.46,
      dynamic_impact_confirmed: false
    },
    topsis_score: 0.4532,
    priority_level: 'Medium',
    topsis_rank: 2,
    vikor_rank: 1,
  },
  {
    id: 'SRMD-IND-000278-01',
    timestamp: '2026-09-24T12:30:01.802066+00:00',
    vehicle_id: 'INSPECTION_FLEET_01',
    latitude: 19.12012,
    longitude: 72.84614,
    rain_detected: false,
    mean_luminance: 79.6,
    lidar_depth_cm: 34.14,
    sonar_depth_cm: 33.92,
    z_accel_g: 1.11,
    surface_area_sqm: 2.897,
    traffic_pcu: 19406,
    dist_hospital_km: 2.03,
    speed_kmh: 33.0,
    data_source: 'MULTIMODAL',
    detector: 'ground-truth lookup',
    damage_class: 'D20',
    damage_label: 'Alligator Crack',
    confidence: 1.0,
    image_id: 'India_000278',
    image_url: '/api/v1/images/India_000278.jpg',
    bbox: [150.0, 250.0, 520.0, 480.0],
    road_name: 'Swami Vivekananda (SV) Road, Andheri West',
    road_class: 'ARTERIAL',
    processed_telemetry: {
      detection_modality: 'optical_lidar',
      is_submerged: false,
      estimated_depth_cm: 4.14,
      calculated_volume_liters: null,
      crack_severity_index: 0.662,
      defect_severity_magnitude: 9.93,
      dynamic_impact_confirmed: false
    },
    topsis_score: 0.3857,
    priority_level: 'Medium',
    topsis_rank: 6,
    vikor_rank: 2,
  }
];

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
