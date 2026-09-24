import { TelemetryResponse, PriorityLevel, TelemetryRecord, DatasetExplorerStats } from '@/types/telemetry';

export const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1/telemetry';

export const FALLBACK_EXPLORER_STATS: DatasetExplorerStats = {
  total_records: 6,
  d00_count: 1,
  d10_count: 1,
  d20_count: 2,
  d40_count: 2,
  avg_lidar_depth: 35.35,
  avg_sonar_depth: 35.30,
  avg_acceleration: 1.38,
  avg_traffic_pcu: 18176.0,
  avg_hospital_distance: 1.31,
  rain_percentage: 33.3,
  lidar_selected_percentage: 50.0,
  sonar_selected_percentage: 50.0,
  priority_distribution: {
    Critical: 0,
    High: 2,
    Medium: 3,
    Low: 1
  }
};

export const FALLBACK_DATA: TelemetryResponse = {
  status: 'fallback',
  total: 4,
  summary: {
    total_defects: 4,
    total_volume_liters: 224.24,
    average_volume_liters: 56.06,
    weather_state: 'Rain Detected (Wet / Submerged Road)',
    is_raining: true,
    submerged_defects: 1,
    average_luminance: 49.8,
    priority_counts: {
      Critical: 0,
      High: 2,
      Medium: 1,
      Low: 1
    },
    latest_update: '2026-09-24T11:52:11.225Z'
  },
  data: [
    {
      id: 'SRMD-IND-000045-00',
      timestamp: '2026-09-24T11:52:11.225Z',
      vehicle_id: 'INSPECTION_FLEET_01',
      latitude: 19.05939,
      longitude: 72.8488,
      rain_detected: true,
      mean_luminance: 22.1,
      lidar_depth_cm: 45.94,
      sonar_depth_cm: 45.95,
      z_accel_g: 2.53,
      surface_area_sqm: 1.81,
      traffic_pcu: 27151,
      dist_hospital_km: 2.4,
      speed_kmh: 41.7,
      data_source: 'MULTIMODAL',
      damage_class: 'D40',
      damage_label: 'Pothole',
      confidence: 0.96,
      image_id: 'India_000045',
      image_url: '/api/v1/images/India_000045.jpg',
      bbox: [180.0, 260.0, 420.0, 480.0],
      road_name: 'Western Express Highway, Bandra-Kalanagar Junction',
      road_class: 'HIGH_TRAFFIC_CORRIDOR',
      processed_telemetry: {
        detection_modality: 'sonar_submerged_acoustic',
        is_submerged: true,
        estimated_depth_cm: 15.95,
        calculated_volume_liters: 144.42,
        dynamic_impact_confirmed: true
      },
      topsis_score: 0.6758,
      priority_level: 'High'
    },
    {
      id: 'SRMD-IND-000102-00',
      timestamp: '2026-09-24T11:52:11.226Z',
      vehicle_id: 'INSPECTION_FLEET_01',
      latitude: 19.07801,
      longitude: 72.84119,
      rain_detected: false,
      mean_luminance: 20.1,
      lidar_depth_cm: 34.13,
      sonar_depth_cm: 34.27,
      z_accel_g: 1.16,
      surface_area_sqm: 3.59,
      traffic_pcu: 16381,
      dist_hospital_km: 1.31,
      speed_kmh: 36.5,
      data_source: 'MULTIMODAL',
      damage_class: 'D20',
      damage_label: 'Alligator Crack',
      confidence: 0.91,
      image_id: 'India_000102',
      image_url: '/api/v1/images/India_000102.jpg',
      bbox: [120.0, 300.0, 580.0, 520.0],
      road_name: 'Swami Vivekananda (SV) Road, Santacruz West',
      road_class: 'ARTERIAL',
      processed_telemetry: {
        detection_modality: 'sonar_submerged_acoustic',
        is_submerged: false,
        estimated_depth_cm: 4.27,
        calculated_volume_liters: 75.51,
        dynamic_impact_confirmed: false
      },
      topsis_score: 0.5075,
      priority_level: 'High'
    },
    {
      id: 'SRMD-IND-000155-00',
      timestamp: '2026-09-24T11:52:11.227Z',
      vehicle_id: 'INSPECTION_FLEET_01',
      latitude: 19.0800,
      longitude: 72.8810,
      rain_detected: false,
      mean_luminance: 75.0,
      lidar_depth_cm: 30.79,
      sonar_depth_cm: 30.92,
      z_accel_g: 1.03,
      surface_area_sqm: 0.32,
      traffic_pcu: 11391,
      dist_hospital_km: 0.88,
      speed_kmh: 50.0,
      data_source: 'MULTIMODAL',
      damage_class: 'D00',
      damage_label: 'Longitudinal Crack',
      confidence: 0.88,
      image_id: 'India_000155',
      image_url: '/api/v1/images/India_000155.jpg',
      bbox: [100.0, 200.0, 350.0, 400.0],
      road_name: 'Link Road Corridor',
      road_class: 'COLLECTOR',
      processed_telemetry: {
        detection_modality: 'optical_lidar',
        is_submerged: false,
        estimated_depth_cm: 0.79,
        calculated_volume_liters: 2.53,
        dynamic_impact_confirmed: false
      },
      topsis_score: 0.2655,
      priority_level: 'Medium'
    },
    {
      id: 'SRMD-IND-000210-00',
      timestamp: '2026-09-24T11:52:11.228Z',
      vehicle_id: 'INSPECTION_FLEET_01',
      latitude: 19.0825,
      longitude: 72.8835,
      rain_detected: false,
      mean_luminance: 82.0,
      lidar_depth_cm: 30.99,
      sonar_depth_cm: 30.80,
      z_accel_g: 1.02,
      surface_area_sqm: 0.18,
      traffic_pcu: 5596,
      dist_hospital_km: 0.96,
      speed_kmh: 55.0,
      data_source: 'MULTIMODAL',
      damage_class: 'D10',
      damage_label: 'Transverse Crack',
      confidence: 0.84,
      image_id: 'India_000210',
      image_url: '/api/v1/images/India_000210.jpg',
      bbox: [150.0, 180.0, 320.0, 380.0],
      road_name: 'Local Arterial Bypass',
      road_class: 'LOCAL',
      processed_telemetry: {
        detection_modality: 'optical_lidar',
        is_submerged: false,
        estimated_depth_cm: 0.99,
        calculated_volume_liters: 1.78,
        dynamic_impact_confirmed: false
      },
      topsis_score: 0.2426,
      priority_level: 'Low'
    }
  ]
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
