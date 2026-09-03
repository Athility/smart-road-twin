import { TelemetryResponse, PriorityLevel, TelemetryRecord } from '@/types/telemetry';

export const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1/telemetry/ingest';

export const FALLBACK_DATA: TelemetryResponse = {
  status: 'fallback',
  total: 4,
  summary: {
    total_defects: 4,
    total_volume_liters: 82.26,
    average_volume_liters: 20.57,
    weather_state: 'Rain Detected (Wet / Submerged Road)',
    is_raining: true,
    submerged_defects: 2,
    average_luminance: 49.4,
    priority_counts: {
      Critical: 1,
      High: 1,
      Medium: 1,
      Low: 1
    },
    latest_update: '2026-09-03T16:00:00.000Z'
  },
  data: [
    {
      id: 'POTHOLE-0001',
      timestamp: '2026-09-03T16:00:00.000Z',
      vehicle_id: 'INSPECTION_FLEET_01',
      latitude: 19.0760,
      longitude: 72.8777,
      rain_detected: true,
      mean_luminance: 18.5,
      lidar_depth_cm: 44.5,
      sonar_depth_cm: 45.8,
      z_accel_g: 2.4,
      surface_area_sqm: 0.65,
      traffic_pcu: 24500,
      dist_hospital_km: 0.4,
      speed_kmh: 42.0,
      processed_telemetry: {
        detection_modality: 'sonar_submerged_acoustic',
        is_submerged: true,
        estimated_depth_cm: 15.8,
        calculated_volume_liters: 51.35,
        dynamic_impact_confirmed: true
      },
      topsis_score: 0.948,
      priority_level: 'Critical'
    },
    {
      id: 'POTHOLE-0002',
      timestamp: '2026-09-03T15:58:00.000Z',
      vehicle_id: 'INSPECTION_FLEET_01',
      latitude: 19.0780,
      longitude: 72.8790,
      rain_detected: true,
      mean_luminance: 22.0,
      lidar_depth_cm: 38.5,
      sonar_depth_cm: 39.2,
      z_accel_g: 1.9,
      surface_area_sqm: 0.45,
      traffic_pcu: 21000,
      dist_hospital_km: 0.6,
      speed_kmh: 44.0,
      processed_telemetry: {
        detection_modality: 'sonar_submerged_acoustic',
        is_submerged: true,
        estimated_depth_cm: 9.2,
        calculated_volume_liters: 20.70,
        dynamic_impact_confirmed: true
      },
      topsis_score: 0.522,
      priority_level: 'High'
    },
    {
      id: 'POTHOLE-0003',
      timestamp: '2026-09-03T15:56:00.000Z',
      vehicle_id: 'INSPECTION_FLEET_01',
      latitude: 19.0800,
      longitude: 72.8810,
      rain_detected: false,
      mean_luminance: 75.0,
      lidar_depth_cm: 35.2,
      sonar_depth_cm: 34.0,
      z_accel_g: 1.4,
      surface_area_sqm: 0.32,
      traffic_pcu: 14200,
      dist_hospital_km: 1.5,
      speed_kmh: 50.0,
      processed_telemetry: {
        detection_modality: 'optical_lidar',
        is_submerged: false,
        estimated_depth_cm: 5.2,
        calculated_volume_liters: 8.32,
        dynamic_impact_confirmed: false
      },
      topsis_score: 0.282,
      priority_level: 'Medium'
    },
    {
      id: 'POTHOLE-0004',
      timestamp: '2026-09-03T15:54:00.000Z',
      vehicle_id: 'INSPECTION_FLEET_01',
      latitude: 19.0825,
      longitude: 72.8835,
      rain_detected: false,
      mean_luminance: 82.0,
      lidar_depth_cm: 32.1,
      sonar_depth_cm: 31.5,
      z_accel_g: 0.9,
      surface_area_sqm: 0.18,
      traffic_pcu: 8500,
      dist_hospital_km: 3.2,
      speed_kmh: 55.0,
      processed_telemetry: {
        detection_modality: 'optical_lidar',
        is_submerged: false,
        estimated_depth_cm: 2.1,
        calculated_volume_liters: 1.89,
        dynamic_impact_confirmed: false
      },
      topsis_score: 0.052,
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
