export type PriorityLevel = 'Critical' | 'High' | 'Medium' | 'Low';

export interface ProcessedTelemetry {
  detection_modality: 'optical_lidar' | 'sonar_submerged_acoustic' | string;
  is_submerged: boolean;
  estimated_depth_cm: number;
  calculated_volume_liters: number;
  dynamic_impact_confirmed: boolean;
}

export interface TelemetryRecord {
  id: string;
  timestamp: string;
  vehicle_id: string;
  latitude: number;
  longitude: number;
  rain_detected: boolean;
  mean_luminance: number;
  lidar_depth_cm: number;
  sonar_depth_cm: number;
  z_accel_g: number;
  surface_area_sqm: number;
  traffic_pcu: number;
  dist_hospital_km: number;
  speed_kmh: number;
  processed_telemetry: ProcessedTelemetry;
  topsis_score: number;
  priority_level: PriorityLevel;
}

export interface PriorityCounts {
  Critical: number;
  High: number;
  Medium: number;
  Low: number;
}

export interface TelemetrySummary {
  total_defects: number;
  total_volume_liters: number;
  average_volume_liters: number;
  weather_state: string;
  is_raining: boolean;
  submerged_defects: number;
  average_luminance: number;
  priority_counts: PriorityCounts;
  latest_update: string | null;
}

export interface TelemetryResponse {
  status: string;
  total: number;
  summary: TelemetrySummary;
  data: TelemetryRecord[];
}
