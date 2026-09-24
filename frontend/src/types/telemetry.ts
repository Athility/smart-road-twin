export type PriorityLevel = 'Critical' | 'High' | 'Medium' | 'Low';

export type DataSourceType = 'RDD2022' | 'SYNTHETIC SENSOR' | 'SYNTHETIC GIS' | 'MULTIMODAL';

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
  // Multimodal provenance & CV inspection attributes
  data_source?: DataSourceType;
  damage_class?: string;
  damage_label?: string;
  confidence?: number;
  image_id?: string;
  image_url?: string;
  bbox?: number[];
  road_name?: string;
  road_class?: string;
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

export interface DatasetExplorerStats {
  total_records: number;
  d00_count: number;
  d10_count: number;
  d20_count: number;
  d40_count: number;
  avg_lidar_depth: number;
  avg_sonar_depth: number;
  avg_acceleration: number;
  avg_traffic_pcu: number;
  avg_hospital_distance: number;
  rain_percentage: number;
  lidar_selected_percentage: number;
  sonar_selected_percentage: number;
  priority_distribution: PriorityCounts;
  records?: TelemetryRecord[];
}

