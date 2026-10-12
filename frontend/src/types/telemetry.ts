export type PriorityLevel = 'Critical' | 'High' | 'Medium' | 'Low';

export type DataSourceType = 'RDD2022' | 'SYNTHETIC SENSOR' | 'SYNTHETIC GIS' | 'MULTIMODAL';

export type DetectorType = 'ground-truth lookup' | 'opencv-contour' | 'yolov8' | string;

export interface ProcessedTelemetry {
  detection_modality: 'optical_lidar' | 'sonar_submerged_acoustic' | string;
  is_submerged: boolean;
  estimated_depth_cm: number;
  calculated_volume_liters: number | null;
  crack_severity_index?: number | null;
  defect_severity_magnitude?: number;
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
  detector?: DetectorType;
  damage_class?: string;
  damage_label?: string;
  confidence?: number;
  image_id?: string;
  image_url?: string;
  bbox?: number[];
  area_clamped?: boolean;
  area_unclamped_m2?: number;
  road_name?: string;
  road_class?: string;
  topsis_rank?: number;
  vikor_rank?: number;
  vikor_Q?: number;
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
  detector?: DetectorType;
  latest_update: string | null;
}

export interface TelemetryResponse {
  status: string;
  detector?: DetectorType;
  total: number;
  summary: TelemetrySummary;
  data: TelemetryRecord[];
}

export interface DatasetExplorerStats {
  detector?: DetectorType;
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

export type WeightProfile = 'safety-first' | 'traffic-first' | 'equity-first';

export interface PrioritizationResultItem {
  id: string;
  rank: number;
  priority_level: PriorityLevel;
  damage_class?: string;
  damage_label?: string;
  road_name?: string;
  road_class?: string;
  surface_area_sqm?: number;
  area_clamped?: boolean;
  area_unclamped_m2?: number;
  calculated_volume_liters?: number | null;
  crack_severity_index?: number | null;
  defect_severity_magnitude?: number;
  traffic_pcu: number;
  dist_hospital_km: number;
  speed_kmh: number;
  school_proximity_km?: number;
  nearest_school_name?: string;
  topsis_score?: number;
  topsis_rank?: number;
  vikor_S?: number;
  vikor_R?: number;
  vikor_Q?: number;
  vikor_rank?: number;
}

export interface PrioritizationResponse {
  status: string;
  method: string;
  profile: string;
  weights: number[];
  spearman_correlation: number;
  vikor_checks?: {
    v: number;
    DQ: number;
    q_gap: number;
    acceptable_advantage: boolean;
    acceptable_stability: boolean;
    compromise_solution: string[];
  };
  total: number;
  results: PrioritizationResultItem[];
}

export interface PreprocessPreviewResponse {
  status: string;
  image_name: string;
  width: number;
  height: number;
  clip_limit: number;
  tile_grid: [number, number];
  backend: string;
  original_url: string;
  original_image: string;
  clahe_image: string;
  original_image_base64?: string;
  clahe_image_base64?: string;
  metrics: {
    original_mean_luminance: number;
    clahe_mean_luminance: number;
    original_contrast_std: number;
    clahe_contrast_std: number;
  };
}
