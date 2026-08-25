export type ColumnRole =
  | "numeric"
  | "categorical"
  | "boolean"
  | "datetime"
  | "identifier"
  | "text_free";

export interface ColumnProfile {
  name: string;
  role: ColumnRole;
  pandas_dtype: string;
  missing_pct: number;
  n_unique: number;
  stats: Record<string, number | string | null>;
  excluded_from_features: boolean;
}

export interface TargetSuggestion {
  name: string;
  suggested_task: "classification" | "regression";
  rationale: string;
}

export interface DatasetProfile {
  dataset_id: string;
  filename: string;
  created_at: string;
  n_rows: number;
  n_cols: number;
  columns: ColumnProfile[];
  target_suggestions: TargetSuggestion[];
}

export type TaskMode = "auto" | "classification" | "regression";

export interface TargetIn {
  name: string;
  task: TaskMode;
}

export interface CVConfig {
  n_folds: number;
  mode: "nested" | "single";
  n_inner_folds: number;
  random_state: number;
}

export interface HPOConfig {
  enabled: boolean;
  n_trials: number;
  timeout_s?: number | null;
}

export interface ExplainConfig {
  shap: boolean;
  background_samples: number;
  max_display: number;
}

export interface ExportConfig {
  figures_dpi: number;
  include_eps: boolean;
  include_shap_dependence: boolean;
}

export interface TrainRequest {
  dataset_id: string;
  targets: TargetIn[];
  models?: string[] | null;
  cv: CVConfig;
  hpo: HPOConfig;
  explain: ExplainConfig;
  export: ExportConfig;
}

export interface JobCreated {
  job_id: string;
  status_url: string;
  download_url: string;
}

export type JobState = "PENDING" | "STARTED" | "PROGRESS" | "SUCCESS" | "FAILURE";

export interface JobStatus {
  job_id: string;
  dataset_id: string | null;
  state: JobState;
  stage: string;
  progress: number;
  error: string | null;
  download_ready: boolean;
}

export interface MetricAggregate {
  metric: string;
  direction: "max" | "min";
  mean: number;
  std: number;
  ci95_low: number;
  ci95_high: number;
  folds: number[];
}

export interface ModelResult {
  model: string;
  citation_key: string | null;
  best_params: Record<string, unknown>;
  primary_metric: string;
  primary_mean: number;
  rank: number;
  metrics: MetricAggregate[];
}

export interface TargetResults {
  target: string;
  task: "classification" | "regression";
  label_encoder_classes: string[] | null;
  best_model: string;
  best_params: Record<string, unknown>;
  models: ModelResult[];
}

export interface RunSummary {
  job_id: string;
  dataset_id: string;
  filename: string;
  n_samples: number;
  n_features: number;
  feature_columns: string[];
  cv: CVConfig;
  hpo: HPOConfig;
  targets: TargetResults[];
  warnings: string[];
  library_versions: Record<string, string>;
  runtime_s: number;
}

export interface FiguresList {
  job_id: string;
  figures: string[];
}

export interface PredictTargetMeta {
  target: string;
  task: string;
  model: string | null;
  classes: string[] | null;
}

export interface PredictResponse {
  job_id: string;
  n_rows: number;
  targets: Record<string, PredictTargetMeta>;
  predictions: Record<string, unknown>[];
  warnings: string[];
}
