import {
  apiDelete,
  apiDownload,
  apiGet,
  apiSend,
  uploadWithProgress,
} from "@/lib/api/client";
import type {
  DatasetProfile,
  FiguresList,
  JobCreated,
  JobStatus,
  PredictResponse,
  RunSummary,
  TrainRequest,
} from "@/lib/api/types";

export const getHealth = () => apiGet<{ status: string }>("/v1/health");

export const getDatasetProfile = (datasetId: string) => apiGet<DatasetProfile>(`/v1/datasets/${datasetId}`);

export const deleteDataset = (datasetId: string) => apiDelete(`/v1/datasets/${datasetId}`);

export const uploadDataset = (
  file: File,
  onProgress?: (pct: number) => void,
): Promise<DatasetProfile> => {
  const formData = new FormData();
  formData.append("file", file);
  return uploadWithProgress<DatasetProfile>("/v1/datasets/upload", formData, onProgress);
};

export const startTraining = (request: TrainRequest): Promise<JobCreated> =>
  apiSend<JobCreated>("/v1/train", "POST", request);

export const getJobStatus = (jobId: string, signal?: AbortSignal) =>
  apiGet<JobStatus>(`/v1/status/${jobId}`, signal);

export const getResults = (jobId: string) => apiGet<RunSummary>(`/v1/results/${jobId}`);

export const getFigures = (jobId: string) => apiGet<FiguresList>(`/v1/results/${jobId}/figures`);

export const figureUrl = (jobId: string, figurePath: string) =>
  `/api/v1/results/${jobId}/figures/${figurePath}`;

export const downloadResultsZip = (jobId: string) => apiDownload(`/v1/download/${jobId}`);

export async function predictRecords(
  jobId: string,
  records: Record<string, unknown>[],
  targets?: string[],
): Promise<PredictResponse> {
  const query = targets?.length ? `?targets=${targets.map(encodeURIComponent).join(",")}` : "";
  return apiSend<PredictResponse>(`/v1/predict/${jobId}${query}`, "POST", { records });
}

export function predictFile(
  jobId: string,
  file: File,
  targets?: string[],
  onProgress?: (pct: number) => void,
): Promise<PredictResponse> {
  const query = targets?.length ? `?targets=${targets.map(encodeURIComponent).join(",")}` : "";
  const formData = new FormData();
  formData.append("file", file);
  return uploadWithProgress<PredictResponse>(`/v1/predict/${jobId}${query}`, formData, onProgress);
}

export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
