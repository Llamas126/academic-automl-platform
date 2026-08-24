export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

const BASE = "/api";

function headers(extra?: HeadersInit): HeadersInit {
  const apiKey = process.env.NEXT_PUBLIC_API_KEY;
  const base: Record<string, string> = {};
  if (apiKey) base["X-API-Key"] = apiKey;
  return { ...base, ...extra };
}

async function parseError(response: Response): Promise<ApiError> {
  let detail = response.statusText || "Request failed";
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") detail = body.detail;
    else if (body && typeof body.error === "string") detail = `${body.error}: ${body.detail ?? ""}`;
  } catch {
    // non-JSON error body
  }
  return new ApiError(response.status, detail);
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: headers(),
    signal,
    cache: "no-store",
  });
  if (!response.ok) throw await parseError(response);
  return response.json() as Promise<T>;
}

export async function apiSend<T>(path: string, method: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    method,
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await parseError(response);
  return response.json() as Promise<T>;
}

export async function apiDelete(path: string): Promise<void> {
  const response = await fetch(`${BASE}${path}`, { method: "DELETE", headers: headers() });
  if (!response.ok) throw await parseError(response);
}

export async function apiDownload(path: string): Promise<{ blob: Blob; filename: string }> {
  const response = await fetch(`${BASE}${path}`, { headers: headers() });
  if (!response.ok) throw await parseError(response);
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = /filename="?([^";]+)"?/.exec(disposition);
  const filename = match?.[1] ?? path.split("/").pop() ?? "download.bin";
  return { blob: await response.blob(), filename };
}

export function uploadWithProgress<T>(
  path: string,
  formData: FormData,
  onProgress?: (pct: number) => void,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}${path}`);
    const apiKey = process.env.NEXT_PUBLIC_API_KEY;
    if (apiKey) xhr.setRequestHeader("X-API-Key", apiKey);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) onProgress(Math.round((event.loaded / event.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText) as T);
      } else {
        let detail = xhr.statusText || "Upload failed";
        try {
          const body = JSON.parse(xhr.responseText) as { detail?: string; error?: string };
          if (typeof body.detail === "string") detail = body.detail;
          else if (typeof body.error === "string") detail = `${body.error}: ${body.detail ?? ""}`;
        } catch {
          // keep default
        }
        reject(new ApiError(xhr.status, detail));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "Network error during upload"));
    xhr.send(formData);
  });
}
