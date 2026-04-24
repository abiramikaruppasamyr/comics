import type {
  ControlNetGenerateResponse,
  GenerateImageRequest,
  GenerateImageResponse,
  SystemMetrics,
} from "../types/api";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";
const OUTPUT_BASE_URL = import.meta.env.VITE_OUTPUT_BASE_URL ?? "http://127.0.0.1:8000";
const CONTROLNET_API_BASE_URL =
  import.meta.env.VITE_CONTROLNET_API_BASE_URL ?? "http://127.0.0.1:8000/api/controlnet";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = "Request failed.";
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export async function generateImage(payload: GenerateImageRequest): Promise<GenerateImageResponse> {
  const response = await fetch(`${API_BASE_URL}/generation`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await parseResponse<GenerateImageResponse>(response);
  return {
    ...data,
    image: {
      ...data.image,
      image_url: `${OUTPUT_BASE_URL}${data.image.image_url}`,
    },
  };
}

export async function getSystemMetrics(): Promise<SystemMetrics> {
  const response = await fetch(`${API_BASE_URL}/system/metrics`);
  return parseResponse<SystemMetrics>(response);
}

export async function generateControlNetImage(
  payload: FormData,
): Promise<ControlNetGenerateResponse> {
  const response = await fetch(`${CONTROLNET_API_BASE_URL}/generate`, {
    method: "POST",
    body: payload,
  });

  const data = await parseResponse<ControlNetGenerateResponse>(response);
  return {
    ...data,
    image_url: `${OUTPUT_BASE_URL}${data.image_url}`,
    lineart_preview_url: `${OUTPUT_BASE_URL}${data.lineart_preview_url}`,
  };
}
