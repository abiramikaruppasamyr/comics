export type GenerateImageRequest = {
  positive_prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  steps: number;
  cfg_scale: number;
  seed: number | null;
};

export type SystemMetrics = {
  cpu_percent: number;
  memory_percent: number;
  memory_used_mb: number;
  memory_available_mb: number;
};

export type GenerateImageResponse = {
  image: {
    filename: string;
    image_url: string;
    seed: number;
    width: number;
    height: number;
    steps: number;
    cfg_scale: number;
    positive_prompt: string;
    negative_prompt: string;
    generated_at: string;
  };
  system: SystemMetrics;
  message: string;
};
