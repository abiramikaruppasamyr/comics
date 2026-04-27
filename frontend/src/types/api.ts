export type GenerateImageRequest = {
  positive_prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  steps: number;
  cfg_scale: number;
  denoise_strength: number;
  seed: number | null;
  lora_style: string;
  lora_strength: number;
};

export type LoraStyleOption = {
  key: string;
  label: string;
  default_strength: number;
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

export type ControlNetGenerateResponse = {
  image_url: string;
  lineart_preview_url: string;
  cpu_usage: number;
  ram_used: number;
  ram_total: number;
  seed_used: number;
  generation_time_seconds: number;
  image_filename: string;
  preprocessed_lineart_filename: string;
};

export type InpaintGenerateResponse = {
  image_url: string;
  cpu_usage: number;
  ram_used: number;
  ram_total: number;
  seed_used: number;
  generation_time_seconds: number;
};

export type IPAdapterGenerateResponse = {
  image_url: string;
  cpu_usage: number;
  ram_used: number;
  ram_total: number;
  seed_used: number;
  generation_time_seconds: number;
};
