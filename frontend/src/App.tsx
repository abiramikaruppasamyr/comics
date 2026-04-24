import type { FormEvent, ReactNode } from "react";
import { useEffect, useState } from "react";

import { generateControlNetImage, generateImage, getSystemMetrics } from "./services/api";
import type {
  ControlNetGenerateResponse,
  GenerateImageResponse,
  SystemMetrics,
} from "./types/api";

type GenerationMode = "normal" | "controlnet";

type FormState = {
  positivePrompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  steps: number;
  cfgScale: number;
  seed: string;
  controlnetConditioningScale: number;
};

const initialState: FormState = {
  positivePrompt: "",
  negativePrompt: "",
  width: 512,
  height: 512,
  steps: 20,
  cfgScale: 7.5,
  seed: "",
  controlnetConditioningScale: 1.0,
};

export default function App() {
  const [mode, setMode] = useState<GenerationMode>("normal");
  const [form, setForm] = useState<FormState>(initialState);
  const [normalResult, setNormalResult] = useState<GenerateImageResponse | null>(null);
  const [controlNetResult, setControlNetResult] = useState<ControlNetGenerateResponse | null>(null);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void refreshMetrics();
  }, []);

  async function refreshMetrics() {
    try {
      const snapshot = await getSystemMetrics();
      setMetrics(snapshot);
    } catch {
      setMetrics(null);
    }
  }

  function updateField<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (mode === "controlnet") {
        if (!uploadedImage) {
          throw new Error("Upload a sketch or reference image for ControlNet Lineart mode.");
        }

        const payload = new FormData();
        payload.append("image", uploadedImage);
        payload.append("positive_prompt", form.positivePrompt.trim());
        payload.append("negative_prompt", form.negativePrompt.trim());
        payload.append("width", String(Number(form.width)));
        payload.append("height", String(Number(form.height)));
        payload.append("steps", String(Number(form.steps)));
        payload.append("cfg_scale", String(Number(form.cfgScale)));
        payload.append(
          "seed",
          form.seed.trim() === "" ? "-1" : String(Number(form.seed)),
        );
        payload.append(
          "controlnet_conditioning_scale",
          String(Number(form.controlnetConditioningScale)),
        );

        const response = await generateControlNetImage(payload);
        setControlNetResult(response);
        setMetrics({
          cpu_percent: response.cpu_usage,
          memory_percent: Number(((response.ram_used / response.ram_total) * 100).toFixed(1)),
          memory_used_mb: response.ram_used,
          memory_available_mb: Number((response.ram_total - response.ram_used).toFixed(1)),
        });
      } else {
        const payload = {
          positive_prompt: form.positivePrompt.trim(),
          negative_prompt: form.negativePrompt.trim(),
          width: Number(form.width),
          height: Number(form.height),
          steps: Number(form.steps),
          cfg_scale: Number(form.cfgScale),
          seed: form.seed.trim() === "" ? null : Number(form.seed),
        };
        const response = await generateImage(payload);
        setNormalResult(response);
        setMetrics(response.system);
      }
    } catch (submitError) {
      const message =
        submitError instanceof Error ? submitError.message : "Unable to generate image.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen px-4 py-8 text-slate-100 sm:px-6 lg:px-10">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <header className="overflow-hidden rounded-[32px] border border-white/10 bg-[rgba(6,12,18,0.78)] p-6 shadow-panel backdrop-blur xl:p-8">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <p className="mb-3 inline-flex rounded-full border border-orange-400/20 bg-orange-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-orange-300">
                Offline CPU Pipeline
              </p>
              <h1 className="font-display text-4xl font-semibold tracking-tight text-white sm:text-5xl">
                Comics Local Generator
              </h1>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-300 sm:text-base">
                Stable Diffusion v1.5 scaffolded for local, offline-first image generation with
                prompt controls, preview output, and live system metrics after each run.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <MetricCard
                label="Scheduler"
                value="DPM++ 2M"
                detail="Karras sigmas"
                accent="orange"
              />
              <MetricCard
                label="Device"
                value="CPU"
                detail="GPU disabled"
                accent="green"
              />
              <MetricCard
                label="Output"
                value="PNG"
                detail="Saved to /output"
                accent="slate"
              />
            </div>
          </div>
        </header>

        <section className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
          <form
            onSubmit={handleSubmit}
            className="rounded-[28px] border border-white/10 bg-[rgba(7,14,22,0.8)] p-5 shadow-panel backdrop-blur sm:p-6"
          >
            <div className="mb-6 flex items-center justify-between">
              <div>
                <h2 className="font-display text-2xl font-semibold text-white">Prompt Controls</h2>
                <p className="mt-1 text-sm text-slate-400">
                  Switch between standard generation and lineart-guided ControlNet generation.
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  setForm(initialState);
                  setUploadedImage(null);
                }}
                className="rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-orange-400/30 hover:bg-white/5 hover:text-white"
              >
                Reset
              </button>
            </div>

            <div className="space-y-5">
              <div className="rounded-[24px] border border-white/10 bg-white/5 p-2">
                <div className="grid gap-2 sm:grid-cols-2">
                  <ModeButton
                    active={mode === "normal"}
                    label="Normal"
                    onClick={() => setMode("normal")}
                  />
                  <ModeButton
                    active={mode === "controlnet"}
                    label="ControlNet Lineart"
                    onClick={() => setMode("controlnet")}
                  />
                </div>
              </div>

              <FieldShell
                label="Positive Prompt"
                helper="Describe the subject, style, composition, lighting, and desired details."
              >
                <textarea
                  value={form.positivePrompt}
                  onChange={(event) => updateField("positivePrompt", event.target.value)}
                  placeholder="A cinematic comic-book hero portrait, dramatic rim light, richly inked detail..."
                  rows={5}
                  required
                  className="w-full resize-none rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-orange-400/50 focus:bg-white/[0.07]"
                />
              </FieldShell>

              <FieldShell
                label="Negative Prompt"
                helper="Suppress unwanted artifacts, styles, or visual defects."
              >
                <textarea
                  value={form.negativePrompt}
                  onChange={(event) => updateField("negativePrompt", event.target.value)}
                  placeholder="blurry, low quality, distorted face, extra fingers..."
                  rows={4}
                  className="w-full resize-none rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-orange-400/50 focus:bg-white/[0.07]"
                />
              </FieldShell>

              {mode === "controlnet" ? (
                <>
                  <FieldShell
                    label="Sketch / Reference Upload"
                    helper="Upload the source image that will be converted to lineart."
                  >
                    <label className="flex cursor-pointer flex-col items-center justify-center rounded-[24px] border border-dashed border-white/15 bg-white/[0.04] px-5 py-8 text-center transition hover:border-orange-400/35 hover:bg-white/[0.06]">
                      <span className="font-display text-lg font-semibold text-white">
                        {uploadedImage ? uploadedImage.name : "Choose an image"}
                      </span>
                      <span className="mt-2 text-sm text-slate-400">
                        PNG, JPG, or WEBP. This file is sent as multipart form data.
                      </span>
                      <input
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={(event) => setUploadedImage(event.target.files?.[0] ?? null)}
                      />
                    </label>
                  </FieldShell>

                  <FieldShell
                    label="ControlNet Strength"
                    helper={`${form.controlnetConditioningScale.toFixed(1)} between 0.1 and 2.0`}
                  >
                    <input
                      type="range"
                      min={0.1}
                      max={2.0}
                      step={0.1}
                      value={form.controlnetConditioningScale}
                      onChange={(event) =>
                        updateField("controlnetConditioningScale", Number(event.target.value))
                      }
                      className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-orange-500"
                    />
                  </FieldShell>
                </>
              ) : null}

              <div className="grid gap-4 sm:grid-cols-2">
                <NumberField
                  label="Width"
                  helper="Default 512"
                  min={64}
                  max={1024}
                  step={8}
                  value={form.width}
                  onChange={(value) => updateField("width", value)}
                />
                <NumberField
                  label="Height"
                  helper="Default 512"
                  min={64}
                  max={1024}
                  step={8}
                  value={form.height}
                  onChange={(value) => updateField("height", value)}
                />
                <NumberField
                  label="Steps"
                  helper="Default 20"
                  min={1}
                  max={100}
                  step={1}
                  value={form.steps}
                  onChange={(value) => updateField("steps", value)}
                />
                <NumberField
                  label="CFG Scale"
                  helper="Default 7.5"
                  min={1}
                  max={30}
                  step={0.5}
                  value={form.cfgScale}
                  onChange={(value) => updateField("cfgScale", value)}
                />
              </div>

              <FieldShell
                label="Seed"
                helper="Leave blank for random. Positive and negative values are accepted."
              >
                <input
                  value={form.seed}
                  onChange={(event) => updateField("seed", event.target.value)}
                  placeholder="Random"
                  inputMode="numeric"
                  className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-orange-400/50 focus:bg-white/[0.07]"
                />
              </FieldShell>
            </div>

            {error ? (
              <div className="mt-5 rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                {error}
              </div>
            ) : null}

            <div className="mt-6 flex flex-col gap-3 border-t border-white/10 pt-5 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm text-slate-400">
                CPU-only generation. Models are loaded from local files, used, then unloaded after each request.
              </p>
              <button
                type="submit"
                disabled={loading}
                className="inline-flex items-center justify-center rounded-full bg-orange-500 px-6 py-3 text-sm font-semibold text-slate-950 transition hover:bg-orange-400 disabled:cursor-not-allowed disabled:bg-orange-700"
              >
                {loading ? "Generating..." : mode === "controlnet" ? "Generate with ControlNet" : "Generate Image"}
              </button>
            </div>
          </form>

          <div className="grid gap-6">
            <section className="rounded-[28px] border border-white/10 bg-[rgba(7,14,22,0.8)] p-5 shadow-panel backdrop-blur sm:p-6">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <h2 className="font-display text-2xl font-semibold text-white">Current Preview</h2>
                  <p className="mt-1 text-sm text-slate-400">
                    {mode === "controlnet"
                      ? "Lineart preview and generated output for the latest ControlNet request."
                      : "The latest image saved in the root output folder."}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => void refreshMetrics()}
                  className="rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-green-400/30 hover:bg-white/5 hover:text-white"
                >
                  Refresh Metrics
                </button>
              </div>

              {mode === "controlnet" ? (
                controlNetResult ? (
                  <div className="grid gap-4 md:grid-cols-2">
                    <PreviewPanel
                      title="Lineart Preview"
                      src={controlNetResult.lineart_preview_url}
                      alt="Preprocessed lineart preview"
                    />
                    <PreviewPanel
                      title="Generated Output"
                      src={controlNetResult.image_url}
                      alt="ControlNet generated output"
                    />
                  </div>
                ) : (
                  <EmptyPreview message="Upload a sketch and generate to see the lineart preview and final output here." />
                )
              ) : normalResult ? (
                <div className="overflow-hidden rounded-[24px] border border-white/10 bg-[rgba(255,255,255,0.03)]">
                  <img
                    src={normalResult.image.image_url}
                    alt={normalResult.image.positive_prompt}
                    className="aspect-square w-full object-cover"
                  />
                </div>
              ) : (
                <EmptyPreview message="Generate an image to preview the latest output here." />
              )}

              {mode === "controlnet" && controlNetResult ? (
                <div className="mt-5 grid gap-3 sm:grid-cols-2">
                  <InfoChip label="Output File" value={controlNetResult.image_filename} />
                  <InfoChip
                    label="Lineart File"
                    value={controlNetResult.preprocessed_lineart_filename}
                  />
                  <InfoChip label="Seed" value={String(controlNetResult.seed_used)} />
                  <InfoChip
                    label="Generation Time"
                    value={`${controlNetResult.generation_time_seconds}s`}
                  />
                </div>
              ) : null}

              {mode === "normal" && normalResult ? (
                <div className="mt-5 grid gap-3 sm:grid-cols-2">
                  <InfoChip label="Filename" value={normalResult.image.filename} />
                  <InfoChip label="Seed" value={String(normalResult.image.seed)} />
                  <InfoChip
                    label="Size"
                    value={`${normalResult.image.width} x ${normalResult.image.height}`}
                  />
                  <InfoChip
                    label="Steps / CFG"
                    value={`${normalResult.image.steps} / ${normalResult.image.cfg_scale}`}
                  />
                </div>
              ) : null}
            </section>

            <section className="rounded-[28px] border border-white/10 bg-[rgba(7,14,22,0.8)] p-5 shadow-panel backdrop-blur sm:p-6">
              <h2 className="font-display text-2xl font-semibold text-white">System Usage</h2>
              <p className="mt-1 text-sm text-slate-400">
                CPU and RAM values from the backend after generation or manual refresh.
              </p>

              <div className="mt-5 grid gap-4 sm:grid-cols-2">
                <UsageCard
                  label="CPU Usage"
                  value={metrics ? `${metrics.cpu_percent}%` : "--"}
                  detail="Snapshot after request"
                  tone="orange"
                />
                <UsageCard
                  label="RAM Usage"
                  value={metrics ? `${metrics.memory_percent}%` : "--"}
                  detail={metrics ? `${metrics.memory_used_mb} MB used` : "Snapshot unavailable"}
                  tone="green"
                />
                <UsageCard
                  label="RAM Available"
                  value={metrics ? `${metrics.memory_available_mb} MB` : "--"}
                  detail="Free system memory"
                  tone="slate"
                />
                <UsageCard
                  label="Execution Mode"
                  value="CPU Only"
                  detail="CUDA visibility disabled"
                  tone="slate"
                />
              </div>
            </section>
          </div>
        </section>
      </div>
    </main>
  );
}

function ModeButton({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-2xl px-4 py-3 text-sm font-semibold transition ${
        active
          ? "bg-orange-500 text-slate-950"
          : "border border-white/10 bg-transparent text-slate-300 hover:border-orange-400/30 hover:bg-white/5 hover:text-white"
      }`}
    >
      {label}
    </button>
  );
}

function FieldShell({
  label,
  helper,
  children,
}: {
  label: string;
  helper: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-sm font-semibold text-white">{label}</span>
        <span className="text-xs text-slate-500">{helper}</span>
      </div>
      {children}
    </label>
  );
}

function NumberField({
  label,
  helper,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  helper: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  return (
    <FieldShell label={label} helper={helper}>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition focus:border-orange-400/50 focus:bg-white/[0.07]"
      />
    </FieldShell>
  );
}

function MetricCard({
  label,
  value,
  detail,
  accent,
}: {
  label: string;
  value: string;
  detail: string;
  accent: "orange" | "green" | "slate";
}) {
  const tone =
    accent === "orange"
      ? "border-orange-400/20 bg-orange-500/10 text-orange-200"
      : accent === "green"
        ? "border-green-400/20 bg-green-500/10 text-green-200"
        : "border-white/10 bg-white/5 text-slate-200";

  return (
    <div className={`rounded-2xl border px-4 py-3 ${tone}`}>
      <p className="text-xs uppercase tracking-[0.2em] text-slate-400">{label}</p>
      <p className="mt-2 font-display text-xl font-semibold">{value}</p>
      <p className="mt-1 text-sm text-slate-400">{detail}</p>
    </div>
  );
}

function UsageCard({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: string;
  detail: string;
  tone: "orange" | "green" | "slate";
}) {
  const palette =
    tone === "orange"
      ? "from-orange-500/15 to-orange-300/5"
      : tone === "green"
        ? "from-green-500/15 to-green-300/5"
        : "from-slate-500/15 to-slate-300/5";

  return (
    <div className={`rounded-3xl border border-white/10 bg-gradient-to-br ${palette} p-5`}>
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-3 font-display text-3xl font-semibold text-white">{value}</p>
      <p className="mt-2 text-sm text-slate-400">{detail}</p>
    </div>
  );
}

function InfoChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
      <p className="text-xs uppercase tracking-[0.2em] text-slate-500">{label}</p>
      <p className="mt-2 text-sm font-medium text-slate-100">{value}</p>
    </div>
  );
}

function EmptyPreview({ message }: { message: string }) {
  return (
    <div className="overflow-hidden rounded-[24px] border border-white/10 bg-[rgba(255,255,255,0.03)]">
      <div className="flex aspect-square items-center justify-center bg-[radial-gradient(circle_at_center,rgba(249,115,22,0.12),transparent_50%)] p-8 text-center text-sm text-slate-400">
        {message}
      </div>
    </div>
  );
}

function PreviewPanel({
  title,
  src,
  alt,
}: {
  title: string;
  src: string;
  alt: string;
}) {
  return (
    <div className="overflow-hidden rounded-[24px] border border-white/10 bg-[rgba(255,255,255,0.03)]">
      <div className="border-b border-white/10 px-4 py-3 text-sm font-semibold text-slate-200">
        {title}
      </div>
      <img src={src} alt={alt} className="aspect-square w-full object-cover" />
    </div>
  );
}
