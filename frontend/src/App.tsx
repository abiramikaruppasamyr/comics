import type { FormEvent, ReactNode } from "react";
import { useEffect, useState } from "react";

import { generateImage, getSystemMetrics } from "./services/api";
import type { GenerateImageResponse, SystemMetrics } from "./types/api";

type FormState = {
  positivePrompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  steps: number;
  cfgScale: number;
  seed: string;
};

const initialState: FormState = {
  positivePrompt: "",
  negativePrompt: "",
  width: 512,
  height: 512,
  steps: 20,
  cfgScale: 7.5,
  seed: "",
};

export default function App() {
  const [form, setForm] = useState<FormState>(initialState);
  const [result, setResult] = useState<GenerateImageResponse | null>(null);
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
      setResult(response);
      setMetrics(response.system);
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
                  Tune generation parameters before each local inference run.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setForm(initialState)}
                className="rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-orange-400/30 hover:bg-white/5 hover:text-white"
              >
                Reset
              </button>
            </div>

            <div className="space-y-5">
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
                CPU-only generation. The model is loaded, used, and unloaded after each request.
              </p>
              <button
                type="submit"
                disabled={loading}
                className="inline-flex items-center justify-center rounded-full bg-orange-500 px-6 py-3 text-sm font-semibold text-slate-950 transition hover:bg-orange-400 disabled:cursor-not-allowed disabled:bg-orange-700"
              >
                {loading ? "Generating..." : "Generate Image"}
              </button>
            </div>
          </form>

          <div className="grid gap-6">
            <section className="rounded-[28px] border border-white/10 bg-[rgba(7,14,22,0.8)] p-5 shadow-panel backdrop-blur sm:p-6">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <h2 className="font-display text-2xl font-semibold text-white">Current Preview</h2>
                  <p className="mt-1 text-sm text-slate-400">
                    The latest image saved in the root output folder.
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

              <div className="overflow-hidden rounded-[24px] border border-white/10 bg-[rgba(255,255,255,0.03)]">
                {result ? (
                  <img
                    src={result.image.image_url}
                    alt={result.image.positive_prompt}
                    className="aspect-square w-full object-cover"
                  />
                ) : (
                  <div className="flex aspect-square items-center justify-center bg-[radial-gradient(circle_at_center,rgba(249,115,22,0.12),transparent_50%)] p-8 text-center text-sm text-slate-400">
                    Generate an image to preview the latest output here.
                  </div>
                )}
              </div>

              {result ? (
                <div className="mt-5 grid gap-3 sm:grid-cols-2">
                  <InfoChip label="Filename" value={result.image.filename} />
                  <InfoChip label="Seed" value={String(result.image.seed)} />
                  <InfoChip label="Size" value={`${result.image.width} x ${result.image.height}`} />
                  <InfoChip label="Steps / CFG" value={`${result.image.steps} / ${result.image.cfg_scale}`} />
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
