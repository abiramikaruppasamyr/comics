import type { ChangeEvent, FormEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

import { generateControlNetImage, generateImage, getSystemMetrics } from "./services/api";
import type {
  ControlNetGenerateResponse,
  GenerateImageResponse,
  SystemMetrics,
} from "./types/api";

type GenerationMode = "normal" | "controlnet" | "canvas";

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

type CanvasComponent = {
  id: string;
  name: string;
  src: string;
  x: number;
  y: number;
  width: number;
  height: number;
  naturalWidth: number;
  naturalHeight: number;
  zIndex: number;
};

type DragState =
  | {
      type: "move";
      id: string;
      pointerId: number;
      offsetX: number;
      offsetY: number;
    }
  | {
      type: "resize";
      id: string;
      pointerId: number;
      startX: number;
      startY: number;
      startWidth: number;
      startHeight: number;
      aspectRatio: number;
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

const MIN_COMPONENT_SIZE = 32;

export default function App() {
  const [mode, setMode] = useState<GenerationMode>("normal");
  const [form, setForm] = useState<FormState>(initialState);
  const [normalResult, setNormalResult] = useState<GenerateImageResponse | null>(null);
  const [controlNetResult, setControlNetResult] = useState<ControlNetGenerateResponse | null>(null);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [canvasComponents, setCanvasComponents] = useState<CanvasComponent[]>([]);
  const [activeComponentId, setActiveComponentId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);

  const canvasStageRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    void refreshMetrics();
  }, []);

  useEffect(() => {
    setCanvasComponents((current) =>
      current.map((component) => {
        const clampedWidth = Math.min(component.width, Number(form.width));
        const clampedHeight = Math.min(component.height, Number(form.height));
        return {
          ...component,
          width: clampedWidth,
          height: clampedHeight,
          x: clamp(component.x, 0, Math.max(0, Number(form.width) - clampedWidth)),
          y: clamp(component.y, 0, Math.max(0, Number(form.height) - clampedHeight)),
        };
      }),
    );
  }, [form.width, form.height]);

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
      if (mode === "normal") {
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
      } else if (mode === "controlnet") {
        if (!uploadedImage) {
          throw new Error("Upload a sketch or reference image for ControlNet Lineart mode.");
        }
        await submitControlNetSource(uploadedImage);
      } else {
        const canvasFile = await exportCanvasAsFile();
        await submitControlNetSource(canvasFile);
      }
    } catch (submitError) {
      const message =
        submitError instanceof Error ? submitError.message : "Unable to generate image.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  async function submitControlNetSource(sourceFile: File) {
    const payload = new FormData();
    payload.append("image", sourceFile);
    payload.append("positive_prompt", form.positivePrompt.trim());
    payload.append("negative_prompt", form.negativePrompt.trim());
    payload.append("width", String(Number(form.width)));
    payload.append("height", String(Number(form.height)));
    payload.append("steps", String(Number(form.steps)));
    payload.append("cfg_scale", String(Number(form.cfgScale)));
    payload.append("seed", form.seed.trim() === "" ? "-1" : String(Number(form.seed)));
    payload.append("controlnet_conditioning_scale", String(Number(form.controlnetConditioningScale)));

    const response = await generateControlNetImage(payload);
    setControlNetResult(response);
    setMetrics({
      cpu_percent: response.cpu_usage,
      memory_percent: Number(((response.ram_used / response.ram_total) * 100).toFixed(1)),
      memory_used_mb: response.ram_used,
      memory_available_mb: Number((response.ram_total - response.ram_used).toFixed(1)),
    });
  }

  async function exportCanvasAsFile(): Promise<File> {
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = Number(form.width);
    exportCanvas.height = Number(form.height);
    const context = exportCanvas.getContext("2d");

    if (!context) {
      throw new Error("Failed to prepare canvas export.");
    }

    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

    const orderedComponents = [...canvasComponents].sort((left, right) => left.zIndex - right.zIndex);
    await Promise.all(
      orderedComponents.map(
        (component) =>
          new Promise<void>((resolve, reject) => {
            const image = new Image();
            image.onload = () => {
              context.drawImage(image, component.x, component.y, component.width, component.height);
              resolve();
            };
            image.onerror = () => reject(new Error(`Failed to render canvas component: ${component.name}`));
            image.src = component.src;
          }),
      ),
    );

    const blob = await new Promise<Blob | null>((resolve) => {
      exportCanvas.toBlob(resolve, "image/png");
    });

    if (!blob) {
      throw new Error("Canvas export failed.");
    }

    return new File([blob], "canvas-compose.png", { type: "image/png" });
  }

  function handleControlNetUpload(event: ChangeEvent<HTMLInputElement>) {
    setUploadedImage(event.target.files?.[0] ?? null);
  }

  function handleCanvasComponentUpload(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    if (files.length === 0) {
      return;
    }

    void Promise.all(files.map((file) => loadCanvasComponent(file))).then((components) => {
      setCanvasComponents((current) => {
        const highestZIndex = current.reduce((maxValue, item) => Math.max(maxValue, item.zIndex), 0);
        return [
          ...current,
          ...components.map((component, index) => ({
            ...component,
            zIndex: highestZIndex + index + 1,
          })),
        ];
      });
      const lastComponent = components.length > 0 ? components[components.length - 1] : null;
      setActiveComponentId(lastComponent?.id ?? null);
    });

    event.target.value = "";
  }

  function loadCanvasComponent(file: File): Promise<CanvasComponent> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const src = reader.result;
        if (typeof src !== "string") {
          reject(new Error(`Failed to read ${file.name}.`));
          return;
        }

        const image = new Image();
        image.onload = () => {
          const stageWidth = Number(form.width);
          const stageHeight = Number(form.height);
          const defaultWidth = Math.min(stageWidth * 0.45, image.width);
          const aspectRatio = image.height / image.width;
          const defaultHeight = Math.max(MIN_COMPONENT_SIZE, Math.round(defaultWidth * aspectRatio));
          const componentWidth = Math.max(MIN_COMPONENT_SIZE, Math.round(defaultWidth));
          const componentHeight = Math.max(MIN_COMPONENT_SIZE, Math.round(defaultHeight));

          resolve({
            id: `${file.name}-${crypto.randomUUID()}`,
            name: file.name,
            src,
            x: Math.max(0, Math.round((stageWidth - componentWidth) / 2)),
            y: Math.max(0, Math.round((stageHeight - componentHeight) / 2)),
            width: componentWidth,
            height: componentHeight,
            naturalWidth: image.width,
            naturalHeight: image.height,
            zIndex: 1,
          });
        };
        image.onerror = () => reject(new Error(`Failed to load ${file.name}.`));
        image.src = src;
      };
      reader.onerror = () => reject(new Error(`Failed to read ${file.name}.`));
      reader.readAsDataURL(file);
    });
  }

  function resetFormState() {
    setForm(initialState);
    setUploadedImage(null);
    setCanvasComponents([]);
    setActiveComponentId(null);
  }

  function bringComponentToFront(id: string) {
    setCanvasComponents((current) => {
      const highestZIndex = current.reduce((maxValue, item) => Math.max(maxValue, item.zIndex), 0);
      return current.map((component) =>
        component.id === id ? { ...component, zIndex: highestZIndex + 1 } : component,
      );
    });
    setActiveComponentId(id);
  }

  function removeCanvasComponent(id: string) {
    setCanvasComponents((current) => current.filter((component) => component.id !== id));
    setActiveComponentId((current) => (current === id ? null : current));
  }

  function clearCanvas() {
    setCanvasComponents([]);
    setActiveComponentId(null);
  }

  function handleStagePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    if (!dragState || !canvasStageRef.current) {
      return;
    }

    const rect = canvasStageRef.current.getBoundingClientRect();
    const pointerX = ((event.clientX - rect.left) / rect.width) * Number(form.width);
    const pointerY = ((event.clientY - rect.top) / rect.height) * Number(form.height);

    if (dragState.type === "move") {
      setCanvasComponents((current) =>
        current.map((component) => {
          if (component.id !== dragState.id) {
            return component;
          }

          const nextX = clamp(pointerX - dragState.offsetX, 0, Number(form.width) - component.width);
          const nextY = clamp(pointerY - dragState.offsetY, 0, Number(form.height) - component.height);
          return { ...component, x: nextX, y: nextY };
        }),
      );
      return;
    }

    setCanvasComponents((current) =>
      current.map((component) => {
        if (component.id !== dragState.id) {
          return component;
        }

        const deltaX = pointerX - dragState.startX;
        const nextWidth = clamp(
          dragState.startWidth + deltaX,
          MIN_COMPONENT_SIZE,
          Number(form.width) - component.x,
        );
        const nextHeight = clamp(
          Math.round(nextWidth * dragState.aspectRatio),
          MIN_COMPONENT_SIZE,
          Number(form.height) - component.y,
        );

        return {
          ...component,
          width: nextWidth,
          height: nextHeight,
        };
      }),
    );
  }

  function handleStagePointerUp() {
    setDragState(null);
  }

  function startMove(
    event: ReactPointerEvent<HTMLDivElement>,
    component: CanvasComponent,
  ) {
    if (!canvasStageRef.current) {
      return;
    }

    const rect = canvasStageRef.current.getBoundingClientRect();
    const pointerX = ((event.clientX - rect.left) / rect.width) * Number(form.width);
    const pointerY = ((event.clientY - rect.top) / rect.height) * Number(form.height);

    bringComponentToFront(component.id);
    setDragState({
      type: "move",
      id: component.id,
      pointerId: event.pointerId,
      offsetX: pointerX - component.x,
      offsetY: pointerY - component.y,
    });
  }

  function startResize(
    event: ReactPointerEvent<HTMLButtonElement>,
    component: CanvasComponent,
  ) {
    event.stopPropagation();
    bringComponentToFront(component.id);
    setDragState({
      type: "resize",
      id: component.id,
      pointerId: event.pointerId,
      startX: component.x + component.width,
      startY: component.y + component.height,
      startWidth: component.width,
      startHeight: component.height,
      aspectRatio: component.naturalHeight / component.naturalWidth,
    });
  }

  const isControlGuidedMode = mode === "controlnet" || mode === "canvas";

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
                prompt controls, ControlNet guidance, and freeform canvas composition.
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
                label="Modes"
                value="3"
                detail="Normal, Lineart, Canvas"
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
                  Switch between standard generation, direct lineart upload, and canvas-driven composition.
                </p>
              </div>
              <button
                type="button"
                onClick={resetFormState}
                className="rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-orange-400/30 hover:bg-white/5 hover:text-white"
              >
                Reset
              </button>
            </div>

            <div className="space-y-5">
              <div className="rounded-[24px] border border-white/10 bg-white/5 p-2">
                <div className="grid gap-2 sm:grid-cols-3">
                  <ModeButton active={mode === "normal"} label="Normal" onClick={() => setMode("normal")} />
                  <ModeButton
                    active={mode === "controlnet"}
                    label="ControlNet Lineart"
                    onClick={() => setMode("controlnet")}
                  />
                  <ModeButton
                    active={mode === "canvas"}
                    label="Canvas Compose"
                    onClick={() => setMode("canvas")}
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
                      onChange={handleControlNetUpload}
                    />
                  </label>
                </FieldShell>
              ) : null}

              {mode === "canvas" ? (
                <FieldShell
                  label="Canvas Components"
                  helper="Add multiple reference images and compose them directly on the canvas."
                >
                  <label className="flex cursor-pointer items-center justify-between rounded-[24px] border border-dashed border-white/15 bg-white/[0.04] px-5 py-4 transition hover:border-orange-400/35 hover:bg-white/[0.06]">
                    <div>
                      <div className="font-display text-lg font-semibold text-white">
                        Add Component to Canvas
                      </div>
                      <div className="mt-1 text-sm text-slate-400">
                        {canvasComponents.length} component{canvasComponents.length === 1 ? "" : "s"} on canvas
                      </div>
                    </div>
                    <div className="rounded-full bg-orange-500 px-4 py-2 text-sm font-semibold text-slate-950">
                      Add
                    </div>
                    <input
                      type="file"
                      accept="image/*"
                      multiple
                      className="hidden"
                      onChange={handleCanvasComponentUpload}
                    />
                  </label>
                </FieldShell>
              ) : null}

              {isControlGuidedMode ? (
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
                CPU-only generation. Control-guided modes reuse the existing ControlNet endpoint and export a flat input image when needed.
              </p>
              <button
                type="submit"
                disabled={loading}
                className="inline-flex items-center justify-center rounded-full bg-orange-500 px-6 py-3 text-sm font-semibold text-slate-950 transition hover:bg-orange-400 disabled:cursor-not-allowed disabled:bg-orange-700"
              >
                {loading
                  ? "Generating..."
                  : mode === "normal"
                    ? "Generate Image"
                    : mode === "canvas"
                      ? "Generate from Canvas"
                      : "Generate with ControlNet"}
              </button>
            </div>
          </form>

          <div className="grid gap-6">
            <section className="rounded-[28px] border border-white/10 bg-[rgba(7,14,22,0.8)] p-5 shadow-panel backdrop-blur sm:p-6">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <h2 className="font-display text-2xl font-semibold text-white">
                    {mode === "canvas" ? "Canvas Compose" : "Current Preview"}
                  </h2>
                  <p className="mt-1 text-sm text-slate-400">
                    {mode === "normal"
                      ? "The latest image saved in the root output folder."
                      : mode === "controlnet"
                        ? "Lineart preview and generated output for the latest ControlNet request."
                        : "Arrange uploaded components, then export the canvas through the existing ControlNet pipeline."}
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

              {mode === "canvas" ? (
                <div className="space-y-4">
                  <div className="overflow-auto rounded-[24px] border border-white/10 bg-[linear-gradient(135deg,rgba(15,23,42,0.92),rgba(30,41,59,0.9))] p-3">
                    <div
                      ref={canvasStageRef}
                      className="relative mx-auto touch-none overflow-hidden rounded-[20px] border border-white/10 bg-[linear-gradient(180deg,#1f2937,#0f172a)] shadow-inner"
                      style={{
                        width: `${form.width}px`,
                        height: `${form.height}px`,
                        backgroundImage:
                          "linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)",
                        backgroundSize: "24px 24px",
                      }}
                      onPointerMove={handleStagePointerMove}
                      onPointerUp={handleStagePointerUp}
                      onPointerLeave={handleStagePointerUp}
                    >
                      {canvasComponents.length === 0 ? (
                        <div className="flex h-full items-center justify-center p-8 text-center text-sm text-slate-400">
                          Add one or more components, drag them into place, resize from the bottom-right handle, and generate when ready.
                        </div>
                      ) : null}

                      {[...canvasComponents]
                        .sort((left, right) => left.zIndex - right.zIndex)
                        .map((component) => (
                          <div
                            key={component.id}
                            className={`absolute overflow-hidden rounded-xl border ${
                              activeComponentId === component.id
                                ? "border-orange-400/70 shadow-[0_0_0_1px_rgba(249,115,22,0.45)]"
                                : "border-white/10"
                            }`}
                            style={{
                              left: `${component.x}px`,
                              top: `${component.y}px`,
                              width: `${component.width}px`,
                              height: `${component.height}px`,
                              zIndex: component.zIndex,
                            }}
                            onPointerDown={(event) => startMove(event, component)}
                          >
                            <img
                              src={component.src}
                              alt={component.name}
                              draggable={false}
                              className="h-full w-full select-none object-cover"
                            />
                            <button
                              type="button"
                              onClick={(event) => {
                                event.stopPropagation();
                                removeCanvasComponent(component.id);
                              }}
                              className="absolute right-2 top-2 inline-flex h-7 w-7 items-center justify-center rounded-full bg-slate-950/85 text-xs font-semibold text-white shadow-lg"
                            >
                              X
                            </button>
                            <button
                              type="button"
                              onPointerDown={(event) => startResize(event, component)}
                              className="absolute bottom-1 right-1 h-5 w-5 cursor-se-resize rounded-md border border-white/20 bg-orange-500/90"
                            />
                          </div>
                        ))}
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="text-sm text-slate-400">
                      Canvas matches the current width and height settings: {form.width} x {form.height}
                    </div>
                    <button
                      type="button"
                      onClick={clearCanvas}
                      className="rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-red-400/30 hover:bg-white/5 hover:text-white"
                    >
                      Clear Canvas
                    </button>
                  </div>

                  {controlNetResult ? (
                    <div className="grid gap-4 md:grid-cols-2">
                      <PreviewPanel
                        title="Lineart Preview"
                        src={controlNetResult.lineart_preview_url}
                        alt="Preprocessed lineart preview"
                      />
                      <PreviewPanel
                        title="Generated Output"
                        src={controlNetResult.image_url}
                        alt="Canvas Compose generated output"
                      />
                    </div>
                  ) : null}
                </div>
              ) : isControlGuidedMode ? (
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
                  <EmptyPreview message="Generate to see the lineart preview and final output here." />
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

              {isControlGuidedMode && controlNetResult ? (
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
                  detail="Canvas mode reuses ControlNet generation"
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

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
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
