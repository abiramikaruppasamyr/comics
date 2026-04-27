import type { ChangeEvent, FormEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

import {
  generateControlNetImage,
  generateImage,
  generateIPAdapterImage,
  generateInpaintImage,
  getLoraStyles,
  getSystemMetrics,
} from "./services/api";
import type {
  ControlNetGenerateResponse,
  GenerateImageResponse,
  IPAdapterGenerateResponse,
  InpaintGenerateResponse,
  LoraStyleOption,
  SystemMetrics,
} from "./types/api";

type GenerationMode = "normal" | "controlnet" | "canvas" | "upload-inpaint" | "ip-adapter";

type FormState = {
  positivePrompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  steps: number;
  cfgScale: number;
  denoiseStrength: number;
  seed: string;
  controlnetConditioningScale: number;
  ipAdapterScale: number;
  loraStyle: string;
  loraStrength: number;
};

type InpaintFormState = {
  positivePrompt: string;
  negativePrompt: string;
  loraStyle: string;
  loraStrength: number;
  width: number;
  height: number;
  steps: number;
  cfgScale: number;
  denoiseStrength: number;
  seed: string;
  brushSize: number;
  eraseMode: boolean;
};

type CanvasComponent = {
  id: string;
  name: string;
  src: string;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  naturalWidth: number;
  naturalHeight: number;
  zIndex: number;
};

type ReferenceImage = {
  id: string;
  name: string;
  src: string;
  file: File;
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
      corner: "bottom-left" | "bottom-right";
      startX: number;
      startY: number;
      startComponentX: number;
      startComponentY: number;
      startWidth: number;
      startHeight: number;
      aspectRatio: number;
      preserveAspectRatio: boolean;
    }
  | {
      type: "rotate";
      id: string;
      pointerId: number;
      centerX: number;
      centerY: number;
      snapToAngles: boolean;
    };

type EditableGeneratedImage = {
  src: string;
  name: string;
  width: number;
  height: number;
  positivePrompt: string;
  negativePrompt: string;
  steps: number;
  cfgScale: number;
  denoiseStrength: number;
  seed: string;
  loraStyle: string;
  loraStrength: number;
  file?: File;
};

type NormalResultState = {
  response: GenerateImageResponse;
  editable: EditableGeneratedImage;
};

type ControlResultState = {
  response: ControlNetGenerateResponse;
  editable: EditableGeneratedImage;
};

type InpaintResultState = {
  response: InpaintGenerateResponse;
  original: EditableGeneratedImage;
};

type IPAdapterResultState = {
  response: IPAdapterGenerateResponse;
};

const MIN_COMPONENT_SIZE = 32;

function createInitialFormState(): FormState {
  return {
    positivePrompt: "",
    negativePrompt: "",
    width: 512,
    height: 512,
    steps: 20,
    cfgScale: 7.5,
    denoiseStrength: 1.0,
    seed: "",
    controlnetConditioningScale: 1.0,
    ipAdapterScale: 0.6,
    loraStyle: "",
    loraStrength: 1.0,
  };
}

function createInitialInpaintState(source: EditableGeneratedImage | null, defaultStyle?: LoraStyleOption): InpaintFormState {
  return {
    positivePrompt: source?.positivePrompt ?? "",
    negativePrompt: source?.negativePrompt ?? "",
    loraStyle: source?.loraStyle ?? defaultStyle?.key ?? "",
    loraStrength: source?.loraStrength ?? defaultStyle?.default_strength ?? 1.0,
    width: source?.width ?? 512,
    height: source?.height ?? 512,
    steps: source?.steps ?? 20,
    cfgScale: source?.cfgScale ?? 7.5,
    denoiseStrength: source?.denoiseStrength ?? 1.0,
    seed: source?.seed ?? "-1",
    brushSize: 20,
    eraseMode: false,
  };
}

export default function App() {
  const [mode, setMode] = useState<GenerationMode>("normal");
  const [form, setForm] = useState<FormState>(createInitialFormState);
  const [isLoraEnabled, setIsLoraEnabled] = useState(true);
  const [normalResult, setNormalResult] = useState<NormalResultState | null>(null);
  const [controlNetResult, setControlNetResult] = useState<ControlResultState | null>(null);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [canvasComponents, setCanvasComponents] = useState<CanvasComponent[]>([]);
  const [activeComponentId, setActiveComponentId] = useState<string | null>(null);
  const [loraStyles, setLoraStyles] = useState<LoraStyleOption[]>([]);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [inpaintLoading, setInpaintLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inpaintError, setInpaintError] = useState<string | null>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [inpaintSource, setInpaintSource] = useState<EditableGeneratedImage | null>(null);
  const [inpaintForm, setInpaintForm] = useState<InpaintFormState>(createInitialInpaintState(null));
  const [inpaintResult, setInpaintResult] = useState<InpaintResultState | null>(null);
  const [showSideBySide, setShowSideBySide] = useState(true);
  const [uploadInpaintSource, setUploadInpaintSource] = useState<EditableGeneratedImage | null>(null);
  const [uploadInpaintForm, setUploadInpaintForm] = useState<InpaintFormState>(createInitialInpaintState(null));
  const [uploadInpaintResult, setUploadInpaintResult] = useState<InpaintResultState | null>(null);
  const [uploadInpaintError, setUploadInpaintError] = useState<string | null>(null);
  const [showUploadSideBySide, setShowUploadSideBySide] = useState(true);
  const [ipAdapterReferenceImages, setIpAdapterReferenceImages] = useState<ReferenceImage[]>([]);
  const [ipAdapterResult, setIpAdapterResult] = useState<IPAdapterResultState | null>(null);
  const [isMaskDrawing, setIsMaskDrawing] = useState(false);

  const canvasStageRef = useRef<HTMLDivElement | null>(null);
  const inpaintPanelRef = useRef<HTMLDivElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const activeInpaintSource = mode === "upload-inpaint" ? uploadInpaintSource : inpaintSource;
  const activeInpaintForm = mode === "upload-inpaint" ? uploadInpaintForm : inpaintForm;
  const mainLoraStrength = isLoraEnabled ? Number(form.loraStrength) : 0.0;
  const uploadInpaintLoraStrength = isLoraEnabled ? Number(uploadInpaintForm.loraStrength) : 0.0;
  const inpaintLoraStrength = isLoraEnabled ? Number(inpaintForm.loraStrength) : 0.0;
  const loraControlOpacity = isLoraEnabled ? "" : "opacity-50";

  useEffect(() => {
    void refreshMetrics();
  }, []);

  useEffect(() => {
    void loadLoraStyles();
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

  useEffect(() => {
    if (!activeInpaintSource) {
      return;
    }

    initializeMaskCanvases();
  }, [activeInpaintSource, activeInpaintForm.width, activeInpaintForm.height]);

  useEffect(() => {
    if (!inpaintSource || !inpaintPanelRef.current) {
      return;
    }

    inpaintPanelRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [inpaintSource]);

  useEffect(() => {
    function handleCanvasKeyDown(event: KeyboardEvent) {
      if (mode !== "canvas" || !activeComponentId) {
        return;
      }

      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement
      ) {
        return;
      }

      if (event.key === "Escape") {
        event.preventDefault();
        setActiveComponentId(null);
        return;
      }

      if (event.key === "Delete" || event.key === "Backspace") {
        event.preventDefault();
        removeCanvasComponent(activeComponentId);
        return;
      }

      const nudgeDistance = event.shiftKey ? 10 : 1;
      const directionMap: Record<string, { x: number; y: number }> = {
        ArrowUp: { x: 0, y: -nudgeDistance },
        ArrowDown: { x: 0, y: nudgeDistance },
        ArrowLeft: { x: -nudgeDistance, y: 0 },
        ArrowRight: { x: nudgeDistance, y: 0 },
      };
      const direction = directionMap[event.key];
      if (!direction) {
        return;
      }

      event.preventDefault();
      setCanvasComponents((current) =>
        current.map((component) =>
          component.id === activeComponentId
            ? {
                ...component,
                x: clamp(component.x + direction.x, 0, Math.max(0, Number(form.width) - component.width)),
                y: clamp(component.y + direction.y, 0, Math.max(0, Number(form.height) - component.height)),
              }
            : component,
        ),
      );
    }

    window.addEventListener("keydown", handleCanvasKeyDown);
    return () => window.removeEventListener("keydown", handleCanvasKeyDown);
  }, [activeComponentId, form.height, form.width, mode]);

  async function refreshMetrics() {
    try {
      const snapshot = await getSystemMetrics();
      setMetrics(snapshot);
    } catch {
      setMetrics(null);
    }
  }

  async function loadLoraStyles() {
    try {
      const styles = await getLoraStyles();
      if (styles.length === 0) {
        throw new Error("No art styles were returned by the backend.");
      }

      setLoraStyles(styles);
      setForm((current) => {
        const selectedStyle = styles.find((style) => style.key === current.loraStyle) ?? styles[0];
        return {
          ...current,
          loraStyle: selectedStyle.key,
          loraStrength: current.loraStyle === selectedStyle.key ? current.loraStrength : selectedStyle.default_strength,
        };
      });
      setInpaintForm((current) => {
        const selectedStyle = styles.find((style) => style.key === current.loraStyle) ?? styles[0];
        return {
          ...current,
          loraStyle: selectedStyle.key,
          loraStrength: current.loraStyle === selectedStyle.key ? current.loraStrength : selectedStyle.default_strength,
        };
      });
      setUploadInpaintForm((current) => {
        const selectedStyle = styles.find((style) => style.key === current.loraStyle) ?? styles[0];
        return {
          ...current,
          loraStyle: selectedStyle.key,
          loraStrength: current.loraStyle === selectedStyle.key ? current.loraStrength : selectedStyle.default_strength,
        };
      });
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : "Unable to load art styles.";
      setError(message);
    }
  }

  function updateField<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function updateInpaintField<K extends keyof InpaintFormState>(key: K, value: InpaintFormState[K]) {
    setInpaintForm((current) => ({ ...current, [key]: value }));
  }

  function updateUploadInpaintField<K extends keyof InpaintFormState>(key: K, value: InpaintFormState[K]) {
    setUploadInpaintForm((current) => ({ ...current, [key]: value }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    if (mode === "upload-inpaint") {
      setUploadInpaintError(null);
    }

    try {
      if (mode === "normal") {
        if (!form.loraStyle) {
          throw new Error("Select an art style before generating.");
        }

        const payload = {
          positive_prompt: form.positivePrompt.trim(),
          negative_prompt: form.negativePrompt.trim(),
          width: Number(form.width),
          height: Number(form.height),
          steps: Number(form.steps),
          cfg_scale: Number(form.cfgScale),
          denoise_strength: Number(form.denoiseStrength),
          seed: form.seed.trim() === "" ? null : Number(form.seed),
          lora_style: form.loraStyle,
          lora_strength: mainLoraStrength,
        };
        const response = await generateImage(payload);
        setNormalResult({
          response,
          editable: {
            src: response.image.image_url,
            name: response.image.filename,
            width: response.image.width,
            height: response.image.height,
            positivePrompt: form.positivePrompt.trim(),
            negativePrompt: form.negativePrompt.trim(),
            steps: Number(form.steps),
            cfgScale: Number(form.cfgScale),
            denoiseStrength: Number(form.denoiseStrength),
            seed: form.seed.trim() === "" ? "-1" : form.seed.trim(),
            loraStyle: form.loraStyle,
            loraStrength: mainLoraStrength,
          },
        });
        setMetrics(response.system);
      } else if (mode === "controlnet") {
        if (!uploadedImage) {
          throw new Error("Upload a sketch or reference image for ControlNet Lineart mode.");
        }
        await submitControlNetSource(uploadedImage);
      } else if (mode === "canvas") {
        const canvasFile = await exportCanvasAsFile();
        await submitControlNetSource(canvasFile);
      } else if (mode === "ip-adapter") {
        if (ipAdapterReferenceImages.length === 0) {
          throw new Error("Upload at least one reference image for IP-Adapter mode.");
        }

        const response = await submitIPAdapterRequest();
        setIpAdapterResult({ response });
        setMetrics({
          cpu_percent: response.cpu_usage,
          memory_percent: Number(((response.ram_used / response.ram_total) * 100).toFixed(1)),
          memory_used_mb: response.ram_used,
          memory_available_mb: Number((response.ram_total - response.ram_used).toFixed(1)),
        });
      } else {
        if (!uploadInpaintSource) {
          throw new Error("Upload an image before generating.");
        }

        const response = await submitInpaintRequest(uploadInpaintSource, uploadInpaintForm);
        setUploadInpaintResult({ response, original: uploadInpaintSource });
        setShowUploadSideBySide(true);
        setMetrics({
          cpu_percent: response.cpu_usage,
          memory_percent: Number(((response.ram_used / response.ram_total) * 100).toFixed(1)),
          memory_used_mb: response.ram_used,
          memory_available_mb: Number((response.ram_total - response.ram_used).toFixed(1)),
        });
      }
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : "Unable to generate image.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  async function submitControlNetSource(sourceFile: File) {
    if (!form.loraStyle) {
      throw new Error("Select an art style before generating.");
    }

    const payload = new FormData();
    payload.append("image", sourceFile);
    payload.append("positive_prompt", form.positivePrompt.trim());
    payload.append("negative_prompt", form.negativePrompt.trim());
    payload.append("width", String(Number(form.width)));
    payload.append("height", String(Number(form.height)));
    payload.append("steps", String(Number(form.steps)));
    payload.append("cfg_scale", String(Number(form.cfgScale)));
    payload.append("denoise_strength", String(Number(form.denoiseStrength)));
    payload.append("seed", form.seed.trim() === "" ? "-1" : String(Number(form.seed)));
    payload.append("controlnet_conditioning_scale", String(Number(form.controlnetConditioningScale)));
    payload.append("lora_style", form.loraStyle);
    payload.append("lora_strength", String(mainLoraStrength));

    const response = await generateControlNetImage(payload);
    setControlNetResult({
      response,
      editable: {
        src: response.image_url,
        name: response.image_filename,
        width: Number(form.width),
        height: Number(form.height),
        positivePrompt: form.positivePrompt.trim(),
        negativePrompt: form.negativePrompt.trim(),
        steps: Number(form.steps),
        cfgScale: Number(form.cfgScale),
        denoiseStrength: Number(form.denoiseStrength),
        seed: form.seed.trim() === "" ? "-1" : form.seed.trim(),
        loraStyle: form.loraStyle,
        loraStrength: mainLoraStrength,
      },
    });
    setMetrics({
      cpu_percent: response.cpu_usage,
      memory_percent: Number(((response.ram_used / response.ram_total) * 100).toFixed(1)),
      memory_used_mb: response.ram_used,
      memory_available_mb: Number((response.ram_total - response.ram_used).toFixed(1)),
    });
  }

  async function handleInpaintSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!inpaintSource) {
      return;
    }

    setInpaintLoading(true);
    setInpaintError(null);

    try {
      const response = await submitInpaintRequest(inpaintSource, inpaintForm);
      setInpaintResult({ response, original: inpaintSource });
      setShowSideBySide(true);
      setMetrics({
        cpu_percent: response.cpu_usage,
        memory_percent: Number(((response.ram_used / response.ram_total) * 100).toFixed(1)),
        memory_used_mb: response.ram_used,
        memory_available_mb: Number((response.ram_total - response.ram_used).toFixed(1)),
      });
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : "Unable to generate inpaint result.";
      setInpaintError(message);
    } finally {
      setInpaintLoading(false);
    }
  }

  async function submitInpaintRequest(source: EditableGeneratedImage, currentForm: InpaintFormState) {
    if (!currentForm.loraStyle) {
      throw new Error("Select an art style before generating.");
    }

    const initFile = source.file ?? (await fetchImageAsFile(source.src, source.name));
    const maskFile = await exportMaskAsFile();

    const payload = new FormData();
    payload.append("init_image", initFile);
    payload.append("mask_image", maskFile);
    payload.append("positive_prompt", currentForm.positivePrompt.trim());
    payload.append("negative_prompt", currentForm.negativePrompt.trim());
    payload.append("width", String(Number(currentForm.width)));
    payload.append("height", String(Number(currentForm.height)));
    payload.append("steps", String(Number(currentForm.steps)));
    payload.append("cfg_scale", String(Number(currentForm.cfgScale)));
    payload.append("denoise_strength", String(Number(currentForm.denoiseStrength)));
    payload.append("seed", currentForm.seed.trim() === "" ? "-1" : String(Number(currentForm.seed)));
    payload.append("lora_style", currentForm.loraStyle);
    payload.append(
      "lora_strength",
      String(currentForm === uploadInpaintForm ? uploadInpaintLoraStrength : inpaintLoraStrength),
    );

    return generateInpaintImage(payload);
  }

  async function fetchImageAsFile(url: string, filename: string): Promise<File> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error("Failed to load the source image for inpainting.");
    }
    const blob = await response.blob();
    return new File([blob], filename, { type: blob.type || "image/png" });
  }

  async function exportMaskAsFile(): Promise<File> {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) {
      throw new Error("Mask canvas is not ready.");
    }

    const blob = await new Promise<Blob | null>((resolve) => {
      maskCanvas.toBlob(resolve, "image/png");
    });
    if (!blob) {
      throw new Error("Failed to export the inpaint mask.");
    }
    return new File([blob], "inpaint-mask.png", { type: "image/png" });
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
              context.save();
              context.translate(component.x + component.width / 2, component.y + component.height / 2);
              context.rotate((component.rotation * Math.PI) / 180);
              context.drawImage(
                image,
                -component.width / 2,
                -component.height / 2,
                component.width,
                component.height,
              );
              context.restore();
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

  async function submitIPAdapterRequest() {
    if (!form.loraStyle) {
      throw new Error("Select an art style before generating.");
    }

    const payload = new FormData();
    for (const referenceImage of ipAdapterReferenceImages) {
      payload.append("reference_images", referenceImage.file);
    }
    payload.append("positive_prompt", form.positivePrompt.trim());
    payload.append("negative_prompt", form.negativePrompt.trim());
    payload.append("ip_adapter_scale", String(Number(form.ipAdapterScale)));
    payload.append("width", String(Number(form.width)));
    payload.append("height", String(Number(form.height)));
    payload.append("steps", String(Number(form.steps)));
    payload.append("cfg_scale", String(Number(form.cfgScale)));
    payload.append("denoise_strength", String(Number(form.denoiseStrength)));
    payload.append("seed", form.seed.trim() === "" ? "-1" : String(Number(form.seed)));
    payload.append("lora_style", form.loraStyle);
    payload.append("lora_strength", String(mainLoraStrength));

    return generateIPAdapterImage(payload);
  }

  function handleControlNetUpload(event: ChangeEvent<HTMLInputElement>) {
    setUploadedImage(event.target.files?.[0] ?? null);
  }

  function handleIPAdapterUpload(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    if (files.length === 0) {
      return;
    }

    void Promise.all(files.map((file) => loadReferenceImage(file))).then(
      (loadedImages) => {
        setIpAdapterReferenceImages((current) => [...current, ...loadedImages]);
        setIpAdapterResult(null);
      },
      (loadError) => {
        const message = loadError instanceof Error ? loadError.message : "Failed to load one or more reference images.";
        setError(message);
      },
    );

    event.target.value = "";
  }

  function handleUploadInpaintImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (!file) {
      return;
    }

    void loadEditableImageFromFile(file, uploadInpaintForm).then(
      (source) => {
        setUploadInpaintSource(source);
        setUploadInpaintForm((current) => ({
          ...current,
          width: source.width,
          height: source.height,
        }));
        setUploadInpaintResult(null);
        setUploadInpaintError(null);
        setShowUploadSideBySide(true);
      },
      (loadError) => {
        const message = loadError instanceof Error ? loadError.message : "Failed to load the uploaded image.";
        setUploadInpaintError(message);
      },
    );

    event.target.value = "";
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

  function loadReferenceImage(file: File): Promise<ReferenceImage> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const src = reader.result;
        if (typeof src !== "string") {
          reject(new Error(`Failed to read ${file.name}.`));
          return;
        }

        resolve({
          id: `${file.name}-${crypto.randomUUID()}`,
          name: file.name,
          src,
          file,
        });
      };
      reader.onerror = () => reject(new Error(`Failed to read ${file.name}.`));
      reader.readAsDataURL(file);
    });
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
            rotation: 0,
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
    const defaultStyle = loraStyles[0];
    setIsLoraEnabled(true);
    setForm({
      ...createInitialFormState(),
      loraStyle: defaultStyle?.key ?? "",
      loraStrength: defaultStyle?.default_strength ?? 1.0,
    });
    setUploadInpaintForm(createInitialInpaintState(null, defaultStyle));
    setUploadedImage(null);
    setUploadInpaintSource(null);
    setUploadInpaintResult(null);
    setUploadInpaintError(null);
    setShowUploadSideBySide(true);
    setIpAdapterReferenceImages([]);
    setIpAdapterResult(null);
    setCanvasComponents([]);
    setActiveComponentId(null);
  }

  function removeIPAdapterReference(id: string) {
    setIpAdapterReferenceImages((current) => current.filter((image) => image.id !== id));
    setIpAdapterResult(null);
  }

  function handleLoraStyleChange(styleKey: string) {
    const selectedStyle = loraStyles.find((style) => style.key === styleKey);
    if (!selectedStyle) {
      return;
    }

    setForm((current) => ({
      ...current,
      loraStyle: selectedStyle.key,
      loraStrength: selectedStyle.default_strength,
    }));
  }

  function handleInpaintStyleChange(styleKey: string) {
    const selectedStyle = loraStyles.find((style) => style.key === styleKey);
    if (!selectedStyle) {
      return;
    }

    setInpaintForm((current) => ({
      ...current,
      loraStyle: selectedStyle.key,
      loraStrength: selectedStyle.default_strength,
    }));
  }

  function handleUploadInpaintStyleChange(styleKey: string) {
    const selectedStyle = loraStyles.find((style) => style.key === styleKey);
    if (!selectedStyle) {
      return;
    }

    setUploadInpaintForm((current) => ({
      ...current,
      loraStyle: selectedStyle.key,
      loraStrength: selectedStyle.default_strength,
    }));
  }

  function openInpaintEditor(source: EditableGeneratedImage) {
    const defaultStyle = loraStyles.find((style) => style.key === source.loraStyle) ?? loraStyles[0];
    setInpaintSource(source);
    setInpaintForm(createInitialInpaintState(source, defaultStyle));
    setInpaintResult(null);
    setInpaintError(null);
    setShowSideBySide(true);
  }

  async function loadEditableImageFromFile(
    file: File,
    defaults: Pick<
      EditableGeneratedImage,
      "positivePrompt" | "negativePrompt" | "steps" | "cfgScale" | "denoiseStrength" | "seed" | "loraStyle" | "loraStrength"
    >,
  ): Promise<EditableGeneratedImage> {
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
          resolve({
            src,
            name: file.name,
            width: image.width,
            height: image.height,
            positivePrompt: defaults.positivePrompt,
            negativePrompt: defaults.negativePrompt,
            steps: defaults.steps,
            cfgScale: defaults.cfgScale,
            denoiseStrength: defaults.denoiseStrength,
            seed: defaults.seed,
            loraStyle: defaults.loraStyle,
            loraStrength: defaults.loraStrength,
            file,
          });
        };
        image.onerror = () => reject(new Error(`Failed to load ${file.name}.`));
        image.src = src;
      };
      reader.onerror = () => reject(new Error(`Failed to read ${file.name}.`));
      reader.readAsDataURL(file);
    });
  }

  function initializeMaskCanvases() {
    const overlayCanvas = overlayCanvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    if (!overlayCanvas || !maskCanvas) {
      return;
    }

    overlayCanvas.width = Number(activeInpaintForm.width);
    overlayCanvas.height = Number(activeInpaintForm.height);
    maskCanvas.width = Number(activeInpaintForm.width);
    maskCanvas.height = Number(activeInpaintForm.height);

    const overlayContext = overlayCanvas.getContext("2d");
    const maskContext = maskCanvas.getContext("2d");
    if (!overlayContext || !maskContext) {
      return;
    }

    overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    maskContext.fillStyle = "rgb(0, 0, 0)";
    maskContext.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
  }

  function clearMask() {
    initializeMaskCanvases();
  }

  function beginMaskStroke(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!overlayCanvasRef.current || !maskCanvasRef.current) {
      return;
    }

    const point = getCanvasPoint(event, overlayCanvasRef.current, activeInpaintForm.width, activeInpaintForm.height);
    const overlayContext = overlayCanvasRef.current.getContext("2d");
    const maskContext = maskCanvasRef.current.getContext("2d");
    if (!overlayContext || !maskContext) {
      return;
    }

    setIsMaskDrawing(true);
    overlayCanvasRef.current.setPointerCapture(event.pointerId);

    overlayContext.beginPath();
    overlayContext.moveTo(point.x, point.y);
    overlayContext.lineWidth = activeInpaintForm.brushSize;
    overlayContext.lineCap = "round";
    overlayContext.lineJoin = "round";
    overlayContext.globalCompositeOperation = activeInpaintForm.eraseMode ? "destination-out" : "source-over";
    overlayContext.strokeStyle = "rgba(255, 0, 0, 0.5)";

    maskContext.beginPath();
    maskContext.moveTo(point.x, point.y);
    maskContext.lineWidth = activeInpaintForm.brushSize;
    maskContext.lineCap = "round";
    maskContext.lineJoin = "round";
    maskContext.globalCompositeOperation = "source-over";
    maskContext.strokeStyle = activeInpaintForm.eraseMode ? "rgb(0, 0, 0)" : "rgb(255, 255, 255)";

    drawMaskStroke(point.x, point.y);
  }

  function continueMaskStroke(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!isMaskDrawing || !overlayCanvasRef.current) {
      return;
    }

    const point = getCanvasPoint(event, overlayCanvasRef.current, activeInpaintForm.width, activeInpaintForm.height);
    drawMaskStroke(point.x, point.y);
  }

  function endMaskStroke(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!overlayCanvasRef.current || !maskCanvasRef.current) {
      return;
    }

    setIsMaskDrawing(false);
    overlayCanvasRef.current.releasePointerCapture(event.pointerId);
    overlayCanvasRef.current.getContext("2d")?.closePath();
    maskCanvasRef.current.getContext("2d")?.closePath();
  }

  function drawMaskStroke(x: number, y: number) {
    const overlayContext = overlayCanvasRef.current?.getContext("2d");
    const maskContext = maskCanvasRef.current?.getContext("2d");
    if (!overlayContext || !maskContext) {
      return;
    }

    overlayContext.lineTo(x, y);
    overlayContext.stroke();
    maskContext.lineTo(x, y);
    maskContext.stroke();
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

  function changeComponentLayer(id: string, action: "front" | "forward" | "backward" | "back") {
    setCanvasComponents((current) => {
      if (current.length === 0) {
        return current;
      }

      const zIndexes = current.map((component) => component.zIndex);
      const highestZIndex = Math.max(...zIndexes);
      const lowestZIndex = Math.min(...zIndexes);

      return current.map((component) => {
        if (component.id !== id) {
          return component;
        }

        if (action === "front") {
          return { ...component, zIndex: highestZIndex + 1 };
        }
        if (action === "back") {
          return { ...component, zIndex: lowestZIndex - 1 };
        }
        if (action === "forward") {
          return { ...component, zIndex: component.zIndex + 1 };
        }
        return { ...component, zIndex: component.zIndex - 1 };
      });
    });
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

    if (dragState.type === "rotate") {
      const rawDegrees = (Math.atan2(pointerY - dragState.centerY, pointerX - dragState.centerX) * 180) / Math.PI + 90;
      const normalizedDegrees = ((rawDegrees % 360) + 360) % 360;
      const nextRotation = event.shiftKey ? Math.round(normalizedDegrees / 45) * 45 : normalizedDegrees;
      setCanvasComponents((current) =>
        current.map((component) =>
          component.id === dragState.id ? { ...component, rotation: Math.round(nextRotation) % 360 } : component,
        ),
      );
      return;
    }

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

        const stageWidth = Number(form.width);
        const stageHeight = Number(form.height);
        const deltaX = pointerX - dragState.startX;
        const deltaY = pointerY - dragState.startY;
        const rawWidth =
          dragState.corner === "bottom-left" ? dragState.startWidth - deltaX : dragState.startWidth + deltaX;
        const maxWidth =
          dragState.corner === "bottom-left"
            ? dragState.startComponentX + dragState.startWidth
            : stageWidth - dragState.startComponentX;
        const nextWidth = clamp(rawWidth, MIN_COMPONENT_SIZE, maxWidth);
        const nextHeight = dragState.preserveAspectRatio
          ? clamp(Math.round(nextWidth * dragState.aspectRatio), MIN_COMPONENT_SIZE, stageHeight - dragState.startComponentY)
          : clamp(dragState.startHeight + deltaY, MIN_COMPONENT_SIZE, stageHeight - dragState.startComponentY);
        const nextX =
          dragState.corner === "bottom-left"
            ? clamp(dragState.startComponentX + dragState.startWidth - nextWidth, 0, stageWidth - nextWidth)
            : dragState.startComponentX;

        return {
          ...component,
          x: nextX,
          width: nextWidth,
          height: nextHeight,
        };
      }),
    );
  }

  function handleStagePointerUp() {
    setDragState(null);
  }

  function startMove(event: ReactPointerEvent<HTMLDivElement>, component: CanvasComponent) {
    event.stopPropagation();
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

  function startResize(event: ReactPointerEvent<HTMLButtonElement>, component: CanvasComponent, corner: "bottom-left" | "bottom-right") {
    event.stopPropagation();
    bringComponentToFront(component.id);
    setDragState({
      type: "resize",
      id: component.id,
      pointerId: event.pointerId,
      corner,
      startX: corner === "bottom-left" ? component.x : component.x + component.width,
      startY: component.y + component.height,
      startComponentX: component.x,
      startComponentY: component.y,
      startWidth: component.width,
      startHeight: component.height,
      aspectRatio: component.naturalHeight / component.naturalWidth,
      preserveAspectRatio: !event.shiftKey,
    });
  }

  function startRotate(event: ReactPointerEvent<HTMLButtonElement>, component: CanvasComponent) {
    event.stopPropagation();
    bringComponentToFront(component.id);
    setDragState({
      type: "rotate",
      id: component.id,
      pointerId: event.pointerId,
      centerX: component.x + component.width / 2,
      centerY: component.y + component.height / 2,
      snapToAngles: event.shiftKey,
    });
  }

  function updateComponentScale(id: string, scalePercent: number) {
    const safeScale = clamp(scalePercent, 10, 500) / 100;
    setCanvasComponents((current) =>
      current.map((component) => {
        if (component.id !== id) {
          return component;
        }

        const nextWidth = clamp(Math.round(component.naturalWidth * safeScale), MIN_COMPONENT_SIZE, Number(form.width));
        const nextHeight = clamp(Math.round(component.naturalHeight * safeScale), MIN_COMPONENT_SIZE, Number(form.height));
        return {
          ...component,
          width: nextWidth,
          height: nextHeight,
          x: clamp(component.x, 0, Math.max(0, Number(form.width) - nextWidth)),
          y: clamp(component.y, 0, Math.max(0, Number(form.height) - nextHeight)),
        };
      }),
    );
  }

  function updateComponentRotation(id: string, degrees: number) {
    const nextDegrees = Number.isFinite(degrees) ? Math.round(degrees) : 0;
    setCanvasComponents((current) =>
      current.map((component) =>
        component.id === id ? { ...component, rotation: nextDegrees } : component,
      ),
    );
  }

  const isControlGuidedMode = mode === "controlnet" || mode === "canvas";
  const isUploadInpaintMode = mode === "upload-inpaint";
  const isIPAdapterMode = mode === "ip-adapter";
  const selectedCanvasComponent = canvasComponents.find((component) => component.id === activeComponentId) ?? null;
  const canvasToolbarWidth = 360;
  const canvasToolbarHeight = 52;
  const canvasToolbarLeft = selectedCanvasComponent
    ? clamp(
        selectedCanvasComponent.x + selectedCanvasComponent.width / 2 - canvasToolbarWidth / 2,
        8,
        Math.max(8, Number(form.width) - canvasToolbarWidth - 8),
      )
    : 0;
  const canvasToolbarTop = selectedCanvasComponent
    ? clamp(
        selectedCanvasComponent.y > canvasToolbarHeight + 20
          ? selectedCanvasComponent.y - canvasToolbarHeight - 12
          : selectedCanvasComponent.y + selectedCanvasComponent.height + 12,
        8,
        Math.max(8, Number(form.height) - canvasToolbarHeight - 8),
      )
    : 0;
  const selectedCanvasScale = selectedCanvasComponent
    ? Math.round((selectedCanvasComponent.width / selectedCanvasComponent.naturalWidth) * 100)
    : 100;
  const isGenerateDisabled =
    loading ||
    loraStyles.length === 0 ||
    (isUploadInpaintMode && !uploadInpaintSource) ||
    (isIPAdapterMode && ipAdapterReferenceImages.length === 0);
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
                prompt controls, ControlNet guidance, freeform canvas composition, direct upload inpainting, IP-Adapter references, and post-generation edits.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <MetricCard label="Scheduler" value="DPM++ 2M" detail="Karras sigmas" accent="orange" />
              <MetricCard label="Device" value="CPU" detail="GPU disabled" accent="green" />
              <MetricCard label="Modes" value="5" detail="Normal, Lineart, Canvas, Upload, IP-Adapter" accent="slate" />
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
                  Switch between standard generation, direct lineart upload, canvas-driven composition, uploaded-image inpainting, and IP-Adapter references.
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
                <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-5">
                  <ModeButton active={mode === "normal"} label="Normal" onClick={() => setMode("normal")} />
                  <ModeButton
                    active={mode === "controlnet"}
                    label="ControlNet Lineart"
                    onClick={() => setMode("controlnet")}
                  />
                  <ModeButton active={mode === "canvas"} label="Canvas Compose" onClick={() => setMode("canvas")} />
                  <ModeButton
                    active={mode === "upload-inpaint"}
                    label="Upload & Inpaint"
                    onClick={() => setMode("upload-inpaint")}
                  />
                  <ModeButton active={mode === "ip-adapter"} label="IP-Adapter" onClick={() => setMode("ip-adapter")} />
                </div>
              </div>

              {isIPAdapterMode ? (
                <FieldShell
                  label="Upload Reference Images"
                  helper="Upload one or more PNG, JPG, or WEBP files. Add more at any time."
                >
                  <div className="space-y-4 rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
                    <label className="flex cursor-pointer items-center justify-between rounded-[20px] border border-dashed border-white/15 bg-white/[0.03] px-4 py-4 transition hover:border-orange-400/35 hover:bg-white/[0.06]">
                      <div>
                        <div className="font-display text-lg font-semibold text-white">Upload Reference Images</div>
                        <div className="mt-1 text-sm text-slate-400">PNG, JPG, or WEBP. Multiple files supported.</div>
                      </div>
                      <div className="rounded-full bg-orange-500 px-4 py-2 text-sm font-semibold text-slate-950">
                        {ipAdapterReferenceImages.length > 0 ? "Upload More" : "Choose"}
                      </div>
                      <input
                        type="file"
                        accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
                        multiple
                        className="hidden"
                        onChange={handleIPAdapterUpload}
                      />
                    </label>

                    {ipAdapterReferenceImages.length > 0 ? (
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                        {ipAdapterReferenceImages.map((referenceImage) => (
                          <div
                            key={referenceImage.id}
                            className="group relative overflow-hidden rounded-[18px] border border-white/10 bg-white/5"
                          >
                            <img
                              src={referenceImage.src}
                              alt={referenceImage.name}
                              className="h-24 w-full object-cover"
                            />
                            <button
                              type="button"
                              onClick={() => removeIPAdapterReference(referenceImage.id)}
                              className="absolute right-2 top-2 inline-flex h-7 w-7 items-center justify-center rounded-full bg-slate-950/85 text-xs font-semibold text-white shadow-lg"
                            >
                              X
                            </button>
                            <div className="truncate px-3 py-2 text-xs text-slate-300">{referenceImage.name}</div>
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </FieldShell>
              ) : null}

              {isUploadInpaintMode ? (
                <FieldShell
                  label="Upload Image"
                  helper="Accepts PNG, JPG, or WEBP. Generation stays disabled until an image is loaded."
                >
                  <div className="space-y-4 rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
                    <label className="flex cursor-pointer items-center justify-between rounded-[20px] border border-dashed border-white/15 bg-white/[0.03] px-4 py-4 transition hover:border-orange-400/35 hover:bg-white/[0.06]">
                      <div>
                        <div className="font-display text-lg font-semibold text-white">Upload Image</div>
                        <div className="mt-1 text-sm text-slate-400">PNG, JPG, or WEBP</div>
                      </div>
                      <div className="rounded-full bg-orange-500 px-4 py-2 text-sm font-semibold text-slate-950">Choose</div>
                      <input
                        type="file"
                        accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
                        className="hidden"
                        onChange={handleUploadInpaintImage}
                      />
                    </label>

                    {uploadInpaintSource ? (
                      <div className="flex items-center gap-4 rounded-[20px] border border-white/10 bg-white/5 p-3">
                        <img
                          src={uploadInpaintSource.src}
                          alt={uploadInpaintSource.name}
                          className="h-16 w-16 rounded-xl border border-white/10 object-cover"
                        />
                        <div className="min-w-0">
                          <div className="truncate text-sm font-semibold text-white">{uploadInpaintSource.name}</div>
                          <div className="mt-1 text-xs text-slate-400">
                            {uploadInpaintSource.width} x {uploadInpaintSource.height}px
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </div>
                </FieldShell>
              ) : null}

              <FieldShell
                label="Positive Prompt"
                helper={
                  isUploadInpaintMode
                    ? "Describe the edit you want inside the masked region."
                    : isIPAdapterMode
                      ? "Describe the scene, style, and composition. Do not describe the character — IP-Adapter handles that from your reference images."
                    : "Describe the subject, style, composition, lighting, and desired details."
                }
              >
                <textarea
                  value={isUploadInpaintMode ? uploadInpaintForm.positivePrompt : form.positivePrompt}
                  onChange={(event) =>
                    isUploadInpaintMode
                      ? updateUploadInpaintField("positivePrompt", event.target.value)
                      : updateField("positivePrompt", event.target.value)
                  }
                  placeholder={
                    isUploadInpaintMode
                      ? "Replace the masked area with a detailed comic-style skyline..."
                      : isIPAdapterMode
                        ? "Describe the scene, style, and composition. Do not describe the character — IP-Adapter handles that from your reference images."
                      : "A cinematic comic-book hero portrait, dramatic rim light, richly inked detail..."
                  }
                  rows={5}
                  required
                  className="w-full resize-none rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-orange-400/50 focus:bg-white/[0.07]"
                />
              </FieldShell>

              <FieldShell
                label="Negative Prompt"
                helper={
                  isUploadInpaintMode
                    ? "Optional constraints for the regenerated region."
                    : "Suppress unwanted artifacts, styles, or visual defects."
                }
              >
                <textarea
                  value={isUploadInpaintMode ? uploadInpaintForm.negativePrompt : form.negativePrompt}
                  onChange={(event) =>
                    isUploadInpaintMode
                      ? updateUploadInpaintField("negativePrompt", event.target.value)
                      : updateField("negativePrompt", event.target.value)
                  }
                  placeholder="blurry, low quality, distorted face, extra fingers..."
                  rows={4}
                  className="w-full resize-none rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-orange-400/50 focus:bg-white/[0.07]"
                />
              </FieldShell>

              <div className="flex items-center justify-between rounded-[20px] border border-white/10 bg-white/[0.04] px-4 py-3">
                <div>
                  <div className="text-sm font-semibold text-white">Art Style (LoRA)</div>
                  <div className="mt-1 text-xs text-slate-400">
                    {isLoraEnabled ? "LoRA will be applied during generation." : "LoRA is disabled for generation."}
                  </div>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={isLoraEnabled}
                  onClick={() => setIsLoraEnabled((current) => !current)}
                  className={`relative h-8 w-14 rounded-full border transition ${
                    isLoraEnabled
                      ? "border-orange-300/50 bg-orange-500"
                      : "border-white/10 bg-white/10"
                  }`}
                >
                  <span
                    className={`absolute top-1 h-6 w-6 rounded-full bg-white shadow transition ${
                      isLoraEnabled ? "left-7" : "left-1"
                    }`}
                  />
                </button>
              </div>

              <div className={`space-y-5 transition-opacity ${loraControlOpacity}`}>
                <FieldShell
                  label="Art Style"
                  helper={
                    loraStyles.length > 0
                      ? "Required. Styles are loaded dynamically from the backend."
                      : "Loading styles from the backend."
                  }
                >
                  <select
                    value={isUploadInpaintMode ? uploadInpaintForm.loraStyle : form.loraStyle}
                    onChange={(event) =>
                      isUploadInpaintMode
                        ? handleUploadInpaintStyleChange(event.target.value)
                        : handleLoraStyleChange(event.target.value)
                    }
                    disabled={loraStyles.length === 0}
                    className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition focus:border-orange-400/50 focus:bg-white/[0.07] disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {loraStyles.map((style) => (
                      <option key={style.key} value={style.key} className="bg-slate-900 text-white">
                        {style.label}
                      </option>
                    ))}
                  </select>
                </FieldShell>

                <FieldShell
                  label="Style Strength"
                  helper={`${
                    (isUploadInpaintMode ? uploadInpaintForm.loraStrength : form.loraStrength).toFixed(2)
                  } between 0.1 and 2.0`}
                >
                  <input
                    type="range"
                    min={0.1}
                    max={2.0}
                    step={0.05}
                    value={isUploadInpaintMode ? uploadInpaintForm.loraStrength : form.loraStrength}
                    onChange={(event) =>
                      isUploadInpaintMode
                        ? updateUploadInpaintField("loraStrength", Number(event.target.value))
                        : updateField("loraStrength", Number(event.target.value))
                    }
                    className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-orange-500"
                  />
                </FieldShell>
              </div>

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
                    <input type="file" accept="image/*" className="hidden" onChange={handleControlNetUpload} />
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
                      <div className="font-display text-lg font-semibold text-white">Add Component to Canvas</div>
                      <div className="mt-1 text-sm text-slate-400">
                        {canvasComponents.length} component{canvasComponents.length === 1 ? "" : "s"} on canvas
                      </div>
                    </div>
                    <div className="rounded-full bg-orange-500 px-4 py-2 text-sm font-semibold text-slate-950">Add</div>
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
                    onChange={(event) => updateField("controlnetConditioningScale", Number(event.target.value))}
                    className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-orange-500"
                  />
                </FieldShell>
              ) : null}

              {isIPAdapterMode ? (
                <FieldShell label="Reference Strength" helper={`${form.ipAdapterScale.toFixed(2)} between 0.1 and 1.0`}>
                  <input
                    type="range"
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    value={form.ipAdapterScale}
                    onChange={(event) => updateField("ipAdapterScale", Number(event.target.value))}
                    className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-orange-500"
                  />
                </FieldShell>
              ) : null}

              <div className="grid gap-4 sm:grid-cols-2">
                <NumberField
                  label={isUploadInpaintMode ? "Width (optional)" : "Width"}
                  helper={isUploadInpaintMode ? "Prefilled from the uploaded image" : "Default 512"}
                  min={64}
                  max={isUploadInpaintMode ? 4096 : 1024}
                  step={8}
                  value={isUploadInpaintMode ? uploadInpaintForm.width : form.width}
                  onChange={(value) =>
                    isUploadInpaintMode ? updateUploadInpaintField("width", value) : updateField("width", value)
                  }
                />
                <NumberField
                  label={isUploadInpaintMode ? "Height (optional)" : "Height"}
                  helper={isUploadInpaintMode ? "Prefilled from the uploaded image" : "Default 512"}
                  min={64}
                  max={isUploadInpaintMode ? 4096 : 1024}
                  step={8}
                  value={isUploadInpaintMode ? uploadInpaintForm.height : form.height}
                  onChange={(value) =>
                    isUploadInpaintMode ? updateUploadInpaintField("height", value) : updateField("height", value)
                  }
                />
                <NumberField
                  label="Steps"
                  helper="Default 20"
                  min={1}
                  max={100}
                  step={1}
                  value={isUploadInpaintMode ? uploadInpaintForm.steps : form.steps}
                  onChange={(value) =>
                    isUploadInpaintMode ? updateUploadInpaintField("steps", value) : updateField("steps", value)
                  }
                />
                <NumberField
                  label="CFG Scale"
                  helper="Default 7.5"
                  min={1}
                  max={30}
                  step={0.5}
                  value={isUploadInpaintMode ? uploadInpaintForm.cfgScale : form.cfgScale}
                  onChange={(value) =>
                    isUploadInpaintMode ? updateUploadInpaintField("cfgScale", value) : updateField("cfgScale", value)
                  }
                />
                <NumberField
                  label="Denoise Strength"
                  helper="0.0 keeps more source structure, 1.0 changes more"
                  min={0}
                  max={1}
                  step={0.05}
                  value={isUploadInpaintMode ? uploadInpaintForm.denoiseStrength : form.denoiseStrength}
                  onChange={(value) =>
                    isUploadInpaintMode
                      ? updateUploadInpaintField("denoiseStrength", value)
                      : updateField("denoiseStrength", value)
                  }
                />
              </div>

              <FieldShell
                label="Seed"
                helper={
                  isUploadInpaintMode
                    ? "Defaults to -1 for random. Positive and negative values are accepted."
                    : "Leave blank for random. Positive and negative values are accepted."
                }
              >
                <input
                  value={isUploadInpaintMode ? uploadInpaintForm.seed : form.seed}
                  onChange={(event) =>
                    isUploadInpaintMode
                      ? updateUploadInpaintField("seed", event.target.value)
                      : updateField("seed", event.target.value)
                  }
                  placeholder="Random"
                  inputMode="numeric"
                  className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-orange-400/50 focus:bg-white/[0.07]"
                />
              </FieldShell>

              {isUploadInpaintMode ? (
                <>
                  <FieldShell
                    label="Brush Size"
                    helper={`${uploadInpaintForm.brushSize}px between 5 and 80`}
                  >
                    <input
                      type="range"
                      min={5}
                      max={80}
                      step={1}
                      value={uploadInpaintForm.brushSize}
                      onChange={(event) => updateUploadInpaintField("brushSize", Number(event.target.value))}
                      className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-orange-500"
                    />
                  </FieldShell>

                  <div className="flex flex-wrap gap-3">
                    <button
                      type="button"
                      onClick={() => updateUploadInpaintField("eraseMode", !uploadInpaintForm.eraseMode)}
                      className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                        uploadInpaintForm.eraseMode
                          ? "bg-white text-slate-950"
                          : "border border-white/10 bg-white/5 text-slate-200 hover:border-orange-400/30 hover:bg-white/10"
                      }`}
                    >
                      {uploadInpaintForm.eraseMode ? "Erase Mode" : "Draw Mode"}
                    </button>
                    <button
                      type="button"
                      onClick={clearMask}
                      className="rounded-full border border-white/10 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:border-red-400/30 hover:bg-white/5"
                    >
                      Clear Mask
                    </button>
                  </div>
                </>
              ) : null}
            </div>

            {error ? (
              <div className="mt-5 rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                {error}
              </div>
            ) : null}

            {isUploadInpaintMode && uploadInpaintError ? (
              <div className="mt-5 rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                {uploadInpaintError}
              </div>
            ) : null}

            <div className="mt-6 flex flex-col gap-3 border-t border-white/10 pt-5 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm text-slate-400">
                {isUploadInpaintMode
                  ? "CPU-only generation. Upload mode reuses the existing inpaint endpoint with the uploaded image and exported mask."
                  : isIPAdapterMode
                    ? "CPU-only generation. IP-Adapter mode sends all reference images as multipart form data to the local IP-Adapter endpoint."
                  : "CPU-only generation. Control-guided modes reuse the existing ControlNet endpoint and export a flat input image when needed."}
              </p>
              <button
                type="submit"
                disabled={isGenerateDisabled}
                className="inline-flex items-center justify-center rounded-full bg-orange-500 px-6 py-3 text-sm font-semibold text-slate-950 transition hover:bg-orange-400 disabled:cursor-not-allowed disabled:bg-orange-700"
              >
                {loading
                  ? "Generating..."
                  : mode === "normal"
                    ? "Generate Image"
                    : mode === "canvas"
                      ? "Generate from Canvas"
                      : mode === "ip-adapter"
                        ? "Generate with IP-Adapter"
                      : mode === "controlnet"
                        ? "Generate with ControlNet"
                        : "Generate"}
              </button>
            </div>
          </form>

          <div className="grid gap-6">
            <section className="rounded-[28px] border border-white/10 bg-[rgba(7,14,22,0.8)] p-5 shadow-panel backdrop-blur sm:p-6">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <h2 className="font-display text-2xl font-semibold text-white">
                    {mode === "canvas"
                      ? "Canvas Compose"
                      : mode === "upload-inpaint"
                        ? "Upload & Inpaint"
                        : mode === "ip-adapter"
                          ? "IP-Adapter Output"
                          : "Current Preview"}
                  </h2>
                  <p className="mt-1 text-sm text-slate-400">
                    {mode === "normal"
                      ? "The latest image saved in the root output folder."
                      : mode === "controlnet"
                        ? "Lineart preview and generated output for the latest ControlNet request."
                        : mode === "canvas"
                          ? "Arrange uploaded components, then export the canvas through the existing ControlNet pipeline."
                          : mode === "ip-adapter"
                            ? "Upload reference images and generate to see output here."
                          : "Upload an image, paint a white mask over areas to regenerate, and submit through the existing inpaint pipeline."}
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

              {mode === "ip-adapter" ? (
                ipAdapterResult ? (
                  <div className="space-y-4">
                    <PreviewPanel title="Generated Output" src={ipAdapterResult.response.image_url} alt="IP-Adapter generated output" />
                    <div className="grid gap-3 sm:grid-cols-2">
                      <InfoChip label="Seed" value={String(ipAdapterResult.response.seed_used)} />
                      <InfoChip label="Generation Time" value={`${ipAdapterResult.response.generation_time_seconds}s`} />
                      <InfoChip label="CPU Usage" value={`${ipAdapterResult.response.cpu_usage}%`} />
                      <InfoChip
                        label="RAM Usage"
                        value={`${Number(((ipAdapterResult.response.ram_used / ipAdapterResult.response.ram_total) * 100).toFixed(1))}%`}
                      />
                    </div>
                  </div>
                ) : (
                  <EmptyPreview message="Upload reference images and generate to see output here." />
                )
              ) : mode === "upload-inpaint" ? (
                <div className="space-y-4">
                  <div className="overflow-auto rounded-[24px] border border-white/10 bg-[linear-gradient(135deg,rgba(15,23,42,0.92),rgba(30,41,59,0.9))] p-3">
                    {uploadInpaintSource ? (
                      <div
                        className="relative mx-auto overflow-hidden rounded-[20px] border border-white/10 shadow-inner"
                        style={{
                          width: `${uploadInpaintForm.width}px`,
                          height: `${uploadInpaintForm.height}px`,
                        }}
                      >
                        <img
                          src={uploadInpaintSource.src}
                          alt={uploadInpaintSource.name}
                          className="absolute inset-0 h-full w-full object-fill"
                        />
                        <canvas
                          ref={overlayCanvasRef}
                          width={uploadInpaintForm.width}
                          height={uploadInpaintForm.height}
                          onPointerDown={beginMaskStroke}
                          onPointerMove={continueMaskStroke}
                          onPointerUp={endMaskStroke}
                          onPointerLeave={endMaskStroke}
                          className="absolute inset-0 h-full w-full touch-none cursor-crosshair"
                        />
                        <canvas
                          ref={maskCanvasRef}
                          width={uploadInpaintForm.width}
                          height={uploadInpaintForm.height}
                          className="hidden"
                        />
                      </div>
                    ) : (
                      <div className="flex min-h-[360px] items-center justify-center rounded-[20px] border-2 border-dashed border-white/15 px-8 text-center text-sm text-slate-400">
                        Upload an image to begin
                      </div>
                    )}
                  </div>

                  {uploadInpaintSource ? (
                    <div className="text-sm text-slate-400">
                      Canvas matches the current width and height settings: {uploadInpaintForm.width} x {uploadInpaintForm.height}
                    </div>
                  ) : null}

                  {uploadInpaintResult ? (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <h3 className="font-display text-xl font-semibold text-white">Inpaint Result</h3>
                        <button
                          type="button"
                          onClick={() => setShowUploadSideBySide((current) => !current)}
                          className="rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-orange-400/30 hover:bg-white/5 hover:text-white"
                        >
                          {showUploadSideBySide ? "Show Result Only" : "Show Side by Side"}
                        </button>
                      </div>

                      {showUploadSideBySide ? (
                        <div className="grid gap-4 md:grid-cols-2">
                          <PreviewPanel title="Original" src={uploadInpaintResult.original.src} alt="Original uploaded image" />
                          <PreviewPanel title="Inpainted Result" src={uploadInpaintResult.response.image_url} alt="Inpainted result" />
                        </div>
                      ) : (
                        <PreviewPanel title="Inpainted Result" src={uploadInpaintResult.response.image_url} alt="Inpainted result" />
                      )}

                      <div className="grid gap-3 sm:grid-cols-2">
                        <InfoChip label="Seed" value={String(uploadInpaintResult.response.seed_used)} />
                        <InfoChip label="Generation Time" value={`${uploadInpaintResult.response.generation_time_seconds}s`} />
                        <InfoChip label="CPU Usage" value={`${uploadInpaintResult.response.cpu_usage}%`} />
                        <InfoChip
                          label="RAM Usage"
                          value={`${Number(((uploadInpaintResult.response.ram_used / uploadInpaintResult.response.ram_total) * 100).toFixed(1))}%`}
                        />
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : mode === "canvas" ? (
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
                      onPointerDown={(event) => {
                        if (event.target === event.currentTarget) {
                          setActiveComponentId(null);
                        }
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
                            className={`absolute rounded-xl border ${
                              activeComponentId === component.id
                                ? "border-2 border-dashed border-orange-400 shadow-[0_0_0_1px_rgba(249,115,22,0.45)]"
                                : "border-white/10"
                            }`}
                            style={{
                              left: `${component.x}px`,
                              top: `${component.y}px`,
                              width: `${component.width}px`,
                              height: `${component.height}px`,
                              zIndex: component.zIndex,
                              transform: `rotate(${component.rotation}deg)`,
                              transformOrigin: "center",
                            }}
                            onPointerDown={(event) => startMove(event, component)}
                          >
                            <img
                              src={component.src}
                              alt={component.name}
                              draggable={false}
                              className="h-full w-full select-none rounded-xl object-cover"
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
                            {activeComponentId === component.id ? (
                              <>
                                <button
                                  type="button"
                                  aria-label="Rotate component"
                                  onPointerDown={(event) => startRotate(event, component)}
                                  className="absolute left-1/2 top-[-42px] h-7 w-7 -translate-x-1/2 cursor-grab rounded-full border-2 border-orange-300 bg-slate-950 shadow-lg"
                                />
                                <button
                                  type="button"
                                  aria-label="Resize from bottom-left"
                                  onPointerDown={(event) => startResize(event, component, "bottom-left")}
                                  className="absolute bottom-[-10px] left-[-10px] h-5 w-5 cursor-sw-resize rounded-md border border-white/40 bg-orange-500 shadow-lg"
                                />
                                <button
                                  type="button"
                                  aria-label="Resize from bottom-right"
                                  onPointerDown={(event) => startResize(event, component, "bottom-right")}
                                  className="absolute bottom-[-10px] right-[-10px] h-5 w-5 cursor-se-resize rounded-md border border-white/40 bg-orange-500 shadow-lg"
                                />
                              </>
                            ) : null}
                          </div>
                        ))}
                      {selectedCanvasComponent ? (
                        <div
                          className="absolute z-[9999] flex h-[52px] items-center gap-3 rounded-2xl border border-orange-400/30 bg-slate-950/95 px-3 py-2 text-xs text-slate-200 shadow-2xl backdrop-blur"
                          style={{
                            left: `${canvasToolbarLeft}px`,
                            top: `${canvasToolbarTop}px`,
                            width: `${canvasToolbarWidth}px`,
                          }}
                          onPointerDown={(event) => event.stopPropagation()}
                        >
                          <label className="flex items-center gap-2">
                            <span className="font-semibold text-white">Scale</span>
                            <input
                              type="number"
                              min={10}
                              max={500}
                              value={selectedCanvasScale}
                              onChange={(event) =>
                                updateComponentScale(selectedCanvasComponent.id, Number(event.target.value))
                              }
                              className="w-16 rounded-lg border border-white/10 bg-white/10 px-2 py-1 text-right text-white outline-none focus:border-orange-400/60"
                            />
                          </label>
                          <label className="flex items-center gap-2">
                            <span className="font-semibold text-white">Rotate</span>
                            <input
                              type="number"
                              value={selectedCanvasComponent.rotation}
                              onChange={(event) =>
                                updateComponentRotation(selectedCanvasComponent.id, Number(event.target.value))
                              }
                              className="w-16 rounded-lg border border-white/10 bg-white/10 px-2 py-1 text-right text-white outline-none focus:border-orange-400/60"
                            />
                          </label>
                          <div className="flex items-center gap-1">
                            <span className="mr-1 font-semibold text-white">Layer</span>
                            <button type="button" onClick={() => changeComponentLayer(selectedCanvasComponent.id, "front")} className="rounded-md bg-white/10 px-2 py-1 font-semibold text-white hover:bg-orange-500/80">↑↑</button>
                            <button type="button" onClick={() => changeComponentLayer(selectedCanvasComponent.id, "forward")} className="rounded-md bg-white/10 px-2 py-1 font-semibold text-white hover:bg-orange-500/80">↑</button>
                            <button type="button" onClick={() => changeComponentLayer(selectedCanvasComponent.id, "backward")} className="rounded-md bg-white/10 px-2 py-1 font-semibold text-white hover:bg-orange-500/80">↓</button>
                            <button type="button" onClick={() => changeComponentLayer(selectedCanvasComponent.id, "back")} className="rounded-md bg-white/10 px-2 py-1 font-semibold text-white hover:bg-orange-500/80">↓↓</button>
                          </div>
                        </div>
                      ) : null}
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
                    <>
                      <div className="grid gap-4 md:grid-cols-2">
                        <PreviewPanel title="Lineart Preview" src={controlNetResult.response.lineart_preview_url} alt="Preprocessed lineart preview" />
                        <PreviewPanel title="Generated Output" src={controlNetResult.response.image_url} alt="Canvas Compose generated output" />
                      </div>
                      <ActionButton label="Edit this image" onClick={() => openInpaintEditor(controlNetResult.editable)} />
                    </>
                  ) : null}
                </div>
              ) : isControlGuidedMode ? (
                controlNetResult ? (
                  <div className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <PreviewPanel title="Lineart Preview" src={controlNetResult.response.lineart_preview_url} alt="Preprocessed lineart preview" />
                      <PreviewPanel title="Generated Output" src={controlNetResult.response.image_url} alt="ControlNet generated output" />
                    </div>
                    <ActionButton label="Edit this image" onClick={() => openInpaintEditor(controlNetResult.editable)} />
                  </div>
                ) : (
                  <EmptyPreview message="Generate to see the lineart preview and final output here." />
                )
              ) : normalResult ? (
                <div className="space-y-4">
                  <div className="overflow-hidden rounded-[24px] border border-white/10 bg-[rgba(255,255,255,0.03)]">
                    <img src={normalResult.response.image.image_url} alt={normalResult.response.image.positive_prompt} className="aspect-square w-full object-cover" />
                  </div>
                  <ActionButton label="Edit this image" onClick={() => openInpaintEditor(normalResult.editable)} />
                </div>
              ) : (
                <EmptyPreview message="Generate an image to preview the latest output here." />
              )}

              {isControlGuidedMode && controlNetResult ? (
                <div className="mt-5 grid gap-3 sm:grid-cols-2">
                  <InfoChip label="Output File" value={controlNetResult.response.image_filename} />
                  <InfoChip label="Lineart File" value={controlNetResult.response.preprocessed_lineart_filename} />
                  <InfoChip label="Seed" value={String(controlNetResult.response.seed_used)} />
                  <InfoChip label="Generation Time" value={`${controlNetResult.response.generation_time_seconds}s`} />
                </div>
              ) : null}

              {mode === "normal" && normalResult ? (
                <div className="mt-5 grid gap-3 sm:grid-cols-2">
                  <InfoChip label="Filename" value={normalResult.response.image.filename} />
                  <InfoChip label="Seed" value={String(normalResult.response.image.seed)} />
                  <InfoChip label="Size" value={`${normalResult.response.image.width} x ${normalResult.response.image.height}`} />
                  <InfoChip label="Steps / CFG" value={`${normalResult.response.image.steps} / ${normalResult.response.image.cfg_scale}`} />
                </div>
              ) : null}
            </section>

            <section className="rounded-[28px] border border-white/10 bg-[rgba(7,14,22,0.8)] p-5 shadow-panel backdrop-blur sm:p-6">
              <h2 className="font-display text-2xl font-semibold text-white">System Usage</h2>
              <p className="mt-1 text-sm text-slate-400">
                CPU and RAM values from the backend after generation or manual refresh.
              </p>

              <div className="mt-5 grid gap-4 sm:grid-cols-2">
                <UsageCard label="CPU Usage" value={metrics ? `${metrics.cpu_percent}%` : "--"} detail="Snapshot after request" tone="orange" />
                <UsageCard label="RAM Usage" value={metrics ? `${metrics.memory_percent}%` : "--"} detail={metrics ? `${metrics.memory_used_mb} MB used` : "Snapshot unavailable"} tone="green" />
                <UsageCard label="RAM Available" value={metrics ? `${metrics.memory_available_mb} MB` : "--"} detail="Free system memory" tone="slate" />
                <UsageCard
                  label="Execution Mode"
                  value="CPU Only"
                  detail={
                    isUploadInpaintMode
                      ? "Upload mode reuses inpaint generation"
                      : isIPAdapterMode
                        ? "IP-Adapter mode reuses local SD1.5 generation"
                        : "Canvas mode reuses ControlNet generation"
                  }
                  tone="slate"
                />
              </div>
            </section>
          </div>
        </section>

        {inpaintSource ? (
          <section
            ref={inpaintPanelRef}
            className="rounded-[28px] border border-white/10 bg-[rgba(7,14,22,0.84)] p-5 shadow-panel backdrop-blur sm:p-6"
          >
            <div className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h2 className="font-display text-2xl font-semibold text-white">Inpainting Edit</h2>
                <p className="mt-1 text-sm text-slate-400">
                  Edit the latest generated image by painting a mask over the areas that should be regenerated.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setInpaintSource(null)}
                className="rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-orange-400/30 hover:bg-white/5 hover:text-white"
              >
                Close Editor
              </button>
            </div>

            <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
              <form onSubmit={handleInpaintSubmit} className="space-y-5">
                <FieldShell label="Current Image" helper="Source image selected from the generated output">
                  <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-200">
                    {inpaintSource.name}
                  </div>
                </FieldShell>

                <FieldShell label="Positive Prompt" helper="Describe the edit you want in the masked area.">
                  <textarea
                    value={inpaintForm.positivePrompt}
                    onChange={(event) => updateInpaintField("positivePrompt", event.target.value)}
                    rows={4}
                    required
                    className="w-full resize-none rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-orange-400/50 focus:bg-white/[0.07]"
                  />
                </FieldShell>

                <FieldShell label="Negative Prompt" helper="Optional constraints for the edited result.">
                  <textarea
                    value={inpaintForm.negativePrompt}
                    onChange={(event) => updateInpaintField("negativePrompt", event.target.value)}
                    rows={3}
                    className="w-full resize-none rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-orange-400/50 focus:bg-white/[0.07]"
                  />
                </FieldShell>

                <div className={`space-y-5 transition-opacity ${loraControlOpacity}`}>
                  <FieldShell label="Art Style" helper="Uses the same LoRA style catalog as the main generator.">
                    <select
                      value={inpaintForm.loraStyle}
                      onChange={(event) => handleInpaintStyleChange(event.target.value)}
                      className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition focus:border-orange-400/50 focus:bg-white/[0.07]"
                    >
                      {loraStyles.map((style) => (
                        <option key={style.key} value={style.key} className="bg-slate-900 text-white">
                          {style.label}
                        </option>
                      ))}
                    </select>
                  </FieldShell>

                  <FieldShell label="Style Strength" helper={`${inpaintForm.loraStrength.toFixed(2)} between 0.1 and 2.0`}>
                    <input
                      type="range"
                      min={0.1}
                      max={2.0}
                      step={0.05}
                      value={inpaintForm.loraStrength}
                      onChange={(event) => updateInpaintField("loraStrength", Number(event.target.value))}
                      className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-orange-500"
                    />
                  </FieldShell>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <NumberField label="Width" helper="Prefilled from the source image" min={64} max={1024} step={8} value={inpaintForm.width} onChange={(value) => updateInpaintField("width", value)} />
                  <NumberField label="Height" helper="Prefilled from the source image" min={64} max={1024} step={8} value={inpaintForm.height} onChange={(value) => updateInpaintField("height", value)} />
                  <NumberField label="Steps" helper="Default 20" min={1} max={100} step={1} value={inpaintForm.steps} onChange={(value) => updateInpaintField("steps", value)} />
                  <NumberField label="CFG Scale" helper="Default 7.5" min={1} max={30} step={0.5} value={inpaintForm.cfgScale} onChange={(value) => updateInpaintField("cfgScale", value)} />
                  <NumberField label="Denoise Strength" helper="0.0 keeps more source structure, 1.0 changes more" min={0} max={1} step={0.05} value={inpaintForm.denoiseStrength} onChange={(value) => updateInpaintField("denoiseStrength", value)} />
                </div>

                <FieldShell label="Seed" helper="Defaults to -1 for random.">
                  <input
                    value={inpaintForm.seed}
                    onChange={(event) => updateInpaintField("seed", event.target.value)}
                    inputMode="numeric"
                    className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-orange-400/50 focus:bg-white/[0.07]"
                  />
                </FieldShell>

                <FieldShell label="Brush Size" helper={`${inpaintForm.brushSize}px between 5 and 80`}>
                  <input
                    type="range"
                    min={5}
                    max={80}
                    step={1}
                    value={inpaintForm.brushSize}
                    onChange={(event) => updateInpaintField("brushSize", Number(event.target.value))}
                    className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-orange-500"
                  />
                </FieldShell>

                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => updateInpaintField("eraseMode", !inpaintForm.eraseMode)}
                    className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                      inpaintForm.eraseMode
                        ? "bg-white text-slate-950"
                        : "border border-white/10 bg-white/5 text-slate-200 hover:border-orange-400/30 hover:bg-white/10"
                    }`}
                  >
                    {inpaintForm.eraseMode ? "Erase Mode" : "Draw Mode"}
                  </button>
                  <button
                    type="button"
                    onClick={clearMask}
                    className="rounded-full border border-white/10 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:border-red-400/30 hover:bg-white/5"
                  >
                    Clear Mask
                  </button>
                </div>

                {inpaintError ? (
                  <div className="rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                    {inpaintError}
                  </div>
                ) : null}

                <button
                  type="submit"
                  disabled={inpaintLoading}
                  className="inline-flex items-center justify-center rounded-full bg-orange-500 px-6 py-3 text-sm font-semibold text-slate-950 transition hover:bg-orange-400 disabled:cursor-not-allowed disabled:bg-orange-700"
                >
                  {inpaintLoading ? "Generating..." : "Generate"}
                </button>
              </form>

              <div className="space-y-5">
                <div className="overflow-auto rounded-[24px] border border-white/10 bg-[linear-gradient(135deg,rgba(15,23,42,0.92),rgba(30,41,59,0.9))] p-3">
                  <div
                    className="relative mx-auto overflow-hidden rounded-[20px] border border-white/10 shadow-inner"
                    style={{ width: `${inpaintForm.width}px`, height: `${inpaintForm.height}px` }}
                  >
                    <img
                      src={inpaintSource.src}
                      alt={inpaintSource.name}
                      className="absolute inset-0 h-full w-full object-fill"
                    />
                    <canvas
                      ref={overlayCanvasRef}
                      width={inpaintForm.width}
                      height={inpaintForm.height}
                      onPointerDown={beginMaskStroke}
                      onPointerMove={continueMaskStroke}
                      onPointerUp={endMaskStroke}
                      onPointerLeave={endMaskStroke}
                      className="absolute inset-0 h-full w-full touch-none cursor-crosshair"
                    />
                    <canvas ref={maskCanvasRef} width={inpaintForm.width} height={inpaintForm.height} className="hidden" />
                  </div>
                </div>

                {inpaintResult ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="font-display text-xl font-semibold text-white">Inpaint Result</h3>
                      <button
                        type="button"
                        onClick={() => setShowSideBySide((current) => !current)}
                        className="rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-orange-400/30 hover:bg-white/5 hover:text-white"
                      >
                        {showSideBySide ? "Show Result Only" : "Show Side by Side"}
                      </button>
                    </div>

                    {showSideBySide ? (
                      <div className="grid gap-4 md:grid-cols-2">
                        <PreviewPanel title="Original" src={inpaintResult.original.src} alt="Original image selected for inpainting" />
                        <PreviewPanel title="Inpainted Result" src={inpaintResult.response.image_url} alt="Inpainted result" />
                      </div>
                    ) : (
                      <PreviewPanel title="Inpainted Result" src={inpaintResult.response.image_url} alt="Inpainted result" />
                    )}

                    <div className="grid gap-3 sm:grid-cols-2">
                      <InfoChip label="Seed" value={String(inpaintResult.response.seed_used)} />
                      <InfoChip label="Generation Time" value={`${inpaintResult.response.generation_time_seconds}s`} />
                      <InfoChip label="CPU Usage" value={`${inpaintResult.response.cpu_usage}%`} />
                      <InfoChip
                        label="RAM Usage"
                        value={`${Number(((inpaintResult.response.ram_used / inpaintResult.response.ram_total) * 100).toFixed(1))}%`}
                      />
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </section>
        ) : null}
      </div>
    </main>
  );
}

function getCanvasPoint(
  event: ReactPointerEvent<HTMLCanvasElement>,
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * width,
    y: ((event.clientY - rect.top) / rect.height) * height,
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function ActionButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center justify-center rounded-full bg-orange-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-orange-400"
    >
      {label}
    </button>
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
      <div className="border-b border-white/10 px-4 py-3 text-sm font-semibold text-slate-200">{title}</div>
      <img src={src} alt={alt} className="aspect-square w-full object-cover" />
    </div>
  );
}
