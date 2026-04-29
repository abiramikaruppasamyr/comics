import type { ChangeEvent, FormEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

import {
  generateControlNetImage,
  generateImage,
  generateIPAdapterImage,
  getLoraStyles,
  getSystemMetrics,
  inpaintImage,
} from "./services/api";
import type {
  ControlNetGenerateResponse,
  GenerateImageResponse,
  IPAdapterGenerateResponse,
  InpaintResponse,
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

type NormalResultState = {
  response: GenerateImageResponse;
};

type ControlResultState = {
  response: ControlNetGenerateResponse;
};

type IPAdapterResultState = {
  response: IPAdapterGenerateResponse;
};

type InpaintFormState = {
  prompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  steps: number;
  guidanceScale: number;
  strength: number;
  seed: string;
  brushSize: number;
};

type InpaintSourceState = {
  file: File;
  src: string;
  name: string;
  width: number;
  height: number;
};

type InpaintResultState = {
  response: InpaintResponse;
  imageUrl: string;
};

type InpaintMaskTool = "brush" | "lasso" | "rectangle";

type InpaintPointerState = {
  pointerId: number;
  erase: boolean;
  lastX: number;
  lastY: number;
  pendingDistance: number;
};

type InpaintSelectionState =
  | {
      type: "lasso";
      pointerId: number;
      points: Array<{ x: number; y: number }>;
    }
  | {
      type: "rectangle";
      pointerId: number;
      startX: number;
      startY: number;
      currentX: number;
      currentY: number;
    };

type InpaintCursorState = {
  x: number;
  y: number;
  visible: boolean;
};

const MIN_COMPONENT_SIZE = 32;
const DEFAULT_INPAINT_CANVAS_SIZE = 512;
const MASK_BRUSH_SPACING_RATIO = 0.03;
const MASK_BRUSH_FEATHER_PX = 30;
const MASK_PREVIEW_FEATHER_PX = 12;

function createInitialFormState(): FormState {
  return {
    positivePrompt: "",
    negativePrompt: "",
    width: 512,
    height: 512,
    steps: 20,
    cfgScale: 7.5,
    denoiseStrength: 0.5,
    seed: "",
    controlnetConditioningScale: 1.0,
    ipAdapterScale: 0.6,
    loraStyle: "",
    loraStrength: 1.0,
  };
}

function createInitialInpaintFormState(): InpaintFormState {
  return {
    prompt: "",
    negativePrompt: "",
    width: DEFAULT_INPAINT_CANVAS_SIZE,
    height: DEFAULT_INPAINT_CANVAS_SIZE,
    steps: 30,
    guidanceScale: 7.5,
    strength: 0.3,
    seed: "-1",
    brushSize: 36,
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
  const [error, setError] = useState<string | null>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [ipAdapterReferenceImages, setIpAdapterReferenceImages] = useState<ReferenceImage[]>([]);
  const [ipAdapterResult, setIpAdapterResult] = useState<IPAdapterResultState | null>(null);
  const [inpaintSource, setInpaintSource] = useState<InpaintSourceState | null>(null);
  const [inpaintControlSource, setInpaintControlSource] = useState<InpaintSourceState | null>(null);
  const [inpaintForm, setInpaintForm] = useState<InpaintFormState>(createInitialInpaintFormState);
  const [inpaintResult, setInpaintResult] = useState<InpaintResultState | null>(null);
  const [inpaintTool, setInpaintTool] = useState<InpaintMaskTool>("brush");
  const [inpaintPointer, setInpaintPointer] = useState<InpaintPointerState | null>(null);
  const [inpaintSelection, setInpaintSelection] = useState<InpaintSelectionState | null>(null);
  const [inpaintCursor, setInpaintCursor] = useState<InpaintCursorState>({ x: 0, y: 0, visible: false });
  const [isInpaintMaskInverted, setIsInpaintMaskInverted] = useState(false);
  const [isInpaintEraserEnabled, setIsInpaintEraserEnabled] = useState(false);
  const [isInpaintSubtractEnabled, setIsInpaintSubtractEnabled] = useState(false);
  const [inpaintMaskOpacity, setInpaintMaskOpacity] = useState(0.58);
  const [showInpaintOriginal, setShowInpaintOriginal] = useState(false);

  const canvasStageRef = useRef<HTMLDivElement | null>(null);
  const inpaintOverlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inpaintMaskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inpaintMaskPreviewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inpaintMaskBufferRef = useRef<Float32Array | null>(null);
  const inpaintUndoStackRef = useRef<Float32Array[]>([]);
  const mainLoraStrength = isLoraEnabled ? Number(form.loraStrength) : 0.0;
  const loraControlOpacity = isLoraEnabled ? "" : "opacity-50";

  useEffect(() => {
    void refreshMetrics();
  }, []);

  useEffect(() => {
    void loadLoraStyles();
  }, []);

  useEffect(() => {
    if (!inpaintSource) {
      return;
    }

    initializeInpaintMask();
  }, [inpaintSource, inpaintForm.width, inpaintForm.height]);

  useEffect(() => {
    renderInpaintPreview();
    renderInpaintMaskPreview();
  }, [inpaintMaskOpacity, isInpaintMaskInverted]);

  useEffect(() => {
    function handleInpaintUndo(event: KeyboardEvent) {
      if (mode !== "upload-inpaint" || !(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "z") {
        return;
      }

      event.preventDefault();
      undoInpaintMask();
    }

    window.addEventListener("keydown", handleInpaintUndo);
    return () => window.removeEventListener("keydown", handleInpaintUndo);
  }, [mode, inpaintMaskOpacity, isInpaintMaskInverted]);

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

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);

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
      } else if (mode === "upload-inpaint") {
        if (!inpaintSource) {
          throw new Error("Upload a base image before running inpaint.");
        }
        if (!hasInpaintMaskPainted()) {
          throw new Error("Paint a mask before running inpaint.");
        }
        const response = await submitInpaintRequest(inpaintSource);
        setInpaintResult({
          response,
          imageUrl: `data:image/png;base64,${response.image_base64}`,
        });
        setShowInpaintOriginal(false);
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
    });
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

  async function submitInpaintRequest(source: InpaintSourceState) {
    const width = Number(inpaintForm.width);
    const height = Number(inpaintForm.height);
    const maskBlob = await exportInpaintMaskBlob(width, height);
    const resizedImageFile = await resizeInpaintSourceFile(source, width, height);
    return inpaintImage(resizedImageFile, maskBlob, {
      prompt: inpaintForm.prompt.trim(),
      negative_prompt: inpaintForm.negativePrompt.trim(),
      control_image_file: inpaintControlSource?.file ?? null,
      width,
      height,
      steps: Number(inpaintForm.steps),
      guidance_scale: Number(inpaintForm.guidanceScale),
      strength: Number(inpaintForm.strength),
      seed: inpaintForm.seed.trim() === "" ? -1 : Number(inpaintForm.seed),
    });
  }

  async function resizeInpaintSourceFile(
    source: InpaintSourceState,
    width: number,
    height: number,
    filename = "inpaint-source.png",
  ): Promise<File> {
    const imageBitmap = await createImageBitmap(source.file);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) {
      imageBitmap.close();
      throw new Error("Failed to resize the inpaint source image.");
    }

    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);
    context.drawImage(imageBitmap, 0, 0, width, height);
    imageBitmap.close();

    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(resolve, "image/png");
    });
    if (!blob) {
      throw new Error("Failed to export resized inpaint source image.");
    }

    return new File([blob], filename, { type: "image/png" });
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

  function handleInpaintUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (!file) {
      return;
    }

    void loadInpaintSource(file).then(
      (source) => {
        setInpaintSource(source);
        setInpaintForm((current) => ({
          ...current,
          width: DEFAULT_INPAINT_CANVAS_SIZE,
          height: DEFAULT_INPAINT_CANVAS_SIZE,
        }));
        setInpaintResult(null);
        setError(null);
      },
      (loadError) => {
        const message = loadError instanceof Error ? loadError.message : "Failed to load the uploaded image.";
        setError(message);
      },
    );

    event.target.value = "";
  }

  function handleInpaintControlUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (!file) {
      return;
    }

    void loadInpaintSource(file).then(
      (source) => {
        setInpaintControlSource(source);
        setInpaintResult(null);
        setError(null);
      },
      (loadError) => {
        const message = loadError instanceof Error ? loadError.message : "Failed to load the ControlNet reference image.";
        setError(message);
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

  function loadInpaintSource(file: File): Promise<InpaintSourceState> {
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
            file,
            src,
            name: file.name,
            width: image.naturalWidth,
            height: image.naturalHeight,
          });
        };
        image.onerror = () => reject(new Error(`Failed to load ${file.name}.`));
        image.src = src;
      };
      reader.onerror = () => reject(new Error(`Failed to read ${file.name}.`));
      reader.readAsDataURL(file);
    });
  }

  function useInpaintResultAsBase() {
    if (!inpaintResult) {
      return;
    }

    const file = dataUrlToFile(inpaintResult.imageUrl, "inpaint-result.png");
    void loadInpaintSource(file).then(
      (source) => {
        setInpaintSource(source);
        setInpaintResult(null);
        setShowInpaintOriginal(false);
      },
      (loadError) => {
        const message = loadError instanceof Error ? loadError.message : "Failed to load the inpaint result.";
        setError(message);
      },
    );
  }

  function dataUrlToFile(dataUrl: string, filename: string) {
    const [metadata, base64Data] = dataUrl.split(",");
    const mimeMatch = metadata.match(/data:(.*?);base64/);
    const mimeType = mimeMatch?.[1] ?? "image/png";
    const binary = atob(base64Data);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return new File([bytes], filename, { type: mimeType });
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
    setUploadedImage(null);
    setIpAdapterReferenceImages([]);
    setIpAdapterResult(null);
    setInpaintSource(null);
    setInpaintControlSource(null);
    setInpaintForm(createInitialInpaintFormState());
    setInpaintResult(null);
    setInpaintPointer(null);
    setIsInpaintMaskInverted(false);
    setInpaintMaskOpacity(0.58);
    setShowInpaintOriginal(false);
    inpaintMaskBufferRef.current = null;
    inpaintUndoStackRef.current = [];
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

  function initializeInpaintMask() {
    if (!inpaintSource) {
      return;
    }

    const overlayCanvas = inpaintOverlayCanvasRef.current;
    const maskCanvas = inpaintMaskCanvasRef.current;
    if (!overlayCanvas || !maskCanvas) {
      return;
    }

    const canvasWidth = Number(inpaintForm.width);
    const canvasHeight = Number(inpaintForm.height);
    overlayCanvas.width = canvasWidth;
    overlayCanvas.height = canvasHeight;
    maskCanvas.width = canvasWidth;
    maskCanvas.height = canvasHeight;
    inpaintMaskBufferRef.current = new Float32Array(canvasWidth * canvasHeight);

    const overlayContext = overlayCanvas.getContext("2d");
    const maskContext = maskCanvas.getContext("2d");
    if (!overlayContext || !maskContext) {
      return;
    }

    overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    maskContext.fillStyle = "#000000";
    maskContext.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    inpaintUndoStackRef.current = [];
    renderInpaintPreview();
    renderInpaintMaskCanvas();
    renderInpaintMaskPreview();
  }

  function clearInpaintMask() {
    const overlayCanvas = inpaintOverlayCanvasRef.current;
    const maskCanvas = inpaintMaskCanvasRef.current;
    if (!overlayCanvas || !maskCanvas) {
      return;
    }

    const overlayContext = overlayCanvas.getContext("2d");
    const maskContext = maskCanvas.getContext("2d");
    if (!overlayContext || !maskContext) {
      return;
    }

    pushInpaintUndoState();
    overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    inpaintMaskBufferRef.current?.fill(0);
    maskContext.globalCompositeOperation = "source-over";
    maskContext.fillStyle = "#000000";
    maskContext.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    renderInpaintPreview();
    renderInpaintMaskCanvas();
    renderInpaintMaskPreview();
  }

  function beginInpaintCanvasPointer(event: ReactPointerEvent<HTMLCanvasElement>) {
    updateInpaintCursor(event);
    if (inpaintTool === "brush") {
      beginInpaintStroke(event);
      return;
    }

    beginInpaintSelection(event);
  }

  function continueInpaintCanvasPointer(event: ReactPointerEvent<HTMLCanvasElement>) {
    updateInpaintCursor(event);
    if (inpaintTool === "brush") {
      continueInpaintStroke(event);
      return;
    }

    continueInpaintSelection(event);
  }

  function endInpaintCanvasPointer(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (inpaintTool === "brush") {
      endInpaintStroke(event);
      return;
    }

    endInpaintSelection(event);
  }

  function leaveInpaintCanvas(event: ReactPointerEvent<HTMLCanvasElement>) {
    setInpaintCursor((current) => ({ ...current, visible: false }));
    if (inpaintTool === "brush") {
      endInpaintStroke(event);
    }
  }

  function beginInpaintStroke(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!inpaintSource || !inpaintOverlayCanvasRef.current || !inpaintMaskCanvasRef.current) {
      return;
    }

    event.preventDefault();
    const erase = event.button === 2 || isInpaintEraserEnabled;
    const point = getCanvasPoint(
      event,
      inpaintOverlayCanvasRef.current,
      inpaintOverlayCanvasRef.current.width,
      inpaintOverlayCanvasRef.current.height,
    );
    pushInpaintUndoState();
    inpaintOverlayCanvasRef.current.setPointerCapture(event.pointerId);
    setInpaintPointer({ pointerId: event.pointerId, erase, lastX: point.x, lastY: point.y, pendingDistance: 0 });
    drawInpaintBrushStamp(point.x, point.y, erase);
    renderInpaintPreview();
    renderInpaintMaskCanvas();
    renderInpaintMaskPreview();
  }

  function continueInpaintStroke(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!inpaintPointer || inpaintPointer.pointerId !== event.pointerId) {
      return;
    }

    event.preventDefault();
    const overlayCanvas = inpaintOverlayCanvasRef.current;
    if (!overlayCanvas) {
      return;
    }

    const point = getCanvasPoint(event, overlayCanvas, overlayCanvas.width, overlayCanvas.height);
    const nextPointer = drawInpaintBrushSegment(
      inpaintPointer.lastX,
      inpaintPointer.lastY,
      point.x,
      point.y,
      inpaintPointer.erase,
      inpaintPointer.pendingDistance,
    );
    renderInpaintPreview();
    renderInpaintMaskCanvas();
    renderInpaintMaskPreview();
    setInpaintPointer((current) =>
      current?.pointerId === event.pointerId
        ? {
            ...current,
            ...nextPointer,
          }
        : current,
    );
  }

  function endInpaintStroke(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (inpaintPointer?.pointerId === event.pointerId) {
      if (inpaintOverlayCanvasRef.current?.hasPointerCapture(event.pointerId)) {
        inpaintOverlayCanvasRef.current.releasePointerCapture(event.pointerId);
      }
      setInpaintPointer(null);
    }
  }

  function pushInpaintUndoState() {
    const maskBuffer = inpaintMaskBufferRef.current;
    if (!maskBuffer) {
      return;
    }

    inpaintUndoStackRef.current = [...inpaintUndoStackRef.current.slice(-4), new Float32Array(maskBuffer)];
  }

  function undoInpaintMask() {
    const previousState = inpaintUndoStackRef.current.pop();
    if (!previousState) {
      return;
    }

    inpaintMaskBufferRef.current = new Float32Array(previousState);
    renderInpaintPreview();
    renderInpaintMaskCanvas();
    renderInpaintMaskPreview();
  }

  function invertInpaintMaskMode() {
    setIsInpaintMaskInverted((current) => !current);
  }

  function beginInpaintSelection(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!inpaintSource || !inpaintOverlayCanvasRef.current || event.button !== 0) {
      return;
    }

    event.preventDefault();
    const overlayCanvas = inpaintOverlayCanvasRef.current;
    const point = getCanvasPoint(event, overlayCanvas, overlayCanvas.width, overlayCanvas.height);
    overlayCanvas.setPointerCapture(event.pointerId);

    if (inpaintTool === "lasso") {
      const nextSelection: InpaintSelectionState = {
        type: "lasso",
        pointerId: event.pointerId,
        points: [point],
      };
      setInpaintSelection(nextSelection);
      drawInpaintSelectionOutline(nextSelection);
      return;
    }

    const nextSelection: InpaintSelectionState = {
      type: "rectangle",
      pointerId: event.pointerId,
      startX: point.x,
      startY: point.y,
      currentX: point.x,
      currentY: point.y,
    };
    setInpaintSelection(nextSelection);
    drawInpaintSelectionOutline(nextSelection);
  }

  function continueInpaintSelection(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!inpaintSelection || inpaintSelection.pointerId !== event.pointerId || !inpaintOverlayCanvasRef.current) {
      return;
    }

    event.preventDefault();
    const overlayCanvas = inpaintOverlayCanvasRef.current;
    const point = getCanvasPoint(event, overlayCanvas, overlayCanvas.width, overlayCanvas.height);
    const nextSelection: InpaintSelectionState =
      inpaintSelection.type === "lasso"
        ? {
            ...inpaintSelection,
            points: [...inpaintSelection.points, point],
          }
        : {
            ...inpaintSelection,
            currentX: point.x,
            currentY: point.y,
          };

    setInpaintSelection(nextSelection);
    drawInpaintSelectionOutline(nextSelection);
  }

  function endInpaintSelection(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!inpaintSelection || inpaintSelection.pointerId !== event.pointerId) {
      return;
    }

    event.preventDefault();
    if (inpaintOverlayCanvasRef.current?.hasPointerCapture(event.pointerId)) {
      inpaintOverlayCanvasRef.current.releasePointerCapture(event.pointerId);
    }

    pushInpaintUndoState();
    applyInpaintSelection(inpaintSelection);
    setInpaintSelection(null);
    renderInpaintPreview();
    renderInpaintMaskCanvas();
    renderInpaintMaskPreview();
  }

  function drawInpaintSelectionOutline(selection: InpaintSelectionState) {
    const overlayCanvas = inpaintOverlayCanvasRef.current;
    if (!overlayCanvas) {
      return;
    }

    renderInpaintPreview();
    const context = overlayCanvas.getContext("2d");
    if (!context) {
      return;
    }

    context.save();
    context.strokeStyle = "rgba(255,255,255,0.95)";
    context.lineWidth = 1.5;
    context.setLineDash([6, 4]);
    context.shadowColor = "rgba(47,94,255,0.95)";
    context.shadowBlur = 4;
    context.beginPath();
    if (selection.type === "lasso") {
      const [firstPoint, ...restPoints] = selection.points;
      if (!firstPoint) {
        context.restore();
        return;
      }
      context.moveTo(firstPoint.x, firstPoint.y);
      restPoints.forEach((point) => context.lineTo(point.x, point.y));
    } else {
      const x = Math.min(selection.startX, selection.currentX);
      const y = Math.min(selection.startY, selection.currentY);
      const width = Math.abs(selection.currentX - selection.startX);
      const height = Math.abs(selection.currentY - selection.startY);
      context.rect(x, y, width, height);
    }
    context.stroke();
    context.restore();
  }

  function applyInpaintSelection(selection: InpaintSelectionState) {
    if (selection.type === "lasso") {
      if (selection.points.length < 3) {
        return;
      }
      fillInpaintSelectionPath((context) => {
        const [firstPoint, ...restPoints] = selection.points;
        context.moveTo(firstPoint.x, firstPoint.y);
        restPoints.forEach((point) => context.lineTo(point.x, point.y));
        context.closePath();
      });
      return;
    }

    const x = Math.min(selection.startX, selection.currentX);
    const y = Math.min(selection.startY, selection.currentY);
    const width = Math.abs(selection.currentX - selection.startX);
    const height = Math.abs(selection.currentY - selection.startY);
    if (width < 1 || height < 1) {
      return;
    }

    fillInpaintSelectionPath((context) => {
      context.rect(x, y, width, height);
    });
  }

  function fillInpaintSelectionPath(drawPath: (context: CanvasRenderingContext2D) => void) {
    const maskCanvas = inpaintMaskCanvasRef.current;
    const maskBuffer = inpaintMaskBufferRef.current;
    if (!maskCanvas || !maskBuffer) {
      return;
    }

    const targetMaskValue = isInpaintSubtractEnabled || isInpaintMaskInverted ? 0 : 1;
    const bufferValue = isInpaintMaskInverted ? 1 - targetMaskValue : targetMaskValue;
    const selectionCanvas = document.createElement("canvas");
    selectionCanvas.width = maskCanvas.width;
    selectionCanvas.height = maskCanvas.height;
    const selectionContext = selectionCanvas.getContext("2d");
    if (!selectionContext) {
      return;
    }

    selectionContext.fillStyle = targetMaskValue === 1 ? "#ffffff" : "#000000";
    selectionContext.beginPath();
    drawPath(selectionContext);
    selectionContext.fill();

    const selectionData = selectionContext.getImageData(0, 0, selectionCanvas.width, selectionCanvas.height).data;
    for (let index = 0; index < maskBuffer.length; index += 1) {
      if (selectionData[index * 4 + 3] > 0) {
        maskBuffer[index] = bufferValue;
      }
    }
  }

  function updateInpaintCursor(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (inpaintTool !== "brush") {
      return;
    }

    const rect = event.currentTarget.getBoundingClientRect();
    setInpaintCursor({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
      visible: true,
    });
  }

  function hasInpaintMaskPainted() {
    const maskBuffer = inpaintMaskBufferRef.current;
    return Boolean(maskBuffer?.some((value) => value > 0.01));
  }

  function getEffectiveInpaintMask(maskBuffer: Float32Array) {
    if (!isInpaintMaskInverted) {
      return new Float32Array(maskBuffer);
    }

    const invertedMask = new Float32Array(maskBuffer.length);
    for (let index = 0; index < maskBuffer.length; index += 1) {
      invertedMask[index] = 1 - clamp(maskBuffer[index], 0, 1);
    }
    return invertedMask;
  }

  function drawInpaintBrushSegment(
    fromX: number,
    fromY: number,
    toX: number,
    toY: number,
    erase: boolean,
    pendingDistance: number,
  ) {
    const distance = Math.hypot(toX - fromX, toY - fromY);
    const spacing = Math.max(1, Number(inpaintForm.brushSize) * MASK_BRUSH_SPACING_RATIO);
    if (distance === 0) {
      return { lastX: fromX, lastY: fromY, pendingDistance };
    }

    const directionX = (toX - fromX) / distance;
    const directionY = (toY - fromY) / distance;
    let travelled = spacing - pendingDistance;
    let lastX = fromX;
    let lastY = fromY;

    while (travelled <= distance) {
      lastX = fromX + directionX * travelled;
      lastY = fromY + directionY * travelled;
      drawInpaintBrushStamp(lastX, lastY, erase);
      travelled += spacing;
    }

    const remainingDistance = distance - (travelled - spacing);
    return {
      lastX: remainingDistance > 0 ? lastX : toX,
      lastY: remainingDistance > 0 ? lastY : toY,
      pendingDistance: remainingDistance > 0 ? remainingDistance : 0,
    };
  }

  function drawInpaintBrushStamp(x: number, y: number, erase: boolean) {
    const overlayCanvas = inpaintOverlayCanvasRef.current;
    const maskCanvas = inpaintMaskCanvasRef.current;
    const maskBuffer = inpaintMaskBufferRef.current;
    if (!overlayCanvas || !maskCanvas || !maskBuffer) {
      return;
    }

    const radius = Math.max(1, Number(inpaintForm.brushSize) / 2);
    const minX = Math.max(0, Math.floor(x - radius));
    const maxX = Math.min(maskCanvas.width - 1, Math.ceil(x + radius));
    const minY = Math.max(0, Math.floor(y - radius));
    const maxY = Math.min(maskCanvas.height - 1, Math.ceil(y + radius));

    for (let pixelY = minY; pixelY <= maxY; pixelY += 1) {
      for (let pixelX = minX; pixelX <= maxX; pixelX += 1) {
        const distance = Math.hypot(pixelX + 0.5 - x, pixelY + 0.5 - y);
        if (distance > radius) {
          continue;
        }

        const falloff = (1 - distance / radius) ** 2;
        const index = pixelY * maskCanvas.width + pixelX;
        maskBuffer[index] = erase
          ? Math.max(0, maskBuffer[index] * (1 - falloff))
          : Math.min(1, maskBuffer[index] + falloff * (1 - maskBuffer[index]));
      }
    }
  }

  function renderInpaintPreview() {
    const overlayCanvas = inpaintOverlayCanvasRef.current;
    const maskBuffer = inpaintMaskBufferRef.current;
    if (!overlayCanvas || !maskBuffer) {
      return;
    }

    const context = overlayCanvas.getContext("2d");
    if (!context) {
      return;
    }

    const previewMask = blurFloatMask(maskBuffer, overlayCanvas.width, overlayCanvas.height, MASK_PREVIEW_FEATHER_PX);
    const imageData = context.createImageData(overlayCanvas.width, overlayCanvas.height);
    const data = imageData.data;
    for (let index = 0; index < previewMask.length; index += 1) {
      const alpha = Math.round(clamp(previewMask[index], 0, 1) * inpaintMaskOpacity * 255);
      const dataIndex = index * 4;
      data[dataIndex] = 47;
      data[dataIndex + 1] = 94;
      data[dataIndex + 2] = 255;
      data[dataIndex + 3] = alpha;
    }

    context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    context.putImageData(imageData, 0, 0);
  }

  function renderInpaintMaskCanvas() {
    const maskCanvas = inpaintMaskCanvasRef.current;
    const maskBuffer = inpaintMaskBufferRef.current;
    if (!maskCanvas || !maskBuffer) {
      return;
    }

    const context = maskCanvas.getContext("2d");
    if (!context) {
      return;
    }

    const imageData = context.createImageData(maskCanvas.width, maskCanvas.height);
    const data = imageData.data;
    for (let index = 0; index < maskBuffer.length; index += 1) {
      const value = Math.round(clamp(maskBuffer[index], 0, 1) * 255);
      const dataIndex = index * 4;
      data[dataIndex] = value;
      data[dataIndex + 1] = value;
      data[dataIndex + 2] = value;
      data[dataIndex + 3] = 255;
    }

    context.putImageData(imageData, 0, 0);
  }

  function renderInpaintMaskPreview() {
    const previewCanvas = inpaintMaskPreviewCanvasRef.current;
    const maskCanvas = inpaintMaskCanvasRef.current;
    const maskBuffer = inpaintMaskBufferRef.current;
    if (!previewCanvas || !maskCanvas || !maskBuffer) {
      return;
    }

    previewCanvas.width = 128;
    previewCanvas.height = Math.max(1, Math.round((maskCanvas.height / maskCanvas.width) * previewCanvas.width));
    const context = previewCanvas.getContext("2d");
    if (!context) {
      return;
    }

    const effectiveMask = getEffectiveInpaintMask(maskBuffer);
    const featheredMask = blurFloatMask(effectiveMask, maskCanvas.width, maskCanvas.height, MASK_BRUSH_FEATHER_PX);
    const sourceCanvas = document.createElement("canvas");
    sourceCanvas.width = maskCanvas.width;
    sourceCanvas.height = maskCanvas.height;
    const sourceContext = sourceCanvas.getContext("2d");
    if (!sourceContext) {
      return;
    }

    const sourceImageData = sourceContext.createImageData(sourceCanvas.width, sourceCanvas.height);
    const sourceData = sourceImageData.data;
    for (let index = 0; index < featheredMask.length; index += 1) {
      const value = Math.round(clamp(featheredMask[index], 0, 1) * 255);
      const dataIndex = index * 4;
      sourceData[dataIndex] = value;
      sourceData[dataIndex + 1] = value;
      sourceData[dataIndex + 2] = value;
      sourceData[dataIndex + 3] = 255;
    }
    sourceContext.putImageData(sourceImageData, 0, 0);

    context.imageSmoothingEnabled = true;
    context.fillStyle = "#000000";
    context.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    context.drawImage(sourceCanvas, 0, 0, previewCanvas.width, previewCanvas.height);
  }

  async function exportInpaintMaskBlob(width: number, height: number): Promise<Blob> {
    const maskCanvas = inpaintMaskCanvasRef.current;
    const maskBuffer = inpaintMaskBufferRef.current;
    if (!maskCanvas || !maskBuffer) {
      throw new Error("Mask canvas is not ready.");
    }

    const effectiveMask = getEffectiveInpaintMask(maskBuffer);
    const featheredMask = blurFloatMask(effectiveMask, maskCanvas.width, maskCanvas.height, MASK_BRUSH_FEATHER_PX);
    const sourceCanvas = document.createElement("canvas");
    sourceCanvas.width = maskCanvas.width;
    sourceCanvas.height = maskCanvas.height;
    const sourceContext = sourceCanvas.getContext("2d");
    if (!sourceContext) {
      throw new Error("Failed to prepare mask canvas.");
    }

    const sourceImageData = sourceContext.createImageData(sourceCanvas.width, sourceCanvas.height);
    const sourceData = sourceImageData.data;
    for (let index = 0; index < featheredMask.length; index += 1) {
      const value = Math.round(clamp(featheredMask[index], 0, 1) * 255);
      const dataIndex = index * 4;
      sourceData[dataIndex] = value;
      sourceData[dataIndex + 1] = value;
      sourceData[dataIndex + 2] = value;
      sourceData[dataIndex + 3] = 255;
    }
    sourceContext.putImageData(sourceImageData, 0, 0);

    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = width;
    exportCanvas.height = height;
    const context = exportCanvas.getContext("2d");
    if (!context) {
      throw new Error("Failed to prepare mask export.");
    }

    context.imageSmoothingEnabled = false;
    context.fillStyle = "#000000";
    context.fillRect(0, 0, width, height);
    context.drawImage(sourceCanvas, 0, 0, width, height);

    const imageData = context.getImageData(0, 0, width, height);
    const data = imageData.data;
    for (let index = 0; index < data.length; index += 4) {
      const value = data[index];
      data[index] = value;
      data[index + 1] = value;
      data[index + 2] = value;
      data[index + 3] = 255;
    }
    context.putImageData(imageData, 0, 0);

    const blob = await new Promise<Blob | null>((resolve) => {
      exportCanvas.toBlob(resolve, "image/png");
    });
    if (!blob) {
      throw new Error("Failed to export inpaint mask.");
    }

    return blob;
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
  const isInpaintMode = mode === "upload-inpaint";
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
    (!isInpaintMode && loraStyles.length === 0) ||
    (isInpaintMode && !inpaintSource) ||
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
                prompt controls, ControlNet guidance, freeform canvas composition, upload inpainting, and IP-Adapter references.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <MetricCard label="Scheduler" value="DPM++ 2M" detail="Karras sigmas" accent="orange" />
              <MetricCard label="Device" value="CPU" detail="GPU disabled" accent="green" />
              <MetricCard label="Modes" value="5" detail="Normal, Lineart, Canvas, Inpaint, IP-Adapter" accent="slate" />
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

              {isInpaintMode ? (
                <>
                  <FieldShell
                    label="Upload Base Image"
                    helper="Upload the image to edit, then paint the mask in the preview."
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
                          onChange={handleInpaintUpload}
                        />
                      </label>

                      {inpaintSource ? (
                        <div className="flex items-center gap-4 rounded-[20px] border border-white/10 bg-white/5 p-3">
                          <img
                            src={inpaintSource.src}
                            alt={inpaintSource.name}
                            className="h-16 w-16 rounded-xl border border-white/10 object-cover"
                          />
                          <div className="min-w-0">
                            <div className="truncate text-sm font-semibold text-white">{inpaintSource.name}</div>
                            <div className="mt-1 text-xs text-slate-400">
                              {inpaintSource.width} x {inpaintSource.height}px
                            </div>
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </FieldShell>

                  <FieldShell
                    label="ControlNet Reference"
                    helper="Optional. If empty, ControlNet uses the base image."
                  >
                    <div className="space-y-4 rounded-[24px] border border-white/10 bg-white/[0.04] p-4">
                      <label className="flex cursor-pointer items-center justify-between rounded-[20px] border border-dashed border-white/15 bg-white/[0.03] px-4 py-4 transition hover:border-orange-400/35 hover:bg-white/[0.06]">
                        <div>
                          <div className="font-display text-lg font-semibold text-white">Upload Control Reference</div>
                          <div className="mt-1 text-sm text-slate-400">Lineart guide source for inpaint ControlNet</div>
                        </div>
                        <div className="rounded-full bg-orange-500 px-4 py-2 text-sm font-semibold text-slate-950">Choose</div>
                        <input
                          type="file"
                          accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
                          className="hidden"
                          onChange={handleInpaintControlUpload}
                        />
                      </label>

                      {inpaintControlSource ? (
                        <div className="flex items-center justify-between gap-3 rounded-[20px] border border-white/10 bg-white/5 p-3">
                          <div className="flex min-w-0 items-center gap-4">
                            <img
                              src={inpaintControlSource.src}
                              alt={inpaintControlSource.name}
                              className="h-16 w-16 rounded-xl border border-white/10 object-cover"
                            />
                            <div className="min-w-0">
                              <div className="truncate text-sm font-semibold text-white">{inpaintControlSource.name}</div>
                              <div className="mt-1 text-xs text-slate-400">
                                {inpaintControlSource.width} x {inpaintControlSource.height}px
                              </div>
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={() => setInpaintControlSource(null)}
                            className="rounded-full border border-white/10 px-3 py-2 text-xs font-semibold text-slate-300 transition hover:border-red-400/30 hover:bg-white/5 hover:text-white"
                          >
                            Remove
                          </button>
                        </div>
                      ) : null}
                    </div>
                  </FieldShell>
                </>
              ) : null}

              <FieldShell
                label="Positive Prompt"
                helper={
                  isInpaintMode
                    ? "Describe what should appear in the white masked area."
                    : isIPAdapterMode
                      ? "Describe the scene, style, and composition. Do not describe the character — IP-Adapter handles that from your reference images."
                      : "Describe the subject, style, composition, lighting, and desired details."
                }
              >
                <textarea
                  value={isInpaintMode ? inpaintForm.prompt : form.positivePrompt}
                  onChange={(event) =>
                    isInpaintMode
                      ? updateInpaintField("prompt", event.target.value)
                      : updateField("positivePrompt", event.target.value)
                  }
                  placeholder={
                    isInpaintMode
                      ? "Add a clean comic-style gold crown, matching lighting and ink detail..."
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
                helper="Suppress unwanted artifacts, styles, or visual defects."
              >
                <textarea
                  value={isInpaintMode ? inpaintForm.negativePrompt : form.negativePrompt}
                  onChange={(event) =>
                    isInpaintMode
                      ? updateInpaintField("negativePrompt", event.target.value)
                      : updateField("negativePrompt", event.target.value)
                  }
                  placeholder="blurry, low quality, distorted face, extra fingers..."
                  rows={4}
                  className="w-full resize-none rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-orange-400/50 focus:bg-white/[0.07]"
                />
              </FieldShell>

              {!isInpaintMode ? (
                <>
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
                        value={form.loraStyle}
                        onChange={(event) => handleLoraStyleChange(event.target.value)}
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
                      helper={`${form.loraStrength.toFixed(2)} between 0.1 and 2.0`}
                    >
                      <input
                        type="range"
                        min={0.1}
                        max={2.0}
                        step={0.05}
                        value={form.loraStrength}
                        onChange={(event) => updateField("loraStrength", Number(event.target.value))}
                        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-orange-500"
                      />
                    </FieldShell>
                  </div>
                </>
              ) : null}

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

              {isInpaintMode ? (
                <>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <NumberField label="Width" helper="Output width" min={64} max={2048} step={8} value={inpaintForm.width} onChange={(value) => updateInpaintField("width", value)} />
                    <NumberField label="Height" helper="Output height" min={64} max={2048} step={8} value={inpaintForm.height} onChange={(value) => updateInpaintField("height", value)} />
                    <NumberField label="Steps" helper="Default 30" min={1} max={100} step={1} value={inpaintForm.steps} onChange={(value) => updateInpaintField("steps", value)} />
                    <NumberField label="Guidance Scale" helper="Default 7.5" min={1} max={30} step={0.5} value={inpaintForm.guidanceScale} onChange={(value) => updateInpaintField("guidanceScale", value)} />
                    <NumberField label="Denoise Strength" helper="0.25-0.35 preserves structure for color edits" min={0.05} max={1} step={0.05} value={inpaintForm.strength} onChange={(value) => updateInpaintField("strength", value)} />
                  </div>
                  <FieldShell label="Brush Size" helper={`${inpaintForm.brushSize}px`}>
                    <input
                      type="range"
                      min={4}
                      max={160}
                      step={1}
                      value={inpaintForm.brushSize}
                      onChange={(event) => updateInpaintField("brushSize", Number(event.target.value))}
                      className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-orange-500"
                    />
                  </FieldShell>
                  <FieldShell label="Seed" helper="-1 or blank for random.">
                    <input
                      value={inpaintForm.seed}
                      onChange={(event) => updateInpaintField("seed", event.target.value)}
                      placeholder="-1"
                      inputMode="numeric"
                      className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-orange-400/50 focus:bg-white/[0.07]"
                    />
                  </FieldShell>
                </>
              ) : (
                <>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <NumberField label="Width" helper="Default 512" min={64} max={1024} step={8} value={form.width} onChange={(value) => updateField("width", value)} />
                    <NumberField label="Height" helper="Default 512" min={64} max={1024} step={8} value={form.height} onChange={(value) => updateField("height", value)} />
                    <NumberField label="Steps" helper="Default 20" min={1} max={100} step={1} value={form.steps} onChange={(value) => updateField("steps", value)} />
                    <NumberField label="CFG Scale" helper="Default 7.5" min={1} max={30} step={0.5} value={form.cfgScale} onChange={(value) => updateField("cfgScale", value)} />
                    <NumberField label="Denoise Strength" helper="Use 0.7-0.85 for adding new features like beards" min={0} max={1} step={0.05} value={form.denoiseStrength} onChange={(value) => updateField("denoiseStrength", value)} />
                  </div>
                  <FieldShell label="Seed" helper="Leave blank for random. Positive and negative values are accepted.">
                    <input
                      value={form.seed}
                      onChange={(event) => updateField("seed", event.target.value)}
                      placeholder="Random"
                      inputMode="numeric"
                      className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-orange-400/50 focus:bg-white/[0.07]"
                    />
                  </FieldShell>
                </>
              )}

            </div>

            {error ? (
              <div className="mt-5 rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                {error}
              </div>
            ) : null}

            <div className="mt-6 flex flex-col gap-3 border-t border-white/10 pt-5 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm text-slate-400">
                {isInpaintMode
                  ? "CPU-only inpainting. White mask pixels are regenerated and black pixels are preserved."
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
                      : mode === "upload-inpaint"
                        ? "Run Inpaint"
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
                          : mode === "upload-inpaint"
                            ? "Paint the mask over areas to regenerate, then run the local inpaint pipeline."
                            : "Upload reference images and generate to see output here."}
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
                    {inpaintSource ? (
                      <div
                        className="relative mx-auto overflow-hidden rounded-[20px] border border-white/10 bg-white shadow-inner"
                        style={{
                          width: `${inpaintForm.width}px`,
                          height: `${inpaintForm.height}px`,
                        }}
                      >
                        <img
                          src={inpaintSource.src}
                          alt={inpaintSource.name}
                          className="absolute inset-0 h-full w-full select-none object-fill"
                          draggable={false}
                        />
                        <canvas
                          ref={inpaintOverlayCanvasRef}
                          width={inpaintForm.width}
                          height={inpaintForm.height}
                          onContextMenu={(event) => event.preventDefault()}
                          onPointerDown={beginInpaintCanvasPointer}
                          onPointerMove={continueInpaintCanvasPointer}
                          onPointerUp={endInpaintCanvasPointer}
                          onPointerCancel={endInpaintCanvasPointer}
                          onPointerEnter={updateInpaintCursor}
                          onPointerLeave={leaveInpaintCanvas}
                          className={`absolute inset-0 h-full w-full touch-none ${inpaintTool === "brush" ? "cursor-none" : "cursor-crosshair"}`}
                        />
                        {inpaintTool === "brush" && inpaintCursor.visible ? (
                          <div
                            className="pointer-events-none absolute rounded-full border border-white/90 shadow-[0_0_0_1px_rgba(47,94,255,0.8),0_0_10px_rgba(47,94,255,0.7)]"
                            style={{
                              width: `${inpaintForm.brushSize}px`,
                              height: `${inpaintForm.brushSize}px`,
                              left: `${inpaintCursor.x}px`,
                              top: `${inpaintCursor.y}px`,
                              transform: "translate(-50%, -50%)",
                            }}
                          />
                        ) : null}
                        <canvas
                          ref={inpaintMaskCanvasRef}
                          width={inpaintForm.width}
                          height={inpaintForm.height}
                          className="hidden"
                        />
                      </div>
                    ) : (
                      <div className="flex min-h-[360px] items-center justify-center rounded-[20px] border-2 border-dashed border-white/15 px-8 text-center text-sm text-slate-400">
                        Upload a base image to begin masking.
                      </div>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center justify-between gap-3 rounded-[20px] border border-white/10 bg-white/[0.04] p-3">
                    <div className="flex flex-wrap gap-2">
                      {(["brush", "lasso", "rectangle"] as InpaintMaskTool[]).map((tool) => (
                        <button
                          key={tool}
                          type="button"
                          onClick={() => setInpaintTool(tool)}
                          disabled={!inpaintSource}
                          className={`rounded-full border px-4 py-2 text-sm font-medium capitalize transition disabled:cursor-not-allowed disabled:opacity-50 ${
                            inpaintTool === tool
                              ? "border-orange-400/50 bg-orange-500 text-slate-950"
                              : "border-white/10 text-slate-300 hover:border-orange-400/30 hover:bg-white/5 hover:text-white"
                          }`}
                        >
                          {tool === "brush" ? "Brush" : tool === "lasso" ? "Lasso" : "Rectangle"}
                        </button>
                      ))}
                    </div>
                    <button
                      type="button"
                      onClick={() => setIsInpaintSubtractEnabled((current) => !current)}
                      disabled={!inpaintSource || inpaintTool === "brush"}
                      className={`rounded-full border px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${
                        isInpaintSubtractEnabled
                          ? "border-red-400/50 bg-red-500 text-white"
                          : "border-white/10 text-slate-300 hover:border-red-400/30 hover:bg-white/5 hover:text-white"
                      }`}
                    >
                      Subtract
                    </button>
                  </div>

                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm text-slate-400">
                      {inpaintSource
                        ? `Mask canvas: ${inpaintForm.width} x ${inpaintForm.height}px`
                        : "Left-drag paints mask. Use Eraser or right-drag to erase mask."}
                    </div>
                    <div className="flex flex-wrap justify-end gap-2">
                      <button
                        type="button"
                        onClick={() => setIsInpaintEraserEnabled((current) => !current)}
                        disabled={!inpaintSource}
                        className={`rounded-full border px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${
                          isInpaintEraserEnabled
                            ? "border-orange-400/50 bg-orange-500 text-slate-950"
                            : "border-white/10 text-slate-300 hover:border-orange-400/30 hover:bg-white/5 hover:text-white"
                        }`}
                      >
                        Eraser
                      </button>
                      <button
                        type="button"
                        onClick={undoInpaintMask}
                        disabled={!inpaintSource}
                        className="rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-orange-400/30 hover:bg-white/5 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        Undo
                      </button>
                      <button
                        type="button"
                        onClick={invertInpaintMaskMode}
                        disabled={!inpaintSource}
                        className="rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-orange-400/30 hover:bg-white/5 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        {isInpaintMaskInverted ? "Paint Change" : "Paint Keep"}
                      </button>
                      <button
                        type="button"
                        onClick={clearInpaintMask}
                        disabled={!inpaintSource}
                        className="rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-red-400/30 hover:bg-white/5 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        Clear Mask
                      </button>
                    </div>
                  </div>

                  {inpaintSource ? (
                    <div className="grid gap-4 rounded-[20px] border border-white/10 bg-white/[0.04] p-4 sm:grid-cols-[1fr_auto]">
                      <FieldShell label="Mask Opacity" helper={`${inpaintMaskOpacity.toFixed(2)} between 0.3 and 1.0`}>
                        <input
                          type="range"
                          min={0.3}
                          max={1}
                          step={0.05}
                          value={inpaintMaskOpacity}
                          onChange={(event) => setInpaintMaskOpacity(Number(event.target.value))}
                          className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-orange-500"
                        />
                      </FieldShell>
                      <div className="space-y-2">
                        <div className="text-xs font-semibold text-slate-400">Exported Mask</div>
                        <canvas
                          ref={inpaintMaskPreviewCanvasRef}
                          className="h-20 rounded-lg border border-white/10 bg-black object-contain"
                        />
                      </div>
                    </div>
                  ) : null}

                  {inpaintResult ? (
                    <div className="space-y-4">
                      <div className="overflow-hidden rounded-[24px] border border-white/10 bg-[rgba(255,255,255,0.03)]">
                        <div className="flex items-center justify-between gap-3 border-b border-white/10 px-4 py-3">
                          <div className="text-sm font-semibold text-slate-200">Inpaint Result</div>
                          <div className="flex flex-wrap gap-2">
                            <button
                              type="button"
                              onClick={() => setShowInpaintOriginal((current) => !current)}
                              className="rounded-full border border-white/10 px-3 py-1.5 text-xs font-semibold text-slate-300 transition hover:border-orange-400/30 hover:bg-white/5 hover:text-white"
                            >
                              {showInpaintOriginal ? "Show Result" : "Compare"}
                            </button>
                            <button
                              type="button"
                              onClick={useInpaintResultAsBase}
                              className="rounded-full border border-white/10 px-3 py-1.5 text-xs font-semibold text-slate-300 transition hover:border-orange-400/30 hover:bg-white/5 hover:text-white"
                            >
                              Use as Base
                            </button>
                          </div>
                        </div>
                        <img
                          src={showInpaintOriginal && inpaintSource ? inpaintSource.src : inpaintResult.imageUrl}
                          alt="Inpainted result"
                          className="max-h-[640px] w-full bg-white object-contain"
                        />
                      </div>
                      <div className="grid gap-3 sm:grid-cols-2">
                        <InfoChip label="Seed" value={String(inpaintResult.response.seed_used)} />
                        <InfoChip label="Steps" value={String(inpaintResult.response.steps_used)} />
                        <InfoChip label="Scheduler Steps" value={String(inpaintResult.response.pipeline_steps)} />
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
                    <div className="grid gap-4 md:grid-cols-2">
                      <PreviewPanel title="Lineart Preview" src={controlNetResult.response.lineart_preview_url} alt="Preprocessed lineart preview" />
                      <PreviewPanel title="Generated Output" src={controlNetResult.response.image_url} alt="Canvas Compose generated output" />
                    </div>
                  ) : null}
                </div>
              ) : isControlGuidedMode ? (
                controlNetResult ? (
                  <div className="space-y-4">
                      <div className="grid gap-4 md:grid-cols-2">
                        <PreviewPanel title="Lineart Preview" src={controlNetResult.response.lineart_preview_url} alt="Preprocessed lineart preview" />
                        <PreviewPanel title="Generated Output" src={controlNetResult.response.image_url} alt="ControlNet generated output" />
                      </div>
                  </div>
                ) : (
                  <EmptyPreview message="Generate to see the lineart preview and final output here." />
                )
              ) : normalResult ? (
                <div className="space-y-4">
                  <div className="overflow-hidden rounded-[24px] border border-white/10 bg-[rgba(255,255,255,0.03)]">
                    <img src={normalResult.response.image.image_url} alt={normalResult.response.image.positive_prompt} className="aspect-square w-full object-cover" />
                  </div>
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
                    isInpaintMode
                      ? "Inpaint mode reuses local SD1.5 inpainting"
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

      </div>
    </main>
  );
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function blurFloatMask(source: Float32Array, width: number, height: number, radius: number) {
  const safeRadius = Math.max(0, Math.round(radius));
  if (safeRadius === 0) {
    return new Float32Array(source);
  }

  const horizontal = new Float32Array(source.length);
  const output = new Float32Array(source.length);
  const windowSize = safeRadius * 2 + 1;

  for (let y = 0; y < height; y += 1) {
    let sum = 0;
    for (let x = -safeRadius; x <= safeRadius; x += 1) {
      const sampleX = clamp(x, 0, width - 1);
      sum += source[y * width + sampleX];
    }

    for (let x = 0; x < width; x += 1) {
      horizontal[y * width + x] = sum / windowSize;
      const removeX = clamp(x - safeRadius, 0, width - 1);
      const addX = clamp(x + safeRadius + 1, 0, width - 1);
      sum += source[y * width + addX] - source[y * width + removeX];
    }
  }

  for (let x = 0; x < width; x += 1) {
    let sum = 0;
    for (let y = -safeRadius; y <= safeRadius; y += 1) {
      const sampleY = clamp(y, 0, height - 1);
      sum += horizontal[sampleY * width + x];
    }

    for (let y = 0; y < height; y += 1) {
      output[y * width + x] = sum / windowSize;
      const removeY = clamp(y - safeRadius, 0, height - 1);
      const addY = clamp(y + safeRadius + 1, 0, height - 1);
      sum += horizontal[addY * width + x] - horizontal[removeY * width + x];
    }
  }

  return output;
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
