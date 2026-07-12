"""Etapa de segmentación de instancias con Mask R-CNN.

Backend principal: el modelo exportado a **TorchScript** (`model_ts.ts`), que
corre solo con `torch` (sin Detectron2). Ver Pipelines/export_torchscript_colab.py
para generar ese archivo.

Backend fallback: Detectron2 con los pesos `.pth` (si estuviera instalado).

El pipeline trabaja a 1600x1600 (tamaño de entrenamiento): las máscaras y la
visualización se devuelven a esa resolución.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SEG_DIR = os.path.join(_REPO_ROOT, "Modelos", "Segmentacion")

DEFAULT_TS_WEIGHTS = os.path.join(_SEG_DIR, "model_ts.ts")       # TorchScript
DEFAULT_WEIGHTS = os.path.join(_SEG_DIR, "model_final.pth")      # Detectron2 (.pth)

INPUT_SIZE = 1600
_CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


@dataclass
class SegmentationResult:
    masks: np.ndarray          # (N, 1600, 1600) booleano
    boxes: np.ndarray          # (N, 4) xyxy
    scores: np.ndarray         # (N,)
    overlay: np.ndarray        # imagen RGB 1600x1600 con la visualización
    count: int
    image: np.ndarray          # imagen de entrada redimensionada a 1600x1600 (RGB)


# --------------------------------------------------------------------------- #
# Disponibilidad de backends
# --------------------------------------------------------------------------- #
def torchscript_available(ts_path: str | None = None) -> bool:
    path = ts_path or DEFAULT_TS_WEIGHTS
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return os.path.exists(path)


def detectron2_available() -> bool:
    try:
        import detectron2  # noqa: F401
        return True
    except Exception:
        return False


def _resize_1600(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.shape[0] == INPUT_SIZE and image_rgb.shape[1] == INPUT_SIZE:
        return image_rgb
    return cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE))


# --------------------------------------------------------------------------- #
# Visualización propia (OpenCV) — no depende de Detectron2
# --------------------------------------------------------------------------- #
def draw_overlay(image_rgb: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image_rgb.copy()
    rng = np.random.default_rng(0)
    for m in masks:
        color = rng.integers(60, 256, size=3).tolist()
        overlay[m] = color
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 0, 0), 1)
    return cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)


# --------------------------------------------------------------------------- #
# Backend TorchScript (torch-only)
# --------------------------------------------------------------------------- #
def load_torchscript(ts_path: str | None = None):
    import torch
    import torchvision  # noqa: F401  (registra torchvision::nms y demás ops del grafo)
    path = ts_path or DEFAULT_TS_WEIGHTS
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model


def _parse_ts_outputs(outputs):
    """Interpreta las salidas del modelo trazado por SHAPE (robusto al orden)."""
    tensors = list(outputs)
    boxes = next((t for t in tensors if t.ndim == 2 and t.shape[-1] == 4), None)

    masks = None
    for t in tensors:
        if t.ndim == 3:
            masks = t
            break
        if t.ndim == 4 and t.shape[1] == 1:
            masks = t[:, 0]
            break

    n = boxes.shape[0] if boxes is not None else (masks.shape[0] if masks is not None else 0)

    scores = None
    classes = None
    for t in tensors:
        if t.ndim == 1 and t.shape[0] == n and n > 0:
            if t.is_floating_point() and scores is None:
                scores = t
            elif not t.is_floating_point() and classes is None:
                classes = t
    return boxes, scores, classes, masks


def segment_torchscript(model, image_rgb: np.ndarray, score_thresh: float = 0.5) -> SegmentationResult:
    import torch

    img = _resize_1600(image_rgb)
    tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    with torch.no_grad():
        outputs = model(tensor)

    boxes, scores, _, masks = _parse_ts_outputs(outputs)

    if masks is None or masks.shape[0] == 0:
        empty = np.empty((0, INPUT_SIZE, INPUT_SIZE), bool)
        return SegmentationResult(empty, np.empty((0, 4)), np.empty((0,)),
                                  img.copy(), 0, img)

    masks_np = masks.cpu().numpy().astype(bool)
    boxes_np = boxes.cpu().numpy() if boxes is not None else np.empty((len(masks_np), 4))
    scores_np = scores.cpu().numpy() if scores is not None else np.ones(len(masks_np))

    # Filtro por umbral (el modelo se exportó con thresh 0.5; esto sólo puede subirlo)
    keep = scores_np >= score_thresh
    masks_np, boxes_np, scores_np = masks_np[keep], boxes_np[keep], scores_np[keep]

    overlay = draw_overlay(img, masks_np)
    return SegmentationResult(masks_np, boxes_np, scores_np, overlay, int(len(scores_np)), img)


# --------------------------------------------------------------------------- #
# Backend Detectron2 (fallback, si estuviera instalado)
# --------------------------------------------------------------------------- #
def load_detectron2(weights_path: str | None = None, device: str = "cpu",
                    score_thresh: float = 0.5):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(_CONFIG_FILE))
    cfg.INPUT.MIN_SIZE_TEST = INPUT_SIZE
    cfg.INPUT.MAX_SIZE_TEST = INPUT_SIZE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = weights_path or DEFAULT_WEIGHTS
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


def segment_detectron2(predictor, image_rgb: np.ndarray) -> SegmentationResult:
    img = _resize_1600(image_rgb)
    outputs = predictor(img)
    inst = outputs["instances"].to("cpu")
    masks = inst.pred_masks.numpy().astype(bool) if inst.has("pred_masks") \
        else np.empty((0, INPUT_SIZE, INPUT_SIZE), bool)
    boxes = inst.pred_boxes.tensor.numpy() if inst.has("pred_boxes") else np.empty((0, 4))
    scores = inst.scores.numpy() if inst.has("scores") else np.empty((0,))
    overlay = draw_overlay(img, masks)
    return SegmentationResult(masks, boxes, scores, overlay, int(len(scores)), img)


# --------------------------------------------------------------------------- #
# API unificada: prefiere TorchScript, cae a Detectron2
# --------------------------------------------------------------------------- #
def available_backend(ts_path: str | None = None) -> str | None:
    if torchscript_available(ts_path):
        return "torchscript"
    if detectron2_available() and os.path.exists(DEFAULT_WEIGHTS):
        return "detectron2"
    return None


def load_segmenter(ts_path: str | None = None, weights_path: str | None = None,
                   device: str = "cpu", score_thresh: float = 0.5):
    """Devuelve (modelo, backend). Prefiere TorchScript."""
    if torchscript_available(ts_path):
        return load_torchscript(ts_path), "torchscript"
    if detectron2_available():
        return load_detectron2(weights_path, device, score_thresh), "detectron2"
    raise RuntimeError(
        "No hay backend de segmentación disponible: falta el modelo TorchScript "
        f"({ts_path or DEFAULT_TS_WEIGHTS}) y Detectron2 no está instalado."
    )


def segment(model, backend: str, image_rgb: np.ndarray, score_thresh: float = 0.5) -> SegmentationResult:
    if backend == "torchscript":
        return segment_torchscript(model, image_rgb, score_thresh)
    return segment_detectron2(model, image_rgb)
