"""Etapa de segmentación de instancias con Mask R-CNN (Detectron2).

Reproduce la configuración usada en el notebook de entrenamiento
(Pipelines/Segmentación.ipynb): mask_rcnn_R_50_FPN_3x, entrada 1600x1600,
1 clase ("chromosome").

Detectron2 se importa de forma perezosa para que el resto de la interfaz
funcione aunque no esté instalado (su instalación en Windows es delicada).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Ruta por defecto al modelo entrenado, relativa a la raíz del repo.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
DEFAULT_WEIGHTS = os.path.join(_REPO_ROOT, "Modelos", "Segmentacion", "model_final.pth")

_CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


@dataclass
class SegmentationResult:
    masks: np.ndarray          # (N, H, W) booleano
    boxes: np.ndarray          # (N, 4) xyxy
    scores: np.ndarray         # (N,)
    overlay: np.ndarray        # imagen RGB con la visualización
    count: int


def detectron2_available() -> bool:
    try:
        import detectron2  # noqa: F401
        return True
    except Exception:
        return False


def build_predictor(weights_path: str | None = None, device: str = "cpu",
                    score_thresh: float = 0.5):
    """Construye un DefaultPredictor de Detectron2 con la config de entrenamiento."""
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(_CONFIG_FILE))
    cfg.INPUT.MIN_SIZE_TEST = 1600
    cfg.INPUT.MAX_SIZE_TEST = 1600
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = weights_path or DEFAULT_WEIGHTS
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


def segment(predictor, image_rgb: np.ndarray) -> SegmentationResult:
    """Corre la inferencia y devuelve máscaras + una visualización overlay."""
    from detectron2.utils.visualizer import Visualizer, ColorMode

    outputs = predictor(image_rgb)
    instances = outputs["instances"].to("cpu")

    masks = instances.pred_masks.numpy() if instances.has("pred_masks") else np.empty((0,) + image_rgb.shape[:2], bool)
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else np.empty((0, 4))
    scores = instances.scores.numpy() if instances.has("scores") else np.empty((0,))

    vis = Visualizer(image_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE)
    overlay = vis.draw_instance_predictions(instances).get_image()

    return SegmentationResult(
        masks=masks, boxes=boxes, scores=scores,
        overlay=overlay, count=int(len(scores)),
    )
