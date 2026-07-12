"""Exportar el Mask R-CNN entrenado a TorchScript — PARA CORRER EN COLAB.

Pegá este contenido en una celda al final de `Segmentación.ipynb` (donde
Detectron2 ya está instalado y el modelo entrenado). Produce:
    Modelos/Segmentacion/model_ts.ts
que la interfaz carga con solo `torch` (sin Detectron2).

El trazado se hace en CPU a propósito, para que el archivo cargue sin problemas
en la máquina local (que no tiene GPU). NO modifica tu model_final.pth.
"""

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.export import TracingAdapter
from detectron2.engine import DefaultPredictor

# --- Ajustá estas rutas a tu Drive ---
WEIGHTS = "/content/drive/MyDrive/TPs PAIByB/FINAL/Modelos/Segmentacion/model_final.pth"
OUT_TS = "/content/drive/MyDrive/TPs PAIByB/FINAL/Modelos/Segmentacion/model_ts.ts"
BASE = "/content/drive/MyDrive/TPs PAIByB/FINAL/Autokary2022_1600x1600"
# Un par de imágenes de test 1600x1600 para muestra y verificación de paridad:
SAMPLE_IMAGES = [
    f"{BASE}/test_labelme/211025-003C/211025-003C_129_1_688_378_0.461.png",
    f"{BASE}/test_labelme/211025-003C/211025-003C_32_1_835_213_0.523.png",
]

# --- Config idéntica a la de inferencia ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.INPUT.MIN_SIZE_TEST = 1600
cfg.INPUT.MAX_SIZE_TEST = 1600
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = WEIGHTS
cfg.MODEL.DEVICE = "cpu"          # <-- trazamos en CPU

model = build_model(cfg)
model.eval()
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)


def load_1600(path):
    img = cv2.imread(path)                       # BGR (canal irrelevante: es gris)
    img = cv2.resize(img, (1600, 1600))
    return img


def to_tensor(img):
    return torch.as_tensor(img.astype("float32").transpose(2, 0, 1))  # CHW


# --- Trazado ---
sample = load_1600(SAMPLE_IMAGES[0])
image_t = to_tensor(sample)
# Sin "height"/"width": TracingAdapter sólo acepta tensores al aplanar los
# inputs. Al omitirlos, el post-procesamiento usa el tamaño real de la imagen
# de entrada (1600x1600), que es exactamente lo que queremos.
inputs = [{"image": image_t}]

wrapper = TracingAdapter(model, inputs)
wrapper.eval()
with torch.no_grad():
    traced = torch.jit.trace(wrapper, (image_t,))
traced.save(OUT_TS)
print("Modelo TorchScript guardado en:", OUT_TS)

# --- Mapa de salidas (para interpretar por shape en la interfaz) ---
with torch.no_grad():
    outs = traced(image_t)
print("\nSalidas del modelo trazado:")
for i, o in enumerate(outs):
    print(f"  [{i}] shape={tuple(o.shape)} dtype={o.dtype}")

# --- Verificación de paridad contra Detectron2 (DefaultPredictor) ---
print("\nVerificación de paridad (Detectron2 vs TorchScript):")
predictor = DefaultPredictor(cfg)
for path in SAMPLE_IMAGES:
    img = load_1600(path)
    inst = predictor(img)["instances"].to("cpu")
    n_d2 = len(inst)
    area_d2 = int(inst.pred_masks.sum()) if n_d2 else 0

    with torch.no_grad():
        outs = traced(to_tensor(img))
    # masks = primer tensor 3D; scores = tensor 1D float
    masks = next((o for o in outs if o.ndim == 3), None)
    scores = next((o for o in outs if o.ndim == 1 and o.is_floating_point()), None)
    n_ts = 0 if masks is None else masks.shape[0]
    area_ts = int(masks.sum()) if n_ts else 0

    print(f"  {path.split('/')[-1]:<40} "
          f"D2: {n_d2} inst, área {area_d2}  |  TS: {n_ts} inst, área {area_ts}")

print("\nSi los conteos y áreas coinciden, la exportación es correcta.")
