"""Etapa de clasificación de cromosomas.

Clasifica cada recorte rectificado de 224x224 (salida de extraction.py) en una de
las 24 clases cromosómicas (1--22, X, Y). El modelo es un VGG16 + atención SE
entrenado en Keras/TensorFlow y exportado a ONNX, de modo que la interfaz lo corre
con `onnxruntime` sin necesidad de instalar TensorFlow (misma filosofía que la
segmentación con TorchScript).

Consistencia de entrada: el clasificador se entrenó con `ImageDataGenerator(rescale
=1/255)` sobre recortes RGB crudos; en inferencia se reproduce exactamente ese
preprocesamiento (RGB en [0,1]), sin denoising ni CLAHE.
"""

from __future__ import annotations

import os

import numpy as np

from .karyogram import index_to_chromosome  # mapeo índice -> '1'..'22','X','Y'

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_CLF_DIR = os.path.join(_REPO_ROOT, "Modelos", "Clasificacion")

DEFAULT_ONNX_WEIGHTS = os.path.join(_CLF_DIR, "model_VGG_v2.onnx")
INPUT_SIZE = 224


def onnx_available(onnx_path: str | None = None) -> bool:
    path = onnx_path or DEFAULT_ONNX_WEIGHTS
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return os.path.exists(path)


def load_classifier(onnx_path: str | None = None):
    """Carga la sesión de onnxruntime del clasificador."""
    import onnxruntime as ort

    path = onnx_path or DEFAULT_ONNX_WEIGHTS
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess


def _to_batch(images) -> np.ndarray:
    """Lista de PIL.Image (RGB) -> tensor (N, 224, 224, 3) float32 en [0,1]."""
    batch = np.empty((len(images), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    for i, im in enumerate(images):
        arr = np.asarray(im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE)), dtype=np.float32)
        batch[i] = arr / 255.0
    return batch


def classify(session, chromosomes):
    """Clasifica una lista de cromosomas (objetos con atributo `.image`).

    Devuelve (indices, labels, probs):
      - indices: np.ndarray (N,) con el índice de clase 0..23
      - labels:  lista (N,) con el nombre de cromosoma ('1'..'22','X','Y')
      - probs:   np.ndarray (N,) con la probabilidad (softmax) de la clase elegida
    """
    if not chromosomes:
        return np.empty((0,), int), [], np.empty((0,), float)

    batch = _to_batch([c.image for c in chromosomes])
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: batch})[0]  # (N, 24) softmax

    indices = np.argmax(logits, axis=1)
    probs = logits[np.arange(len(indices)), indices]
    labels = [index_to_chromosome(int(i)) for i in indices]
    return indices, labels, probs
