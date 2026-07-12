"""Etapa de pre-procesamiento (Clase 1: mejoramiento / denoising).

Importante: el modelo de segmentación fue entrenado sobre las imágenes crudas
de AutoKary2022 (Detectron2 aplica internamente su propia normalización de
intensidad, PIXEL_MEAN/PIXEL_STD). Por consistencia train/inferencia, este
pre-procesamiento NO se aplica por defecto antes de la segmentación: su valor
está en (a) la visualización, (b) los recortes que alimentan la clasificación
y (c) la robustez frente a imágenes más ruidosas que el set de entrenamiento.
"""

from __future__ import annotations

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Parámetros CANÓNICOS del preprocesamiento para clasificación.
#
# Estos valores son el "contrato" con el entrenamiento del clasificador: los
# recortes que ve el modelo en inferencia deben generarse con EXACTAMENTE este
# preprocesamiento que se usó para generar los recortes de entrenamiento.
# No los cambies sin regenerar los recortes y reentrenar.
# --------------------------------------------------------------------------- #
CANONICAL = {
    "do_denoise": True,
    "denoise_method": "nlm",
    "denoise_strength": 7.0,
    "do_clahe": True,
    # clip bajo: las imágenes de AutoKary casi no tienen ruido, un CLAHE fuerte
    # amplifica el fondo uniforme en manchas. 1.5 realza el bandeo sin ensuciar.
    "clahe_clip": 1.5,
    # Sin normalización global: estiraba el rango y amplificaba el fondo. La
    # normalización de intensidad, si se quiere, va por-cromosoma en la extracción.
    "do_normalize": False,
}


def to_gray_rgb(image: np.ndarray) -> np.ndarray:
    """Devuelve una imagen RGB de 3 canales en escala de grises real.

    Las imágenes de cariotipo son en escala de grises; homogeneizamos la
    entrada (puede venir como gris, BGR o RGBA) a un RGB de 3 canales.
    """
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Estima el desvío del ruido con el método de Immerkær (Laplaciano).

    Sirve para reportar cuantitativamente el nivel de ruido antes/después.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    h, w = gray.shape
    laplacian = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, laplacian)
    sigma = np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2))
    return float(sigma)


def denoise(image: np.ndarray, method: str = "nlm", strength: float = 7.0) -> np.ndarray:
    """Denoising preservando bordes.

    method: 'nlm' (Non-Local Means), 'bilateral' o 'median'.
    """
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    if method == "bilateral":
        d = max(5, int(strength))
        return cv2.bilateralFilter(image, d, strength * 10, strength * 10)
    if method == "median":
        k = int(strength) | 1  # fuerza impar
        return cv2.medianBlur(image, max(3, k))
    raise ValueError(f"método de denoising desconocido: {method}")


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Realce de contraste local (CLAHE) sobre el canal de luminancia."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Normalización min-max del rango dinámico a [0, 255]."""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess(
    image: np.ndarray,
    do_denoise: bool = True,
    denoise_method: str = "nlm",
    denoise_strength: float = 7.0,
    do_clahe: bool = True,
    clahe_clip: float = 2.0,
    do_normalize: bool = True,
) -> dict:
    """Corre la cadena completa y devuelve cada etapa intermedia.

    Returns dict con claves: 'gray', 'denoised', 'clahe', 'result' y las
    métricas de ruido estimado antes/después.
    """
    steps: dict = {}
    gray = to_gray_rgb(image)
    steps["gray"] = gray
    steps["sigma_in"] = estimate_noise_sigma(gray)

    current = gray
    if do_denoise:
        current = denoise(current, denoise_method, denoise_strength)
    steps["denoised"] = current

    if do_clahe:
        current = apply_clahe(current, clahe_clip)
    steps["clahe"] = current

    if do_normalize:
        current = normalize_intensity(current)

    steps["result"] = current
    steps["sigma_out"] = estimate_noise_sigma(current)
    return steps


def preprocess_for_classification(image: np.ndarray) -> np.ndarray:
    """Preprocesamiento CANÓNICO (parámetros fijos) para la rama de clasificación.

    Es la única función que deben usar tanto el entrenamiento del clasificador
    (al generar los recortes) como la interfaz en inferencia, para garantizar
    consistencia train/inferencia. Devuelve solo la imagen resultante.
    """
    return preprocess(image, **CANONICAL)["result"]


# --------------------------------------------------------------------------- #
# Métricas de calidad (para el informe: comparación antes/después)
# --------------------------------------------------------------------------- #
def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio en dB entre dos imágenes uint8."""
    ref = reference.astype(np.float64)
    tst = test.astype(np.float64)
    mse = np.mean((ref - tst) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))


def ssim(reference: np.ndarray, test: np.ndarray) -> float | None:
    """Structural Similarity Index. Requiere scikit-image; None si no está."""
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None
    ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY) if reference.ndim == 3 else reference
    tst = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY) if test.ndim == 3 else test
    return float(structural_similarity(ref, tst))
