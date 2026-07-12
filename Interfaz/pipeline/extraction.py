"""Extracción de características + registración (Clases 2 y 3).

Toma la imagen original y las máscaras binarias predichas por Mask R-CNN y
devuelve, por cada cromosoma, un recorte rectificado a 224x224:

    máscara -> contorno -> eje principal (PCA) -> rotación a vertical
            -> escalado global (preserva tamaño relativo) -> centrado en canvas

Es el mismo procedimiento que en Pipelines/Clasificación.ipynb, pero tomando
los puntos desde la máscara predicha en vez de los polígonos de LabelMe, para
que los recortes que ve el clasificador en inferencia coincidan con los de
entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


def compute_principal_axis(points: np.ndarray) -> tuple[float, float]:
    """Eje principal vía PCA. points: Nx2 (x, y). Devuelve (ángulo_rad, largo_px)."""
    pts = points - np.mean(points, axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    if direction[1] < 0:
        direction = -direction
    angle = np.arctan2(direction[1], direction[0])
    projections = pts @ direction
    length = float(projections.max() - projections.min())
    return float(angle), length


def _paste_centered(image: Image.Image, canvas_size, background=(255, 255, 255)) -> Image.Image:
    canvas = Image.new("RGB", canvas_size, background)
    x = (canvas_size[0] - image.width) // 2
    y = (canvas_size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return c.reshape(-1, 2)  # Nx2 (x, y)


@dataclass
class Chromosome:
    image: Image.Image      # recorte rectificado 224x224
    length_px: float        # largo del eje principal (para ordenar por tamaño)
    box: tuple              # bbox (x0, y0, x1, y1) en la imagen original


def extract_chromosomes(
    image_rgb: np.ndarray,
    masks: np.ndarray,
    target_size=(224, 224),
    border: int = 5,
    max_object_ratio: float = 0.95,
) -> list[Chromosome]:
    """Rectifica cada instancia segmentada a un canvas fijo con fondo blanco."""
    original = Image.fromarray(image_rgb)

    # --- Primer pase: geometría (para el escalado global) ---
    infos = []
    for mask in masks:
        contour = _largest_contour(mask)
        if contour is None or len(contour) < 5:
            continue
        angle, length = compute_principal_axis(contour.astype(np.float64))
        infos.append({"mask": mask, "contour": contour, "angle": angle, "length": length})

    if not infos:
        return []

    max_length = max(o["length"] for o in infos)
    scale_global = (target_size[1] * max_object_ratio) / max_length

    # --- Segundo pase: recorte + rotación + escala + centrado ---
    results: list[Chromosome] = []
    for o in infos:
        pts = o["contour"]
        min_x, min_y = np.min(pts, axis=0).astype(int)
        max_x, max_y = np.max(pts, axis=0).astype(int)
        w = (max_x - min_x) + 2 * border
        h = (max_y - min_y) + 2 * border

        # máscara local recortada al bbox
        local_mask = o["mask"][max(0, min_y):max_y, max(0, min_x):max_x].astype(np.uint8) * 255
        mask_img = Image.new("L", (w, h), 0)
        mask_img.paste(Image.fromarray(local_mask), (border, border))

        crop = original.crop((
            max(0, min_x - border), max(0, min_y - border),
            min(original.width, max_x + border), min(original.height, max_y + border),
        ))
        temp = Image.new("RGB", (w, h), (255, 255, 255))
        temp.paste(crop, (border, border))
        isolated = Image.new("RGB", (w, h), (255, 255, 255))
        isolated.paste(temp, (0, 0), mask_img)

        angle_deg = np.degrees(o["angle"]) - 90
        rotated = isolated.rotate(
            angle_deg, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255)
        )
        new_w = max(1, int(rotated.width * scale_global))
        new_h = max(1, int(rotated.height * scale_global))
        resized = rotated.resize((new_w, new_h), Image.LANCZOS)
        final_img = _paste_centered(resized, target_size)

        results.append(Chromosome(
            image=final_img,
            length_px=o["length"],
            box=(int(min_x), int(min_y), int(max_x), int(max_y)),
        ))

    return results
