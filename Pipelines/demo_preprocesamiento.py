"""Validación / demostración de la etapa de preprocesamiento.

Corre el preprocesamiento canónico sobre una imagen real de AutoKary2022,
imprime las métricas (σ de ruido, PSNR, SSIM) y guarda una figura comparativa
antes/después para el informe.

Uso:
    python demo_preprocesamiento.py [ruta_imagen]
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Importa el módulo canónico de la interfaz (única fuente de verdad).
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "Interfaz"))
from pipeline import preprocessing as pp  # noqa: E402

DEFAULT_IMG = os.path.join(
    _REPO, "Datasets", "Autokary2022_1600x1600", "test_labelme",
    "211025-003C", "211025-003C_129_1_688_378_0.461.png",
)


def main(img_path: str) -> None:
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise SystemExit(f"No se pudo leer la imagen: {img_path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    steps = pp.preprocess(image, **pp.CANONICAL)
    result = steps["result"]

    sigma_in, sigma_out = steps["sigma_in"], steps["sigma_out"]
    psnr = pp.psnr(steps["gray"], result)
    ssim = pp.ssim(steps["gray"], result)

    print(f"Imagen: {os.path.basename(img_path)}  {image.shape[1]}x{image.shape[0]}")
    print(f"σ ruido (Immerkær):  {sigma_in:.2f} → {sigma_out:.2f}  "
          f"(reducción {100*(1-sigma_out/sigma_in):.1f}%)")
    print(f"PSNR (gris vs preproc): {psnr:.2f} dB")
    print(f"SSIM (gris vs preproc): {ssim:.4f}" if ssim is not None
          else "SSIM: scikit-image no instalado")

    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    for a, im, t in zip(
        ax,
        [steps["gray"], steps["denoised"], steps["clahe"], result],
        ["Original (gris)", "Denoised (NLM)", "+ CLAHE", "Resultado (canónico)"],
    ):
        a.imshow(im); a.set_title(t); a.axis("off")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "preprocesamiento_demo.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Figura guardada en: {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG)
