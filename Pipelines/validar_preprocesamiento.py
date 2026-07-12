"""Validación del preprocesamiento canónico sobre múltiples imágenes.

Toma N imágenes de distintos pacientes de AutoKary2022, aplica el
preprocesamiento canónico y reporta métricas (σ ruido, PSNR, SSIM) por imagen,
más un resumen. Guarda un montaje antes/después para inspección visual.

Uso:
    python validar_preprocesamiento.py [N]
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import matplotlib.pyplot as plt

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "Interfaz"))
from pipeline import preprocessing as pp  # noqa: E402

DATASET = os.path.join(_REPO, "Datasets", "Autokary2022_1600x1600")


def sample_images(n: int) -> list[str]:
    """Elige n imágenes de distintos pacientes (test + train)."""
    paths = []
    for split in ("test_labelme", "train_labelme"):
        root = os.path.join(DATASET, split)
        if not os.path.isdir(root):
            continue
        for patient in sorted(os.listdir(root)):
            pdir = os.path.join(root, patient)
            if not os.path.isdir(pdir):
                continue
            pngs = sorted(f for f in os.listdir(pdir) if f.endswith(".png"))
            if pngs:
                paths.append(os.path.join(pdir, pngs[0]))  # 1 por paciente
    # espaciar la selección entre pacientes para tener variedad
    if len(paths) <= n:
        return paths
    step = len(paths) / n
    return [paths[int(i * step)] for i in range(n)]


def main(n: int = 10) -> None:
    imgs = sample_images(n)
    print(f"Validando {len(imgs)} imágenes\n")
    header = f"{'imagen':<38} {'σ_in':>6} {'σ_out':>6} {'PSNR':>7} {'SSIM':>7}"
    print(header)
    print("-" * len(header))

    rows = []
    for path in imgs:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"{os.path.basename(path):<38}  (no se pudo leer)")
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        steps = pp.preprocess(image, **pp.CANONICAL)
        res = steps["result"]
        s_in, s_out = steps["sigma_in"], steps["sigma_out"]
        psnr = pp.psnr(steps["gray"], res)
        ssim = pp.ssim(steps["gray"], res)
        rows.append((path, s_in, s_out, psnr, ssim, image, res))
        name = os.path.basename(path)[:37]
        print(f"{name:<38} {s_in:>6.2f} {s_out:>6.2f} {psnr:>7.2f} "
              f"{ssim:>7.4f}" if ssim is not None else f"{name:<38} {s_in:>6.2f} {s_out:>6.2f} {psnr:>7.2f}   n/a")

    if not rows:
        return
    ssims = [r[4] for r in rows if r[4] is not None]
    psnrs = [r[3] for r in rows]
    print("-" * len(header))
    print(f"Resumen: PSNR media={np.mean(psnrs):.2f} dB (min {np.min(psnrs):.2f}) | "
          f"SSIM media={np.mean(ssims):.4f} (min {np.min(ssims):.4f})")

    # Montaje antes/después (2 filas x N columnas)
    k = len(rows)
    fig, ax = plt.subplots(2, k, figsize=(2.4 * k, 5))
    for j, (_, _, _, _, _, orig, res) in enumerate(rows):
        ax[0, j].imshow(orig); ax[0, j].axis("off")
        ax[1, j].imshow(res); ax[1, j].axis("off")
    ax[0, 0].set_ylabel("original")
    ax[1, 0].set_ylabel("preproc")
    fig.suptitle("Preprocesamiento canónico — validación multi-imagen")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "validacion_preprocesamiento.png")
    fig.savefig(out, dpi=90, bbox_inches="tight")
    print(f"\nMontaje guardado en: {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
