"""Demo del módulo pipeline/preprocessing.py sobre un recorte de cromosoma.

Corre la cadena completa de preprocesamiento sobre una imagen y guarda una
tira comparativa con cada etapa intermedia + las métricas de calidad.

Uso:
    python demo_preprocessing.py <ruta_imagen> [ruta_salida.png]

Si no pasás ninguna ruta, usa un recorte de ejemplo del repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from pipeline import preprocessing as pp


def etiquetar(img: np.ndarray, texto: str) -> np.ndarray:
    """Pega una banda negra con el nombre de la etapa arriba de la imagen."""
    banda = np.zeros((28, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(banda, texto, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banda, img])


def main() -> None:
    # --- Entrada -----------------------------------------------------------
    if len(sys.argv) > 1:
        ruta = Path(sys.argv[1])
    else:
        # Recorte de ejemplo del repo (cromosoma tipo 1).
        ruta = Path(__file__).resolve().parents[2] / "TP_Final" / "aislados_test" / \
            "1" / "211029-011C_197_1_1080_477_0.927_1_13.png"
    salida = Path(sys.argv[2]) if len(sys.argv) > 2 else ruta.with_name(ruta.stem + "_etapas.png")

    print(f"Imagen de entrada : {ruta}")
    bgr = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise SystemExit(f"No pude leer la imagen: {ruta}")
    # cv2 lee en BGR; el módulo trabaja en RGB.
    if bgr.ndim == 3 and bgr.shape[2] >= 3:
        image = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        image = bgr

    # --- Cadena configurable: devuelve TODAS las etapas --------------------
    steps = pp.preprocess(
        image,
        do_denoise=True, denoise_method="nlm", denoise_strength=7.0,
        do_clahe=True, clahe_clip=1.5,
        do_normalize=False,
    )

    # --- Versión CANÓNICA (la que ve el clasificador) ---------------------
    canonica = pp.preprocess_for_classification(image)

    # --- Métricas de calidad ----------------------------------------------
    ref = steps["gray"]  # imagen gris de referencia
    res = steps["result"]
    print("\n--- Métricas ---")
    print(f"Ruido sigma (Immerkaer): {steps['sigma_in']:.2f}  ->  {steps['sigma_out']:.2f}")
    print(f"PSNR gris vs result : {pp.psnr(ref, res):.2f} dB")
    s = pp.ssim(ref, res)
    print(f"SSIM gris vs result : {s:.4f}" if s is not None else "SSIM: (scikit-image no disponible)")

    # --- Tira comparativa (RGB -> BGR para guardar con cv2) ---------------
    etapas = [
        (steps["gray"],     "1. gray"),
        (steps["denoised"], "2. denoised (NLM)"),
        (steps["clahe"],    "3. CLAHE"),
        (steps["result"],   "4. result"),
        (canonica,          "5. canonica"),
    ]
    tira = np.hstack([etiquetar(img, txt) for img, txt in etapas])
    cv2.imwrite(str(salida), cv2.cvtColor(tira, cv2.COLOR_RGB2BGR))
    print(f"\nTira comparativa guardada en: {salida}")


if __name__ == "__main__":
    main()
