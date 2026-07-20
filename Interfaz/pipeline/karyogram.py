"""Ensamblado del cariograma (etapa de reporte / diagnóstico).

Toma los cromosomas ya recortados y rectificados (salida de extraction.py) junto
con la clase predicha por el clasificador, y los ordena en la grilla estándar del
cariograma: pares 1--22 seguidos de los cromosomas sexuales X e Y. Cuenta las
instancias por clase y marca posibles anomalías numéricas (un número de cromosomas
por clase distinto de dos).

Mapeo de clases
---------------
El clasificador se entrenó con `flow_from_directory` de Keras sobre carpetas
nombradas '1'..'24'. Keras ordena las clases ALFABÉTICAMENTE como strings, de modo
que el índice de salida NO coincide con el número de cromosoma. `CLASS_INDEX_TO_LABEL`
codifica ese orden real, y las etiquetas '23' y '24' corresponden a X e Y.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Orden EXACTO de clases tal como lo produce flow_from_directory (sorted() sobre
# los nombres de carpeta '1'..'24' tratados como strings).
CLASS_INDEX_TO_LABEL = sorted([str(i) for i in range(1, 25)])
# -> ['1','10','11','12','13','14','15','16','17','18','19','2','20','21','22',
#     '23','24','3','4','5','6','7','8','9']


def index_to_chromosome(idx: int) -> str:
    """Índice de salida del clasificador (0--23) -> nombre de cromosoma ('1'..'22','X','Y')."""
    label = CLASS_INDEX_TO_LABEL[idx]
    if label == "23":
        return "X"
    if label == "24":
        return "Y"
    return label


# Disposición estándar del cariograma (filas de la grilla).
KARYOGRAM_LAYOUT = [
    ["1", "2", "3", "4", "5"],
    ["6", "7", "8", "9", "10", "11", "12"],
    ["13", "14", "15", "16", "17", "18"],
    ["19", "20", "21", "22", "X", "Y"],
]

# Número esperado de instancias por clase (2 = un par). Para X/Y el conteo normal
# depende del sexo (XX o XY); no se marca anomalía en los sexuales por defecto.
_AUTOSOMES = [str(i) for i in range(1, 23)]


@dataclass
class KaryogramCell:
    chromosome: str          # '1'..'22','X','Y'
    images: list             # recortes (PIL.Image) asignados a esta clase
    count: int
    anomaly: bool            # True si el conteo es inesperado (autosomas != 2)


def _load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def group_by_class(chromosomes, class_indices) -> dict:
    """Agrupa los recortes por nombre de cromosoma según la clase predicha."""
    groups: dict = {}
    for chrom, idx in zip(chromosomes, class_indices):
        name = index_to_chromosome(int(idx))
        groups.setdefault(name, []).append(chrom)
    return groups


def build_cells(chromosomes, class_indices) -> list[KaryogramCell]:
    """Construye una celda por clase de la grilla, con conteo y flag de anomalía."""
    groups = group_by_class(chromosomes, class_indices)
    cells: list[KaryogramCell] = []
    for row in KARYOGRAM_LAYOUT:
        for name in row:
            chroms = groups.get(name, [])
            # ordenar por tamaño descendente (los homólogos quedan juntos y prolijos)
            chroms = sorted(chroms, key=lambda c: -getattr(c, "length_px", 0.0))
            count = len(chroms)
            anomaly = name in _AUTOSOMES and count != 2
            cells.append(KaryogramCell(name, [c.image for c in chroms], count, anomaly))
    return cells


def assemble_karyogram(
    chromosomes,
    class_indices,
    cell_size: int = 110,
    max_per_cell: int = 4,
    background=(255, 255, 255),
) -> tuple[Image.Image, list[KaryogramCell]]:
    """Ensambla el cariograma en una única imagen ordenada por la grilla estándar.

    chromosomes: lista de objetos con atributo `.image` (recorte 224x224) y
                 opcionalmente `.length_px` (para ordenar los homólogos).
    class_indices: clase predicha por cada cromosoma (índice 0--23 del clasificador).

    Devuelve (imagen_del_cariograma, celdas) donde `celdas` permite inspeccionar
    conteos y anomalías.
    """
    cells = build_cells(chromosomes, class_indices)

    n_cols = max(len(r) for r in KARYOGRAM_LAYOUT)
    n_rows = len(KARYOGRAM_LAYOUT)

    thumb = cell_size
    label_h = 22
    pad = 8
    slot_w = max_per_cell * (thumb // 2) + pad  # ancho reservado por celda
    cell_w = slot_w
    cell_h = thumb + label_h

    W = n_cols * cell_w + pad
    H = n_rows * cell_h + pad
    canvas = Image.new("RGB", (W, H), background)
    draw = ImageDraw.Draw(canvas)
    font = _load_font(15)

    cell_iter = iter(cells)
    for r, row in enumerate(KARYOGRAM_LAYOUT):
        # centrar filas más cortas que la más ancha
        offset = (n_cols - len(row)) * cell_w // 2
        for c, _name in enumerate(row):
            cell = next(cell_iter)
            x0 = pad + offset + c * cell_w
            y0 = pad + r * cell_h

            # pegar cada homólogo (miniatura) lado a lado
            tw = thumb // 2
            for k, im in enumerate(cell.images[:max_per_cell]):
                t = im.resize((tw, thumb))
                canvas.paste(t, (x0 + k * tw, y0))

            # etiqueta: número de cromosoma + conteo; rojo si hay anomalía
            color = (200, 0, 0) if cell.anomaly else (0, 0, 0)
            texto = f"{cell.chromosome} ({cell.count})"
            draw.text((x0 + 2, y0 + thumb + 3), texto, fill=color, font=font)

    return canvas, cells


def anomalies_summary(cells: list[KaryogramCell]) -> list[str]:
    """Devuelve descripciones legibles de las anomalías numéricas detectadas."""
    out = []
    for cell in cells:
        if cell.anomaly:
            out.append(f"Cromosoma {cell.chromosome}: {cell.count} instancias (se esperaban 2)")
    return out
