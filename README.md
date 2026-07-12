# Karyomatic: cariotipado automático de cromosomas

Trabajo Práctico Final — 16.85 Procesamiento Avanzado de Imágenes en Biología y Biomedicina.
Instituto Tecnológico de Buenos Aires (ITBA)

Alumnos:
- Santiago Gruss (63465) | sgruss@itba.edu.ar
- Agustín Miguel (63214) | agmiguel@itba.edu.ar

## Pipeline

```
imagen cruda ─► segmentación (Mask R-CNN) ─► máscaras
                                              │
         preprocesamiento (NLM + CLAHE) ──────┤
                                              ▼
                         extracción + rectificación (PCA) ─► clasificación (VGG16) ─► cariograma
```

- La segmentación trabaja sobre la imagen cruda.
- Los cromosomas individuales se recortan desde la imagen preprocesada (mismo
  preprocesamiento canónico que usa el entrenamiento del clasificador).

## Estructura

```
├── Pipelines/          Notebooks por etapa (1 preprocesamiento, 2 segmentación, 3 extracción, 4 clasificación)
├── Interfaz/           Interfaz Streamlit + paquete pipeline/ (preprocessing, segmentation, extraction)
├── Modelos/            Métricas de entrenamiento (los pesos .pth/.h5 NO están en git)
├── PLAN_DE_TRABAJO.md  Plan y estado del proyecto
├── Datasets/           AutoKary2022 (NO está en git, ver abajo)
└── Papers/ Clases/     Material de referencia (excluidos de git)
```

## Datos y modelos (no versionados)

Por tamaño, **no** están en el repo y se comparten aparte:
- **Dataset AutoKary2022** (~3.4 GB) → `Datasets/Autokary2022_1600x1600/`
- **Segmentación:** `model_final.pth` (Detectron2, 351 MB) y `model_ts.ts`
  (TorchScript, el que usa la interfaz) → `Modelos/Segmentacion/`
- **Modelo de clasificación** (VGG16) → `Modelos/Clasificacion/`

La interfaz corre la segmentación con el modelo **TorchScript** (solo `torch`,
sin Detectron2). Se genera desde Colab, en la sección de exportación a TorchScript
del notebook `Pipelines/2_Segmentación.ipynb`.

## Interfaz

```bash
cd Interfaz
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Ver [Interfaz/README.md](Interfaz/README.md) para detalles (incluido cómo obtener
el modelo TorchScript para la etapa de segmentación).
