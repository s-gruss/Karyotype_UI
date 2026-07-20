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
       extracción + rectificación (PCA) ─► clasificación (VGG16+SE) ─► cariograma
```

- La segmentación trabaja sobre la imagen cruda.
- Los cromosomas individuales se recortan desde la imagen preprocesada (mismo
  preprocesamiento canónico que usa el entrenamiento del clasificador).
- El cariograma final se ensambla en la grilla estándar (24 pares), con conteo
  por par y marcado de anomalías numéricas.

## Resultados

Detalle completo en [Informe/informe_final.pdf](Informe/informe_final.pdf).

- **Segmentación** (Mask R-CNN, protocolo COCO sobre test): AP@0.5 = 97.8%
  (cajas) / 97.9% (máscaras), AP@0.75 > 94%.
- **Clasificación** (VGG16 + atención SE): exactitud ≈ 87% sobre las 24 clases
  cromosómicas.
- **Pipeline end-to-end** (interfaz, segmentación + clasificación encadenadas):
  ≈ 85% de exactitud, coherente con la clasificación aislada.

## Estructura

```
├── Pipelines/          Notebooks por etapa (1 preprocesamiento, 2 segmentación, 3 extracción, 4 clasificación)
├── Interfaz/           Interfaz Streamlit + paquete pipeline/ (preprocessing, segmentation, extraction, classification, karyogram)
├── Modelos/            Métricas de entrenamiento (los pesos .pth/.h5/.onnx NO están en git)
├── Informe/            Informe final (LaTeX + PDF) y figuras
├── PLAN_DE_TRABAJO.md  Planificación inicial del proyecto (checklist de la consigna)
├── Datasets/           AutoKary2022 (NO está en git, ver abajo)
└── Papers/ Clases/     Material de referencia (excluidos de git)
```

## Datos y modelos (no versionados)

Por tamaño, **no** están en el repo y se comparten aparte:
- **Dataset AutoKary2022** (~3.4 GB) → `Datasets/Autokary2022_1600x1600/`
- **Segmentación:** `model_final.pth` (Detectron2, 351 MB) y `model_ts.ts`
  (TorchScript, el que usa la interfaz) → `Modelos/Segmentacion/`
- **Clasificación:** `model_VGG_v2.h5` (Keras) y `model_VGG_v2.onnx`
  (el que usa la interfaz) → `Modelos/Clasificacion/`

La interfaz corre la segmentación con el modelo **TorchScript** (solo `torch`,
sin Detectron2) y la clasificación con **ONNX** (`onnxruntime`, sin TensorFlow).
Ambos se generan una sola vez a partir de los modelos originales — ver
[Interfaz/README.md](Interfaz/README.md) para el detalle de conversión.

## Interfaz

```bash
cd Interfaz
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Ver [Interfaz/README.md](Interfaz/README.md) para detalles (incluido cómo obtener
el modelo TorchScript para la etapa de segmentación).
