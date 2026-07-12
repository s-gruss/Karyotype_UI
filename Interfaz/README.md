# Interfaz — Cariotipado automático

Interfaz gráfica en **Streamlit** que ejecuta el pipeline de cariotipado y
visualiza cada etapa del procesamiento.

## Pipeline

```
original → preprocesamiento → segmentación → extracción/registración → clasificación → cariograma
```

- **Preprocesamiento** (`pipeline/preprocessing.py`): denoising (NLM) + CLAHE.
  Por defecto NO se aplica antes de la segmentación (el modelo se entrenó sobre
  imágenes crudas); sí se aplica a la rama de clasificación (parámetros canónicos
  en `preprocess_for_classification`).
- **Segmentación** (`pipeline/segmentation.py`): Mask R-CNN ResNet-50 FPN.
  Corre el modelo exportado a **TorchScript** (`model_ts.ts`) con solo `torch`,
  sin necesidad de Detectron2. Trabaja a 1600×1600.
- **Extracción/registración** (`pipeline/extraction.py`): recorte por máscara,
  rectificación a vertical por eje principal (PCA) y escalado a 224×224.
- **Clasificación / cariograma**: pendientes (integración del modelo VGG16).

## Instalación

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt  # streamlit, opencv, torch, etc. (no requiere Detectron2)
```

## Modelo de segmentación

La interfaz necesita `model_ts.ts` en `../Modelos/Segmentacion/`. Se genera una
sola vez desde Colab (donde está Detectron2 y el modelo entrenado), en la sección
de exportación a TorchScript del notebook `Pipelines/2_Segmentación.ipynb`, que
exporta el `.pth` a TorchScript y verifica la paridad. El archivo es grande
(>100 MB) → se comparte por Drive, no por git.

## Uso

```bash
streamlit run streamlit_app.py
```

Subí una imagen de metafase y recorré las etapas. Si falta `model_ts.ts`, la
interfaz avisa y el resto de las etapas siguen disponibles.
