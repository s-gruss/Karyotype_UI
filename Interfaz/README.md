# Interfaz — Cariotipado automático

Interfaz gráfica en **Streamlit** que ejecuta el pipeline de cariotipado y
visualiza cada etapa del procesamiento.

## Pipeline

```
original → preprocesamiento → segmentación → extracción/registración → clasificación → cariograma
```

- **Preprocesamiento** (`pipeline/preprocessing.py`): denoising (NLM/bilateral/mediana),
  CLAHE, normalización + estimación de ruido. Por defecto NO se aplica antes de la
  segmentación (el modelo se entrenó sobre imágenes crudas); beneficia a la
  visualización y a la clasificación.
- **Segmentación** (`pipeline/segmentation.py`): Mask R-CNN ResNet-50 FPN (Detectron2),
  entrada 1600×1600, 1 clase. Usa `Modelos/Segmentacion/model_final.pth`.
- **Extracción/registración** (`pipeline/extraction.py`): recorte por máscara,
  rectificación a vertical por eje principal (PCA) y escalado a 224×224.
- **Clasificación / cariograma**: pendientes (integración del modelo VGG16).

## Instalación

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# Detectron2 (aparte, según tu CUDA/PyTorch):
python -m pip install "git+https://github.com/facebookresearch/detectron2.git"
```

## Uso

```bash
streamlit run streamlit_app.py
```

Subí una imagen de metafase y recorré las etapas. Si Detectron2 no está
instalado, la interfaz avisa y las etapas de preprocesamiento siguen disponibles.

## Nota

La versión anterior en Flask (`app.py`, `templates/`) queda como referencia y
puede eliminarse una vez validada la migración a Streamlit.
