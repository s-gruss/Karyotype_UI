"""Interfaz de cariotipado automático — Streamlit.

Ejecuta el pipeline y visualiza cada etapa:
    original -> preprocesamiento -> segmentación -> cromosomas rectificados
             -> clasificación (pendiente) -> cariograma (pendiente)

Correr con:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from pipeline import preprocessing, segmentation, extraction, classification, karyogram

st.set_page_config(page_title="Cariotipado automático", layout="wide")

st.title("🧬 Cariotipado automático de cromosomas")
st.caption(
    "TP Final — Procesamiento Avanzado de Imágenes en Biología y Biomedicina (16.85). "
    "Pipeline: preprocesamiento → segmentación → extracción/registración → clasificación → cariograma."
)

# --------------------------------------------------------------------------- #
# Sidebar: configuración
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Configuración")
    ts_path = st.text_input(
        "Modelo de segmentación (TorchScript)",
        value=segmentation.DEFAULT_TS_WEIGHTS,
    )
    score_thresh = st.slider("Umbral de confianza (segmentación)", 0.1, 0.9, 0.5, 0.05)

    st.divider()
    st.subheader("Preprocesamiento")
    do_denoise = st.checkbox("Denoising", value=True)
    denoise_method = st.selectbox("Método", ["nlm", "bilateral", "median"], index=0)
    denoise_strength = st.slider("Intensidad", 1.0, 20.0, 7.0, 1.0)
    do_clahe = st.checkbox("Realce de contraste (CLAHE)", value=True)
    clahe_clip = st.slider("CLAHE clip limit", 1.0, 8.0, 2.0, 0.5)
    feed_preproc_to_seg = st.checkbox(
        "Segmentar sobre la imagen preprocesada", value=False,
        help="Por defecto NO. El modelo se entrenó sobre imágenes crudas; "
             "cambiar la entrada en inferencia introduce un desfasaje train/test.",
    )


@st.cache_resource(show_spinner="Cargando modelo de segmentación…")
def get_segmenter(ts: str, thresh: float):
    return segmentation.load_segmenter(ts_path=ts, score_thresh=thresh)


@st.cache_resource(show_spinner="Cargando clasificador…")
def get_classifier():
    return classification.load_classifier()


# --------------------------------------------------------------------------- #
# Carga de imagen
# --------------------------------------------------------------------------- #
uploaded = st.file_uploader(
    "Subí una imagen de metafase (cariotipo)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded is None:
    st.info("Esperando una imagen microscópica de metafase para procesar.")
    st.stop()

image = np.array(Image.open(uploaded).convert("RGB"))
# Estandarizamos a 1600x1600 (tamaño de entrenamiento): así la imagen, las
# máscaras y los recortes quedan todos en la misma resolución.
if image.shape[:2] != (segmentation.INPUT_SIZE, segmentation.INPUT_SIZE):
    image = cv2.resize(image, (segmentation.INPUT_SIZE, segmentation.INPUT_SIZE))

# --------------------------------------------------------------------------- #
# Etapa 1 — Preprocesamiento
# --------------------------------------------------------------------------- #
st.header("1 · Preprocesamiento")
steps = preprocessing.preprocess(
    image,
    do_denoise=do_denoise, denoise_method=denoise_method, denoise_strength=denoise_strength,
    do_clahe=do_clahe, clahe_clip=clahe_clip,
)
c1, c2 = st.columns(2)
with c1:
    st.image(image, caption="Original", use_container_width=True)
with c2:
    st.image(steps["result"], caption="Preprocesada", use_container_width=True)
st.caption(
    f"Ruido estimado (σ, método de Immerkær): {steps['sigma_in']:.2f} → {steps['sigma_out']:.2f}. "
    "El preprocesamiento es una herramienta de visualización: los modelos se entrenaron "
    "sobre la imagen cruda, así que segmentación y clasificación se ejecutan sobre la "
    "imagen cruda (por consistencia train/inferencia) salvo que lo cambies en la barra lateral."
)

seg_input = steps["result"] if feed_preproc_to_seg else image

# --------------------------------------------------------------------------- #
# Etapa 2 — Segmentación
# --------------------------------------------------------------------------- #
st.header("2 · Segmentación (Mask R-CNN)")

backend = segmentation.available_backend(ts_path)
if backend is None:
    st.warning(
        "No hay backend de segmentación disponible. Exportá el modelo a TorchScript "
        "(ver la sección de exportación del notebook 2_Segmentación.ipynb) y colocá `model_ts.ts` en "
        f"`{segmentation.DEFAULT_TS_WEIGHTS}`. El resto de la interfaz funciona igual."
    )
    st.stop()

st.caption(f"Backend activo: **{backend}**")

if st.button("Ejecutar segmentación", type="primary"):
    with st.spinner("Segmentando cromosomas… (CPU, puede tardar unos segundos)"):
        model, backend = get_segmenter(ts_path, score_thresh)
        result = segmentation.segment(model, backend, seg_input, score_thresh)
        # La rama de clasificación recorta desde la imagen CRUDA: el clasificador se
        # entrenó sobre recortes crudos, así que en inferencia debe recibir recortes
        # crudos (consistencia train/inferencia). El preprocesamiento queda para
        # visualización, no en el camino de los modelos.
        chromosomes = extraction.extract_chromosomes(image, result.masks)
        # Clasificación (si el modelo ONNX está disponible)
        clf_indices = clf_labels = clf_probs = None
        if classification.onnx_available():
            clf = get_classifier()
            clf_indices, clf_labels, clf_probs = classification.classify(clf, chromosomes)
    st.session_state["seg_result"] = result
    st.session_state["chromosomes"] = chromosomes
    st.session_state["clf_indices"] = clf_indices
    st.session_state["clf_labels"] = clf_labels
    st.session_state["clf_probs"] = clf_probs

if "seg_result" in st.session_state:
    result = st.session_state["seg_result"]
    st.image(result.overlay, caption=f"{result.count} cromosomas detectados", use_container_width=True)

    # ------------------------------------------------------------------- #
    # Etapa 3 — Extracción + registración
    # ------------------------------------------------------------------- #
    st.header("3 · Extracción y rectificación (PCA)")
    chromosomes = st.session_state["chromosomes"]
    st.caption(
        f"{len(chromosomes)} cromosomas recortados **desde la imagen cruda** "
        "(la misma entrada con la que se entrenó el clasificador), "
        "rotados a orientación vertical (eje principal por PCA) y escalados a 224×224 "
        "preservando el tamaño relativo."
    )
    cols = st.columns(8)
    for i, chrom in enumerate(sorted(chromosomes, key=lambda c: -c.length_px)):
        with cols[i % 8]:
            st.image(chrom.image, use_container_width=True)

    # ------------------------------------------------------------------- #
    # Etapa 4 — Clasificación (VGG16 + SE, vía ONNX)
    # ------------------------------------------------------------------- #
    st.header("4 · Clasificación")
    clf_indices = st.session_state.get("clf_indices")
    clf_labels = st.session_state.get("clf_labels")
    clf_probs = st.session_state.get("clf_probs")

    if clf_indices is None:
        st.info(
            "Clasificador no disponible: falta el modelo ONNX en "
            f"`{classification.DEFAULT_ONNX_WEIGHTS}`. Convertí el `.h5` a ONNX "
            "(ver Interfaz/README.md) o copialo ahí. El resto del pipeline funciona igual."
        )
    else:
        st.caption(
            "Cada recorte de la etapa 3 se clasifica en su tipo (1–22, X, Y) con el "
            "modelo VGG16 + atención SE (exportado a ONNX, corre con onnxruntime)."
        )
        order = sorted(range(len(chromosomes)), key=lambda i: -chromosomes[i].length_px)
        cols = st.columns(8)
        for slot, i in enumerate(order):
            with cols[slot % 8]:
                st.image(chromosomes[i].image, use_container_width=True)
                st.caption(f"**{clf_labels[i]}** ({clf_probs[i]*100:.0f}%)")

        # --------------------------------------------------------------- #
        # Etapa 5 — Cariograma
        # --------------------------------------------------------------- #
        st.header("5 · Cariograma")
        kimg, cells = karyogram.assemble_karyogram(chromosomes, clf_indices)
        st.image(kimg, caption="Cariograma ensamblado (grilla estándar 1–22, X, Y)",
                 use_container_width=True)
        anomalias = karyogram.anomalies_summary(cells)
        if anomalias:
            st.warning("Posibles anomalías numéricas detectadas:\n\n- " + "\n- ".join(anomalias))
        else:
            st.success("Conteo por par sin anomalías numéricas (2 instancias por autosoma).")
