"""Pipeline de cariotipado automático.

Módulos:
    preprocessing  -- mejoramiento de imagen (denoising, CLAHE, normalización)
    segmentation   -- Mask R-CNN (Detectron2) para segmentación de instancias
    extraction     -- recorte + rectificación PCA de cada cromosoma a 224x224

Cada módulo es independiente de Streamlit para poder reutilizarlo en los
notebooks y en tests.
"""
