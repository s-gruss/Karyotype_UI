# Modelos de segmentación

Los pesos no se versionan (superan el límite de GitHub; ver `.gitignore`). Se
comparten por Google Drive. `metrics.json` (métricas de entrenamiento) sí está en el repo.

Para correr la interfaz, colocar aquí:

- **`model_ts.ts`** — Mask R-CNN exportado a TorchScript. Es el único que necesita la
  interfaz (corre solo con `torch`, sin Detectron2).

Opcional (solo para reentrenar o re-exportar):
- `model_final.pth` — pesos de Detectron2. La exportación a TorchScript está en el
  notebook `Pipelines/2_Segmentación.ipynb`.
