# Modelos de clasificación

Los pesos no se versionan (superan el límite de GitHub; ver `.gitignore`). Se
comparten por Google Drive.

Para correr la interfaz, colocar aquí:

- **`model_VGG_v2.onnx`** — clasificador (VGG16 + SE) exportado a ONNX. Es el único
  que necesita la interfaz.

Opcionales (solo para reentrenar o regenerar el ONNX):
- `model_VGG_v2.h5` — modelo original en Keras.
- Regeneración del `.onnx` desde el `.h5`: ver `Interfaz/convert_classifier_to_onnx.py`.
