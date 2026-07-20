"""Convierte el clasificador de Keras (.h5) a ONNX para la interfaz.

La interfaz corre el clasificador con onnxruntime (sin TensorFlow). El .onnx no se
versiona (es grande y se deriva del .h5), así que se regenera con este script una
sola vez, en un entorno con TensorFlow.

Requisitos (entorno de conversión, NO el de la interfaz):
    pip install tensorflow-cpu==2.15.1 tf2onnx onnx

Uso:
    python convert_classifier_to_onnx.py \
        ../Modelos/Clasificacion/model_VGG_v2.h5 \
        ../Modelos/Clasificacion/model_VGG_v2.onnx

La interfaz espera el .onnx en Modelos/Clasificacion/model_VGG_v2.onnx.
"""
import sys

import tensorflow as tf
import tf2onnx


def main() -> None:
    h5 = sys.argv[1] if len(sys.argv) > 1 else "../Modelos/Clasificacion/model_VGG_v2.h5"
    onnx = sys.argv[2] if len(sys.argv) > 2 else "../Modelos/Clasificacion/model_VGG_v2.onnx"

    model = tf.keras.models.load_model(h5, compile=False)
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=15, output_path=onnx)
    print("ONNX guardado en", onnx)


if __name__ == "__main__":
    main()
