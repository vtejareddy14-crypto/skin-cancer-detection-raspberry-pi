import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load ONNX model
onnx_model = onnx.load("skin_cancer_model.onnx")

# Convert to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph("tf_model")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete: model.tflite created")