from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io
import os

port = int(os.environ.get("PORT", 5001))

app = Flask(__name__)

print("Загружаем модель...")

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(101, activation='softmax')
])

model.build((None, 224, 224, 3))
model.load_weights("model/efficient_weights.weights.h5")

classes = open("food-101/meta/classes.txt").read().splitlines()

print("Модель загружена! Сервер готов.")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_data = base64.b64decode(request.json["image"])
        img = Image.open(io.BytesIO(img_data)).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        index = np.argmax(predictions)
        confidence = float(predictions[0][index])

        return jsonify({
            "dish": classes[index],
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"Flask сервер запущен на порту {port}")
    app.run(host="0.0.0.0", port=port)