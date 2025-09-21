import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify


app = Flask(__name__)


model_path = "best_butterfly_model.keras"
model = load_model(model_path)


class_indices = {
    0: "class_name_0",
    1: "class_name_1",
    2: "class_name_2",
    
    
}

IMG_SIZE = (128, 128)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"})
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"})

        
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

    
        img = image.load_img(filepath, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = tf.expand_dims(img_array, axis=0)

        
        preds = model.predict(img_array)
        pred_class = class_indices[int(tf.argmax(preds[0]))]

    
        os.remove(filepath)

        return jsonify({"predicted_class": pred_class})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
