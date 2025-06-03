from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model("model/garbage_classifier.keras")

# Define tus clases
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Asegura la existencia de la carpeta estática
os.makedirs("static", exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == "POST":
        img_file = request.files.get("image")
        if img_file:
            img_filename = os.path.join("static", img_file.filename)
            img_file.save(img_filename)

            # img = Image.open(img_filename).resize((180, 180)).convert("RGB")
            # img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            img = load_img(img_filename, target_size=(180, 180))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            # img_array /= 255.0

            pred = model.predict(img_array)[0]
            class_idx = np.argmax(pred)
            prediction = class_names[class_idx]
            confidence = float(pred[class_idx]) * 100
            img_path = img_filename

    return render_template("index.html", 
                           prediction=prediction, 
                           confidence=confidence, 
                           img_path=img_path, 
                           class_names=class_names)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
