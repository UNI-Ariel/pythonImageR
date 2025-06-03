# Ocultar mensajes de advertencia sobre GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Librerias necesarias tensorflow 2.10 numpy 1.26
from tensorflow.keras.models import load_model
import numpy as np
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory, load_img, img_to_array

# Cargar el dataset
data_dir = "dataset/Garbage classification"
ds = image_dataset_from_directory(data_dir, image_size=(180,180), batch_size=32)

# Obtener los classnames
class_names = ds.class_names

print(class_names)

# Cargar el modelo
model = load_model("model/garbage_classifier.keras", compile=False)

# Preparar imagen
#img = image.load_img("dataset/Garbage classification/paper/paper5.jpg", target_size=(180, 180))
#img_array = image.img_to_array(img)
#img_array = np.expand_dims(img_array, axis=0)
#img_array /= 255.0
img = load_img("dataset/Garbage classification/cardboard/cardboard5.jpg", target_size=(180,180))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predecir
prediction = model.predict(img_array)
print(prediction)

# Interpretar
predicted_class = np.argmax(prediction)
print("Prediccion:", class_names[predicted_class])