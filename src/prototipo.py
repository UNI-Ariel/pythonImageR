# Se importan todas las librerias que seran utilizadas para el aprendizaje
import os
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.utils import image_dataset_from_directory

# Se establecen parametros que se usaran para el entrenamiento
# tamaño de imagen de entrada, directorio del dataset, batch, epochs
# y nombre del modelo resultante
image_size = (180, 180)
data_dir = "dataset"
batch_size = 20
epochs= 10
outfile= "garbage_classifier.h5"

# Se cargan las imagenes del dataset para crear un modelo
train_ds = image_dataset_from_directory(
  data_dir,
  image_size=image_size,
  batch_size=batch_size,
  validation_split=0.2,
  subset="training", #Entrenar
  seed=100,
)

#Se cargan las mismas imagenes para hacer la validacion
val_ds = image_dataset_from_directory(
  data_dir,
  image_size=image_size,
  batch_size=batch_size,
  validation_split=0.2,
  subset="validation",  # Validar
  seed=100,
)

# Los nombres de las clases del dataset
class_names = train_ds.class_names
print(class_names)

# Construir el modelo
model = models.Sequential()
# Agregar capas
# Redimencionar las imagenes
model.add(layers.Rescaling(1./255, input_shape=(180,180,3)))
# Agregar capas con filtros
model.add(layers.Conv2D(32, 3, activation='relu'))
# agregar una capa limite para la caracterización
model.add(layers.MaxPooling2D())

# Repetir el proceso
model.add(layers.Conv2D(64, 3, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128, 3, activation='relu'))
model.add(layers.MaxPooling2D())

# Agregar capa que aplasta (convierte) la salida de capas anteriores
# a una capa densa unidimensional (reformateo de datos)
model.add(layers.Flatten())
# Dense conecta todas las salidas de la capa anterior a todas las 
# neuronas de esta capa
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(class_names))

# Compilar el modelo
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

# Mostrar detalles de las capas
model.summary()

# Comenzar el entrenamiento
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Guardar el modelo entrenado
model.save(outfile)
