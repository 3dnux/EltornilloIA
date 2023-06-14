"""
    Dev :: Eduardo ZV
    Copyright :: EL TORNILLO 2023

    Proyecto dedicado a la detección de ventas en las sucursales de la empresa EL TORNILLO. Su versión 
    inicial solo detecta la interacción en caja. Su versión a futuro podrá detectar una venta completa,
    desde que el cliente llega hasta que este se va.
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam

# Cargamos el modelo preentrenado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Añadimos nuevas capas al final del modelo
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Creamos el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilamos el modelo
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Creamos generadores de datos
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'ruta/a/tu/carpeta',    # Cambia esto a la ruta de tu carpeta
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training')

validation_generator = datagen.flow_from_directory(
    'ruta/a/tu/carpeta',    # Cambia esto a la ruta de tu carpeta
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation')

# Entrenamos el modelo
model.fit(train_generator, validation_data=validation_generator, epochs=10)
