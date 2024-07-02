import os

import cv2
import modelCGNet as mu
import numpy as np
import pandas as pd
import tensorflow as tf
from config import clases
from imageAugment import DataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def normalizar(imagen):
    imagen /= 255
    return imagen


def salida2RGB(salida, clases):
    print(salida.shape)
    heigth = salida.shape[0]
    width = salida.shape[1]
    salidaRGB = np.zeros((salida.shape[0] * salida.shape[1], 3), dtype=np.uint8)
    salida = salida.reshape(-1, len(clases))
    for idx, x in enumerate(salida):
        salidaRGB[idx] = clases[np.argmax(x)]["rgb"]
    salidaRGB = salidaRGB.reshape((heigth, width, 3))
    salidaRGB = cv2.cvtColor(salidaRGB, cv2.COLOR_BGR2RGB)
    return salidaRGB


dirImagenes = "images"  # img
dirAnotaciones = "labels"  # json


def sorted_fns(dir):
    return sorted(os.listdir(dir), key=lambda x: x.split(".")[0])


image_paths = [os.path.join(dirImagenes, x) for x in sorted_fns(dirImagenes)]
annot_paths = [os.path.join(dirAnotaciones, x) for x in sorted_fns(dirAnotaciones)]

tg = DataGenerator(
    image_paths=image_paths, annot_paths=annot_paths, batch_size=8, augment=False
)  # Cambiar a false, batch = 8

filepath = "SegmentacionSemantica\\best_pesos_CGNet.h5"
# Create a callback that saves the model's weights every 5 epochs
checkpoint = ModelCheckpoint(
    filepath, monitor="dice", verbose=1, save_best_only=True, mode="max"
)

modelo = mu.modelCGNet()

# model = load_model('modelPrebatch512v2.h5', custom_objects={'dice': mu.dice})
modelo.load_weights("SegmentacionSemantica\\pesosCGNet.h5")
history = modelo.fit_generator(
    tg,
    steps_per_epoch=125,  # cambiar por cantidad de imagenes
    callbacks=[checkpoint],
    epochs=200,
)
modelo.save("SegmentacionSemantica\\modelCGNet.h5")
modelo.save_weights("SegmentacionSemantica\\pesosCGNet.h5")
# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(history.history)

# save to json:
hist_json_file = "SegmentacionSemantica\\historyCGNet.json"
with open(hist_json_file, mode="w") as f:
    hist_df.to_json(f)

# or save to csv:
hist_csv_file = "SegmentacionSemantica\\historyCGNet.csv"
with open(hist_csv_file, mode="w") as f:
    hist_df.to_csv(f)
