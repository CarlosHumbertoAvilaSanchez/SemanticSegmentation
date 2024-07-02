import os

import cv2
import modelUnetSeparableLargeFiltertf2 as mu
import numpy as np
import pandas as pd
import tensorflow as tf
from config import clases
from imageAugment import DataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

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


dirImagenes = "../testImage"
dirAnotaciones = "../testLabel"


def sorted_fns(dir):
    return sorted(os.listdir(dir), key=lambda x: x.split(".")[0])


image_paths = [os.path.join(dirImagenes, x) for x in sorted_fns(dirImagenes)]
annot_paths = [os.path.join(dirAnotaciones, x) for x in sorted_fns(dirAnotaciones)]

tg = DataGenerator(
    image_paths=image_paths, annot_paths=annot_paths, batch_size=1, augment=False
)

width = 256
height = 256
dim = (width, height)
img = cv2.imread("../testImage/00042.jpg")  # CAMBIAR
img = cv2.resize(img, dim)
cv2.imshow("input", img)
img = np.expand_dims(img, axis=0) / 255

modelo = mu.modelUnet()
modelo.load_weights("best_pesos_norm_separableLF.h5")
modelo.evaluate(tg)

algo = modelo.evaluate(tg)

preds_train = np.array(modelo.predict(img))

# print(preds_train)
salidaFinal = preds_train.squeeze()
print("salida ", salidaFinal.shape)
print("tamaño de prediction", preds_train.shape)
for i in range(len(clases)):
    cv2.imshow(clases[i]["label"], salidaFinal[:, :, i])
    # print(salidaFinal[:,:,i].dtype)
    cv2.waitKey()
# cv2.imshow("salida",np.uint8(salidaFinal*255))

salidaNew = salida2RGB(preds_train.squeeze(), clases)
# print(salidaNew.shape)
cv2.imshow("salidaBien", salidaNew)
cv2.waitKey()
cv2.destroyAllWindows()
