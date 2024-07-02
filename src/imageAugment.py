import json
import os
import pickle

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from config import clases, imshape
from imgaug.augmentables import Keypoint, KeypointsOnImage
from PIL import Image


def salida2RGB(salida, clases):
    heigth = salida.shape[0]
    width = salida.shape[1]
    salidaRGB = np.zeros((salida.shape[0] * salida.shape[1], 3), dtype=np.uint8)
    salida = salida.reshape(-1, len(clases))
    for idx, x in enumerate(salida):
        salidaRGB[idx] = clases[np.argmax(x)]["rgb"]
    salidaRGB = salidaRGB.reshape((heigth, width, 3))
    return salidaRGB


ia.seed(1)
n_classes = len(clases)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
sometimes2 = lambda aug: iaa.Sometimes(0.05, aug)


class DataGenerator(tf.keras.utils.Sequence):
    # Generates data for Keras
    def __init__(
        self, image_paths, annot_paths, batch_size=1, shuffle=True, augment=False
    ):
        self.image_paths = image_paths
        self.annot_paths = annot_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        image_paths = [self.image_paths[k] for k in indexes]
        annot_paths = [self.annot_paths[k] for k in indexes]

        X, y = self.__data_generation(image_paths, annot_paths)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def find(self, lst, key, value):
        for i, dic in enumerate(lst):
            # print(i)
            # print(dic)
            if dic[key] == value:
                return i
        return -1

    def resizePoints(self, pointsList, imageSize):
        nuevosPuntos = []
        """
      print(imageSize)
      print(imshape)
      
      print(pointsList)
      """
        for point in pointsList:
            pointX = imshape[1] / imageSize[1] * point[0]
            pointY = imshape[0] / imageSize[0] * point[1]
            nuevosPuntos.append([pointX, pointY])
        """
      print('-----------------------------------')
      print(nuevosPuntos)
      print('**************')
      """
        return nuevosPuntos

    def get_poly(self, annot_path):
        #
        with open(annot_path) as handle:
            data = json.load(handle)

        shape_dicts = data["shapes"]

        return shape_dicts

    def create_multi_masks(self, im, shape_dicts, tama):

        chanels = []
        clsInImage = []
        polysInImage = []
        # por cada poligono hay que cada uno de sus puntos al tamaño de la entrada de la imagen
        for x in shape_dicts:
            # obtener el canal destinado a la clase del poligono
            clsInImage.append(self.find(clases, "label", x["label"]))
            # agregar el poligono
            polysInImage.append(np.array(x["points"], dtype=np.int32))

        # crear el canal del background
        background = np.zeros(shape=im.shape[0:2], dtype=np.uint8)
        # crear el canal para la imagen final
        finalImage = np.zeros(shape=im.shape, dtype=np.uint8)
        # agregar un canal para cada clase
        for l in clases:
            chanels.append(np.zeros(shape=im.shape[0:2], dtype=np.float32))
        # Por cada clase obtenemos el id de la clase y el indice de su poligono correspondiente
        for i, channel in enumerate(clsInImage):
            # agregamos el poligno al background
            cv2.fillPoly(background, [polysInImage[i]], 255)
            # agregamos el poligno al canal correspondiente
            cv2.fillPoly(chanels[channel], [polysInImage[i]], 255)
            # agregamos el poligono a la imagen final
            cv2.fillPoly(finalImage, [polysInImage[i]], clases[channel]["rgb"])
            # cv2.imshow(clases[channel]['label'],chanels[channel])
            # cv2.waitKey()
        # el canal sero es el background por eso se obtiene el complemento de la union de todos los poligonos
        chanels[0] = cv2.bitwise_not(background)

        chanels = np.array(chanels)
        Y = np.stack(chanels, axis=2)
        Y = Y / 255.0
        return Y

    def augment_anotations(self, im, anotations):
        keyAnotations = []
        for a in anotations:
            for idxK, k in enumerate(a["points"]):
                keyAnotations.append(Keypoint(x=k[0], y=k[1]))

        kps = KeypointsOnImage(keyAnotations, shape=im.shape)
        seq = iaa.Sequential(
            [
                iaa.Multiply((0.9, 1.2)),  # change brightness, doesn't affect keypoints
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(
                    rotate=10, scale=(0.9, 1.2), translate_percent=(-0.1, 0.1)
                ),  # rotate by exactly 10deg and scale to 50-70%, affects keypoints
                sometimes(iaa.Crop(px=(0, 10))),
                sometimes2(iaa.Fog()),
            ]
        )
        # Augment keypoints and images.
        image_aug, kps_aug = seq(image=im, keypoints=kps)

        contador = 0
        for a in anotations:
            for idxK, k in enumerate(a["points"]):
                kps_aug.keypoints[contador]
                a["points"][idxK] = [
                    kps_aug.keypoints[contador].x,
                    kps_aug.keypoints[contador].y,
                ]
                contador += 1
        """
      ia.imshow(
          np.hstack([
              kps.draw_on_image(im, size=7),
              kps_aug.draw_on_image(image_aug, size=7)
          ])
      )
      """
        return image_aug, anotations

    def __data_generation(self, image_paths, annot_paths):
        # se crea el arreglo para la imagen
        X = np.empty(
            (self.batch_size, imshape[0], imshape[1], imshape[2]), dtype=np.float32
        )
        # se crea el arreglo para las clases
        Y = np.empty(
            (self.batch_size, imshape[0], imshape[1], n_classes), dtype=np.float32
        )
        # por cada archivo generar sus salida
        for i, (im_path, annot_path) in enumerate(zip(image_paths, annot_paths)):
            # print(im_path)
            # if para omitir el archivo .ini
            if im_path.split(".")[-1] != "ini":
                # Se lee la imagen
                im = cv2.imread(im_path, 1)
                # im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                # obtener el tamaño de la imagen original
                tama = im.shape
                # se escala la imagen al tamaño de la entrada de la red
                im = cv2.resize(im, (imshape[1], imshape[0]))
                # cv2.imshow('imagengen',im)
                # se obtienen los poligonos de las clases anotadas en cada imagen
                anotations = self.get_poly(annot_path)
                # redimensionar los puntos para que cuadren con el tamño de la entrada de la red
                # print(anotations)
                for x in anotations:
                    x["points"] = self.resizePoints(x["points"], tama)
                # print(anotations[0])

                # check for augmentation
                if self.augment:
                    im, anotations = self.augment_anotations(im, anotations)
                # print(anotations[0])
                mask = self.create_multi_masks(im, anotations, tama)
                """
              cv2.imshow('backgrouns',mask[:,:,0])
              cv2.waitKey()
              """
                im = im / 255.0
                X[i,] = im
                # para que quede normalizada a la salida
                Y[i,] = mask

                # print(im_path)
                # print(X)
                # np.save('imageninput',X)

        return X, Y


# # para probar el generador
# dirImagenes = "../images"
# dirAnotaciones = "../labels"


# def sorted_fns(dir):
#     return sorted(os.listdir(dir), key=lambda x: x.split(".")[0])


# print(sorted_fns(dirImagenes))
# image_paths = [os.path.join(dirImagenes, x) for x in sorted_fns(dirImagenes)]
# annot_paths = [os.path.join(dirAnotaciones, x) for x in sorted_fns(dirAnotaciones)]
# tg = DataGenerator(
#     image_paths=image_paths, annot_paths=annot_paths, batch_size=1, augment=False
# )


# imagenes, salidas = tg.__getitem__(0)
# # print(imagenes)
# print(imagenes.shape)
# print(salidas.shape)
# print("tipo imagen", imagenes[0].dtype)

# salida = salidas.squeeze()
# for i in range(len(clases)):
#     cv2.imshow(str(clases[i]["label"]), salida[:, :, i])
#     print(salida[:, :, i].dtype)
# cv2.imshow("imagen", np.uint8(imagenes[0] * 255))
# cv2.imshow("salidaRGB", salida2RGB(salidas[0], clases))
# cv2.waitKey()
