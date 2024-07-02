import numpy as np
import tensorflow as tf
from keras import backend as K


def dice(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
    return dice


def dicePerClass(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def dice_vect(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.dot(y_true, K.transpose(y_pred))
    union = K.dot(y_true, K.transpose(y_true)) + K.dot(y_pred, K.transpose(y_pred))
    return (2.0 * intersection + smooth) / (union + smooth)


def iou_coef(y_true, y_pred, smooth=1):
    # arreglo=tf.make_ndarray(y_pred)
    # np.save('predictedEval',arreglo)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
