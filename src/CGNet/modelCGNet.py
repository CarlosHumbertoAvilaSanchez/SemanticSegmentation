import os

import numpy as np
import tensorflow as tf
from config import imshape
from metricasCustom import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Add,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    GlobalAveragePooling2D,
    Input,
    Multiply,
    PReLU,
    UpSampling2D,
)
from keras.models import Model
from keras.optimizers import Adam


def BNPReLU(x):
    x = BatchNormalization(axis=3)(x)
    x = PReLU(shared_axes=[1, 2, 3])(x)
    return x


def CGBlockDown(input, kernels, reduction, d, number):
    output = Conv2D(
        kernels,
        (1, 1),
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        kernel_initializer="he_normal",
        name="Down" + str(number),
    )(input)
    output = BNPReLU(output)
    f_loc = DepthwiseConv2D((3, 3), kernel_initializer="he_normal", padding="same")(
        output
    )
    f_sur = DepthwiseConv2D(
        (3, 3), kernel_initializer="he_normal", dilation_rate=d, padding="same"
    )(output)

    f_join = Concatenate()([f_loc, f_sur])
    f_join = BNPReLU(f_join)
    f_join = Conv2D(
        kernels, (1, 1), padding="same", use_bias=False, kernel_initializer="he_normal"
    )(f_join)

    fc = GlobalAveragePooling2D()(f_join)
    fc = Dense(kernels // reduction, activation="relu")(fc)
    fc = Dense(kernels, activation="sigmoid")(fc)
    output = Multiply()([f_join, fc])
    return output


def CGBlock(input, reduction, d):
    n = input.shape[-1]
    n = n // 2
    output = Conv2D(
        n,
        (1, 1),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(input)
    output = BNPReLU(output)
    f_loc = DepthwiseConv2D((3, 3), kernel_initializer="he_normal", padding="same")(
        output
    )
    f_sur = DepthwiseConv2D(
        (3, 3), kernel_initializer="he_normal", dilation_rate=d, padding="same"
    )(output)

    f_join = Concatenate()([f_loc, f_sur])
    f_join = BNPReLU(f_join)

    f_join = Conv2D(
        input.shape[-1],
        (1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(input)

    fc = GlobalAveragePooling2D()(f_join)
    fc = Dense(input.shape[-1] // reduction, activation="relu")(fc)
    fc = Dense(input.shape[-1], activation="sigmoid")(fc)
    output = Multiply()([f_join, fc])

    output = Add()([input, output])

    return output


def modelCGNet():
    IMG_HEIGHT = imshape[0]
    IMG_WIDTH = imshape[1]
    IMG_CHANNELS = imshape[2]
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    output0 = Conv2D(
        32,
        (3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="layer1",
    )(inputs)
    output0 = BNPReLU(output0)
    output0 = Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="layer2",
    )(output0)
    output0 = BNPReLU(output0)
    output0 = Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="layer3",
    )(output0)
    output0 = BNPReLU(output0)

    inp1 = AveragePooling2D(
        pool_size=(3, 3), strides=2, padding="same", name="downsample1"
    )(inputs)

    inp2 = AveragePooling2D(pool_size=(3, 3), strides=2, padding="same")(inputs)
    inp2 = AveragePooling2D(
        pool_size=(3, 3), strides=2, padding="same", name="downsample2"
    )(inp2)

    output0_cat = Concatenate(name="cat1")([output0, inp1])
    output0_cat = BNPReLU(output0_cat)

    output1_0 = CGBlockDown(output0_cat, 64, 8, 2, 1)
    output1 = CGBlock(output1_0, 8, 2)
    output1 = CGBlock(output1, 8, 2)

    output1_cat = Concatenate(name="cat2")([output1, output1_0, inp2])
    output1_cat = BNPReLU(output1_cat)

    output2_0 = CGBlockDown(output1_cat, 128, 16, 4, 2)
    output2 = CGBlock(output2_0, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)
    output2 = CGBlock(output2, 8, 2)

    output2_cat = Concatenate(name="cat3")([output2, output2_0])
    output2_cat = BNPReLU(output2_cat)

    output = Conv2D(2, (1, 1), activation="softmax")(output2_cat)
    outputs = UpSampling2D(size=(8, 8), interpolation="bilinear")(output)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[dice, dice_coef, dicePerClass, "accuracy", iou_coef],
    )
    model.summary()
    return model
