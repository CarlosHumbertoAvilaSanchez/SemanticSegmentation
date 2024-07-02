import os

import numpy as np
import tensorflow as tf
from config import imshape
from metricasCustom import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    Add,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    DepthwiseConv2D,
    Input,
    MaxPool2D,
    PReLU,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def BNPReLU(x):
    x = BatchNormalization(axis=3)(x)
    x = PReLU(shared_axes=[1, 2, 3])(x)
    return x


def downSampleBlock(x, ins, outs):
    if ins < outs:
        outputFilters = outs - ins
    else:
        outputFilters = outs
    output = Conv2D(
        outputFilters,
        (3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(x)
    if ins < outs:
        maxPool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        output = Concatenate()([output, maxPool])
    output = BNPReLU(output)

    return output


def DABModule(input, d=1, module_name="MyBlock"):
    with K.name_scope(module_name):
        outputCahnnels = input.shape[-1] // 2
        output = BNPReLU(input)
        output = Conv2D(
            outputCahnnels,
            (3, 3),
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(output)
        output = BNPReLU(output)

        outputpadTB = ZeroPadding2D(padding=((1, 1), (0, 0)))(output)
        conv3x1 = DepthwiseConv2D((3, 1), kernel_initializer="he_normal")(outputpadTB)
        conv3x1 = BNPReLU(conv3x1)
        conv3x1 = ZeroPadding2D(padding=((0, 0), (1, 1)))(conv3x1)
        conv1x3 = DepthwiseConv2D((1, 3), kernel_initializer="he_normal")(conv3x1)
        conv1x3 = BNPReLU(conv1x3)

        outputpadTBD = ZeroPadding2D(padding=((1 * d, 1 * d), (0, 0)))(output)
        conv3x1d = DepthwiseConv2D(
            (3, 1), dilation_rate=((1 * d), 1), kernel_initializer="he_normal"
        )(outputpadTBD)
        conv3x1d = BNPReLU(conv3x1d)
        conv3x1d = ZeroPadding2D(padding=((0, 0), (1 * d, 1 * d)))(conv3x1d)
        conv1x3d = DepthwiseConv2D(
            (1, 3), dilation_rate=(1, 1 * d), kernel_initializer="he_normal"
        )(conv3x1d)
        conv1x3d = BNPReLU(conv1x3d)

        output = Add()([conv1x3, conv1x3d])
        output = BNPReLU(output)

        output = Conv2D(
            input.shape[-1],
            (1, 1),
            padding="valid",
            use_bias=False,
            kernel_initializer="he_normal",
        )(output)
        output = Add()([input, output])
    return output


def modelDABNet():
    IMG_HEIGHT = imshape[0]
    IMG_WIDTH = imshape[1]
    IMG_CHANNELS = imshape[2]
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    x = Conv2D(
        32,
        (3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="layer1",
    )(inputs)
    x = BNPReLU(x)
    x = Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="layer2",
    )(x)
    x = BNPReLU(x)
    x = Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="layer3",
    )(x)
    x = BNPReLU(x)

    down1 = AveragePooling2D(pool_size=(3, 3), strides=2, padding="same")(inputs)
    down2 = AveragePooling2D(pool_size=(3, 3), strides=2, padding="same")(down1)
    down3 = AveragePooling2D(pool_size=(3, 3), strides=2, padding="same")(down2)

    x = Concatenate()([x, down1])
    x = BNPReLU(x)
    # DAB Block 1
    downSample1 = downSampleBlock(x, 32 + 3, 64)

    x = DABModule(downSample1, d=2, module_name="DAB1")
    x = DABModule(x, d=2, module_name="DAB3")
    x = DABModule(x, d=2, module_name="DAB3")

    x = Concatenate()([x, downSample1, down2])
    x = BNPReLU(x)

    # DAB Block 2
    downSample2 = downSampleBlock(x, 128 + 3, 128)
    x = DABModule(downSample2, d=4)
    x = DABModule(x, d=4)
    x = DABModule(x, d=8)
    x = DABModule(x, d=8)
    x = DABModule(x, d=16)
    x = DABModule(x, d=16)

    x = Concatenate()([x, downSample2, down3])
    x = BNPReLU(x)

    output = Conv2D(2, (1, 1), activation="softmax")(x)
    outputs = UpSampling2D(size=(8, 8), interpolation="bilinear")(output)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[dice, "accuracy", iou_coef],
    )
    model.summary()
    return model
