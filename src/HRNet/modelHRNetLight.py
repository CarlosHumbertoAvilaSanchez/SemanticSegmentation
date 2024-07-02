import os

import numpy as np
import tensorflow as tf
from config import imshape
from metricasCustom import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    Lambda,
    MaxPooling2D,
    UpSampling2D,
    add,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def bottleneck_Block(input, out_filters, strides=(1, 1)):

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer="he_normal")(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(
        out_filters,
        (3, 3),
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(int(out_filters), 1, use_bias=False, kernel_initializer="he_normal")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    if input.shape[3] != out_filters:
        residual = Conv2D(
            out_filters,
            1,
            strides=strides,
            use_bias=False,
            kernel_initializer="he_normal",
        )(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation("relu")(x)
    return x


def stemPart(inputs):
    x = Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    x = Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = bottleneck_Block(x, 64)
    x = bottleneck_Block(x, 64)
    x = bottleneck_Block(x, 64)
    x = bottleneck_Block(x, 64)
    x = Conv2D(
        16,
        (3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    return x


def transitionLayer1(n11, out_filters_list=[32, 64]):
    n21 = n11
    n22 = Conv2D(
        out_filters_list[1],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n11)
    n22 = BatchNormalization(axis=3)(n22)
    n22 = Activation("relu")(n22)

    return n21, n22


def transitionLayer2(n21, n22, out_filters_list=[32, 64, 128]):

    from_n21_n31 = n21

    from_n21_n32 = Conv2D(
        out_filters_list[1],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n21)
    from_n21_n32 = BatchNormalization(axis=3)(from_n21_n32)
    from_n21_n32 = Activation("relu")(from_n21_n32)

    from_n21_n33 = Conv2D(
        out_filters_list[1],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n21)
    from_n21_n33 = BatchNormalization(axis=3)(from_n21_n33)

    from_n21_n33 = Conv2D(
        out_filters_list[2],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(from_n21_n33)
    from_n21_n33 = BatchNormalization(axis=3)(from_n21_n33)
    from_n21_n33 = Activation("relu")(from_n21_n33)

    from_n22_n31 = Conv2D(
        out_filters_list[0], 1, use_bias=False, kernel_initializer="he_normal"
    )(n22)
    from_n22_n31 = BatchNormalization(axis=3)(from_n22_n31)
    from_n22_n31 = UpSampling2D(size=(2, 2))(from_n22_n31)

    from_n22_n32 = n22

    from_n22_n33 = Conv2D(
        out_filters_list[2],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n22)
    from_n22_n33 = BatchNormalization(axis=3)(from_n22_n33)
    from_n22_n33 = Activation("relu")(from_n22_n33)

    n31 = add([from_n21_n31, from_n22_n31])
    n32 = add([from_n21_n32, from_n22_n32])
    n33 = add([from_n21_n33, from_n22_n33])

    return n31, n32, n33


def transitionLayer3(n31, n32, n33, out_filters_list=[32, 64, 128, 256]):

    # from n31 block
    from_n31_n41 = n31

    from_n31_n42 = Conv2D(
        out_filters_list[1],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n31)
    from_n31_n42 = BatchNormalization(axis=3)(from_n31_n42)
    from_n31_n42 = Activation("relu")(from_n31_n42)

    from_n31_n43 = Conv2D(
        out_filters_list[1],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n31)
    from_n31_n43 = BatchNormalization(axis=3)(from_n31_n43)

    from_n31_n43 = Conv2D(
        out_filters_list[2],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(from_n31_n43)
    from_n31_n43 = BatchNormalization(axis=3)(from_n31_n43)
    from_n31_n43 = Activation("relu")(from_n31_n43)

    from_n31_n44 = Conv2D(
        out_filters_list[1],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n31)
    from_n31_n44 = BatchNormalization(axis=3)(from_n31_n44)

    from_n31_n44 = Conv2D(
        out_filters_list[2],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(from_n31_n44)
    from_n31_n44 = BatchNormalization(axis=3)(from_n31_n44)

    from_n31_n44 = Conv2D(
        out_filters_list[3],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(from_n31_n44)
    from_n31_n44 = BatchNormalization(axis=3)(from_n31_n44)
    from_n31_n44 = Activation("relu")(from_n31_n44)

    # from 32 block
    from_n32_n41 = Conv2D(
        out_filters_list[0], 1, use_bias=False, kernel_initializer="he_normal"
    )(n32)
    from_n32_n41 = BatchNormalization(axis=3)(from_n32_n41)
    from_n32_n41 = UpSampling2D(size=(2, 2))(from_n32_n41)

    from_n32_n42 = n32

    from_n32_n43 = Conv2D(
        out_filters_list[2],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n32)
    from_n32_n43 = BatchNormalization(axis=3)(from_n32_n43)
    from_n32_n43 = Activation("relu")(from_n32_n43)

    from_n32_n44 = Conv2D(
        out_filters_list[2],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n32)
    from_n32_n44 = BatchNormalization(axis=3)(from_n32_n44)
    from_n32_n44 = Conv2D(
        out_filters_list[3],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(from_n32_n44)
    from_n32_n44 = BatchNormalization(axis=3)(from_n32_n44)
    from_n32_n44 = Activation("relu")(from_n32_n44)

    # from n33_block
    from_n33_n41 = Conv2D(
        out_filters_list[0], 1, use_bias=False, kernel_initializer="he_normal"
    )(n33)
    from_n33_n41 = BatchNormalization(axis=3)(from_n33_n41)
    from_n33_n41 = UpSampling2D(size=(4, 4))(from_n33_n41)

    from_n33_n42 = Conv2D(
        out_filters_list[1], 1, use_bias=False, kernel_initializer="he_normal"
    )(n33)
    from_n33_n42 = BatchNormalization(axis=3)(from_n33_n42)
    from_n33_n42 = UpSampling2D(size=(2, 2))(from_n33_n42)

    from_n33_n43 = n33

    from_n33_n44 = Conv2D(
        out_filters_list[3],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n33)
    from_n33_n44 = BatchNormalization(axis=3)(from_n33_n44)
    from_n33_n44 = Activation("relu")(from_n33_n44)

    n41 = add([from_n31_n41, from_n32_n41, from_n33_n41])
    n42 = add([from_n31_n42, from_n32_n42, from_n33_n42])
    n43 = add([from_n31_n43, from_n32_n43, from_n33_n43])
    n44 = add([from_n31_n44, from_n32_n44, from_n33_n44])

    return n41, n42, n43, n44


def transitionLayer4(n41, n42, n43, n44, out_filters_list=[32, 64, 128, 256]):

    # from n41 block
    from_n41_n41 = n41

    from_n41_n42 = Conv2D(
        out_filters_list[1],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n41)
    from_n41_n42 = BatchNormalization(axis=3)(from_n41_n42)
    from_n41_n42 = Activation("relu")(from_n41_n42)

    from_n41_n43 = Conv2D(
        out_filters_list[1],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n41)
    from_n41_n43 = BatchNormalization(axis=3)(from_n41_n43)

    from_n41_n43 = Conv2D(
        out_filters_list[2],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(from_n41_n43)
    from_n41_n43 = BatchNormalization(axis=3)(from_n41_n43)
    from_n41_n43 = Activation("relu")(from_n41_n43)

    from_n41_n44 = Conv2D(
        out_filters_list[1],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n41)
    from_n41_n44 = BatchNormalization(axis=3)(from_n41_n44)

    from_n41_n44 = Conv2D(
        out_filters_list[2],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(from_n41_n44)
    from_n41_n44 = BatchNormalization(axis=3)(from_n41_n44)

    from_n41_n44 = Conv2D(
        out_filters_list[3],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(from_n41_n44)
    from_n41_n44 = BatchNormalization(axis=3)(from_n41_n44)
    from_n41_n44 = Activation("relu")(from_n41_n44)

    # from 42 block
    from_n42_n41 = Conv2D(
        out_filters_list[0], 1, use_bias=False, kernel_initializer="he_normal"
    )(n42)
    from_n42_n41 = BatchNormalization(axis=3)(from_n42_n41)
    from_n42_n41 = UpSampling2D(size=(2, 2))(from_n42_n41)

    from_n42_n42 = n42

    from_n42_n43 = Conv2D(
        out_filters_list[2],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n42)
    from_n42_n43 = BatchNormalization(axis=3)(from_n42_n43)
    from_n42_n43 = Activation("relu")(from_n42_n43)

    from_n42_n44 = Conv2D(
        out_filters_list[2],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n42)
    from_n42_n44 = BatchNormalization(axis=3)(from_n42_n44)
    from_n42_n44 = Conv2D(
        out_filters_list[3],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(from_n42_n44)
    from_n42_n44 = BatchNormalization(axis=3)(from_n42_n44)
    from_n42_n44 = Activation("relu")(from_n42_n44)

    # from n43_block
    from_n43_n41 = Conv2D(
        out_filters_list[0], 1, use_bias=False, kernel_initializer="he_normal"
    )(n43)
    from_n43_n41 = BatchNormalization(axis=3)(from_n43_n41)
    from_n43_n41 = UpSampling2D(size=(4, 4))(from_n43_n41)

    from_n43_n42 = Conv2D(
        out_filters_list[1], 1, use_bias=False, kernel_initializer="he_normal"
    )(n43)
    from_n43_n42 = BatchNormalization(axis=3)(from_n43_n42)
    from_n43_n42 = UpSampling2D(size=(2, 2))(from_n43_n42)

    from_n43_n43 = n43

    from_n43_n44 = Conv2D(
        out_filters_list[3],
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(n43)
    from_n43_n44 = BatchNormalization(axis=3)(from_n43_n44)
    from_n43_n44 = Activation("relu")(from_n43_n44)

    # from n44_block
    from_n44_n41 = Conv2D(
        out_filters_list[0], 1, use_bias=False, kernel_initializer="he_normal"
    )(n44)
    from_n44_n41 = BatchNormalization(axis=3)(from_n44_n41)
    from_n44_n41 = UpSampling2D(size=(8, 8))(from_n44_n41)

    from_n44_n42 = Conv2D(
        out_filters_list[1], 1, use_bias=False, kernel_initializer="he_normal"
    )(n44)
    from_n44_n42 = BatchNormalization(axis=3)(from_n44_n42)
    from_n44_n42 = UpSampling2D(size=(4, 4))(from_n44_n42)

    from_n44_n43 = Conv2D(
        out_filters_list[2], 1, use_bias=False, kernel_initializer="he_normal"
    )(n44)
    from_n44_n43 = BatchNormalization(axis=3)(from_n44_n43)
    from_n44_n43 = UpSampling2D(size=(2, 2))(from_n44_n43)

    from_n44_n44 = n44

    n41 = add([from_n41_n41, from_n42_n41, from_n43_n41, from_n44_n41])
    n42 = add([from_n41_n42, from_n42_n42, from_n43_n42, from_n44_n42])
    n43 = add([from_n41_n43, from_n42_n43, from_n43_n43, from_n44_n43])
    n44 = add([from_n41_n44, from_n42_n44, from_n43_n44, from_n44_n44])

    return n41, n42, n43, n44


def basic_Block(input, out_filters, strides=(1, 1)):
    x = Conv2D(
        out_filters,
        3,
        padding="same",
        strides=strides,
        use_bias=False,
        kernel_initializer="he_normal",
    )(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(
        out_filters,
        3,
        padding="same",
        strides=strides,
        use_bias=False,
        kernel_initializer="he_normal",
    )(x)
    x = BatchNormalization(axis=3)(x)
    x = add([x, input])
    x = Activation("relu")(x)
    return x


def make_branch(x, out_filters=32):
    x = basic_Block(x, out_filters)
    x = basic_Block(x, out_filters)
    x = basic_Block(x, out_filters)
    x = basic_Block(x, out_filters)
    return x


def concatRes(n41, n42, n43, n44):
    n42up = UpSampling2D(size=(2, 2))(n42)
    n43up = UpSampling2D(size=(4, 4))(n43)
    n44up = UpSampling2D(size=(8, 8))(n44)
    return concatenate([n41, n42up, n43up, n44up])


def modelHRNet():
    IMG_HEIGHT = imshape[0]
    IMG_WIDTH = imshape[1]
    IMG_CHANNELS = imshape[2]
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # stem_net part
    n11 = stemPart(inputs)
    n21, n22 = transitionLayer1(n11, [16, 32])
    n21 = make_branch(n21, 16)
    n22 = make_branch(n22, 32)
    n31, n32, n33 = transitionLayer2(n21, n22, [16, 32, 64])
    n31 = make_branch(n31, 16)
    n32 = make_branch(n32, 32)
    n33 = make_branch(n33, 64)
    n41, n42, n43, n44 = transitionLayer3(n31, n32, n33, [16, 32, 64, 128])
    n41 = make_branch(n41, 16)
    n42 = make_branch(n42, 32)
    n43 = make_branch(n43, 64)
    n44 = make_branch(n44, 128)
    n41, n42, n43, n44 = transitionLayer4(n41, n42, n43, n44, [16, 32, 64, 128])
    concatenated = concatRes(n41, n42, n43, n44)
    outputs = Conv2D(2, (1, 1), activation="softmax")(concatenated)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[dice, dice_coef, dicePerClass, "accuracy", iou_coef],
    )
    model.summary()
    return model
