import time

import cv2
import modelResNet18 as mu
import numpy as np
import tensorflow as tf
from config import clases, imshape
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
def dice(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def salida2RGB(salida, clases):
    # print(salida.shape)
    heigth = salida.shape[0]
    width = salida.shape[1]
    salidaRGB = np.zeros((salida.shape[0] * salida.shape[1], 3), dtype=np.uint8)
    salida = salida.reshape(-1, len(clases))
    for idx, x in enumerate(salida):
        salidaRGB[idx] = clases[np.argmax(x)]["rgb"]
    salidaRGB = salidaRGB.reshape((heigth, width, 3))
    salidaRGB = cv2.cvtColor(salidaRGB, cv2.COLOR_BGR2RGB)
    return salidaRGB


width = 256
height = 256
dim = (width, height)

import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture("../testVideo/test.mp4")
model = mu.modelResNet18()
model.load_weights("best_pesos_Exfuse2.h5")
# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")

# Read until video is completed
counter = 0
start = time.time()
while cap.isOpened():
    # Capture frame-by-frame
    counter += 1

    ret, frame = cap.read()
    if ret == True:
        if counter % 15 != 0:
            continue
        if counter == 1000000:
            counter = 0
        frameExp = cv2.resize(frame, dim)

        # Display the resulting frame
        # cv2.imshow('Frame',frame)
        frameExp = np.expand_dims(frameExp, axis=0) / 255
        preds_train = np.array(model.predict(frameExp))
        salidaNew = salida2RGB(preds_train.squeeze(), clases)
        # print(salidaNew.shape)
        # cv2.imshow('salidaBien',salidaNew)
        salidaNew = cv2.resize(salidaNew, (1920, 1080))
        frame = (frame * 0.8).astype("uint8")
        todo = salidaNew // 5 + frame
        cv2.imshow("salidaBien", todo)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
end = time.time()
print("tiempo total", end - start)
