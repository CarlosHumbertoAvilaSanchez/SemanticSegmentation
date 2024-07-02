import cv2
import modelUnetSeparabletf2 as mu
import numpy as np
import tensorflow as tf
from config import clases, imshape
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

IMAGE_FILE_PATH = "../images/00105.jpg"
WEIGHTS_FILE_PATH = "pesosunetseparable256final.h5"

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
def dice(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


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


def get_resize_image(image_path, image_dimentions):
    prediction_image = cv2.imread(image_path)
    prediction_image = cv2.resize(prediction_image, image_dimentions)
    cv2.imshow("input", prediction_image)
    prediction_image = np.expand_dims(prediction_image, axis=0) / 255
    return prediction_image


image_dimentions = (IMAGE_WIDTH, IMAGE_HEIGHT)

prediction_image = get_resize_image(IMAGE_FILE_PATH, image_dimentions)

# img-=0.5
print(prediction_image.shape)

model = mu.modelUnet()
model.load_weights(WEIGHTS_FILE_PATH)
preds_train = np.array(model.predict(prediction_image))

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
"""


import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('C:\\Users\\Gabriel\\Google Drive (malloc.pro@gmail.com)\\datasetAereo\\videos\\video5.mp4')
model=mu.modelUnet()
model.load_weights('pesosmodelUnetNorm.h5')
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
counter=0
while(cap.isOpened()):
  # Capture frame-by-frame
  counter+=1
  
    
  ret, frame = cap.read()
  if ret == True:
    if  counter%15 != 0:    
      continue
    if counter ==1000000:
      counter=0
    frame= cv2.resize(frame,image_dimentions)

    # Display the resulting frame
    cv2.imshow('Frame',frame)
    frame = np.expand_dims(frame, axis=0)/255
    preds_train = np.array(model.predict(frame))
    salidaNew=salida2RGB(preds_train.squeeze(),clases)
    salidaNew=cv2.cvtColor(salidaNew,cv2.COLOR_BGR2RGB)
    #print(salidaNew.shape)
    cv2.imshow('salidaBien',salidaNew)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
"""
