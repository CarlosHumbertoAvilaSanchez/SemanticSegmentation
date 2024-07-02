from keras.callbacks import EarlyStopping, ModelCheckpoint
from imageAugment import DataGenerator
from keras.models import load_model
import modelUnettf2 as mu
import cv2
import numpy as np
import os
import tensorflow as tf
from config import clases
import pandas as pd
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
def normalizar(imagen):
  imagen /= 255
  return imagen

def salida2RGB(salida,clases):
  print(salida.shape)
  heigth=salida.shape[0]
  width=salida.shape[1]
  salidaRGB=np.zeros((salida.shape[0]*salida.shape[1],3),dtype=np.uint8)
  salida=salida.reshape(-1,len(clases))
  for idx,x in enumerate(salida):
    salidaRGB[idx]=clases[np.argmax(x)]['rgb']
  salidaRGB=salidaRGB.reshape((heigth,width,3))
  salidaRGB=cv2.cvtColor(salidaRGB,cv2.COLOR_BGR2RGB)
  return salidaRGB
dirImagenes='images'
dirAnotaciones='labels'
def sorted_fns(dir):
    return sorted(os.listdir(dir), key=lambda x: x.split('.')[0])


image_paths = [os.path.join(dirImagenes, x) for x in sorted_fns(dirImagenes)]
annot_paths = [os.path.join(dirAnotaciones, x) for x in sorted_fns(dirAnotaciones)]

tg = DataGenerator(image_paths=image_paths, annot_paths=annot_paths,
                   batch_size=8, augment=False)


filepath = "SegmentacionSemantica\\best_pesos_unet_norm.h5"
# Create a callback that saves the model's weights every 5 epochs
checkpoint = ModelCheckpoint(filepath,
                            monitor='dice',
                            verboe=1,
                            save_best_only=True,
                            mode='max')



modelo=mu.modelUnet()
#model = load_model("SegmentacionSemantica\modelUnetNorm.h5")
modelo.load_weights("SegmentacionSemantica\\pesosmodelUnetNorm.h5")
history=modelo.fit_generator(
    tg,
    callbacks=[checkpoint],
    steps_per_epoch=125,
    epochs=1000)
modelo.save("SegmentacionSemantica\modelUnetNorm.h5")
modelo.save_weights("SegmentacionSemantica\\pesosmodelUnetNorm.h5")

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 
 
# save to json:00  
hist_json_file = 'SegmentacionSemantica\historyUnetNormal.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv: 
hist_csv_file = 'SegmentacionSemantica\historyUnetNormal.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)





"""
for i,batch in enumerate(image_generator):
    if(i >= num_batch):
        break
for i,batch in enumerate(mask_generator):
    if(i >= num_batch):
        break
"""
"""
train_generator = zip(image_generator, mask_generator)

modelo.fit_generator(
    train_generator,
    steps_per_epoch=1000,
    epochs=20)
modelo.save("modelnormalizado20i.h5")
modelo.save_weights("pesosnormalizado20i.h5")
"""