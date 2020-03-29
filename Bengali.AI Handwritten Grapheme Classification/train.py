import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
import time, gc
import cv2
import tensorflow as tf
import keras
from keras import backend as K
import matplotlib.pyplot as plt
import efficientnet.keras as efn
from utils import add_mask,GridMask,onehot,val_generator,train_generator
import efficientnet.tfkeras
from keras.models import load_model

PATH = 'bengaliai-cv19/training/'

def create_model(input_shape):
    '''
    base model is efficientnetB3(B4)
    '''
    input = keras.Input(input_shape)
    base_model = efn.EfficientNetB3(weights='imagenet',include_top=False,
                                   input_tensor=input,pooling='avg')
    for layer in base_model.layers:
        layer.trainable=True
    x = base_model.output
    root = keras.layers.Dense(units=168,activation='softmax',name='root')(x)
    vowel=keras.layers.Dense(units=11,activation='softmax',name='vowel')(x)
    consonant=keras.layers.Dense(units=7,activation='softmax',name='consonant')(x)
    model = keras.models.Model(base_model.input,[root,vowel,consonant],name='efficientnet3')

    return model

model = create_model((300,300,3))
print(model.summary()) #see the model
df_train = pd.read_csv('df_train_0.csv')
df_validation = pd.read_csv('df_validation_0.csv')

#make generator
train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
df_train['y1'] = df_train['y1'].astype('str')
df_validation['y1'] = df_validation['y1'].astype('str')
df_train['y2'] = df_train['y2'].astype('str')
df_validation['y2'] = df_validation['y2'].astype('str')
df_train['y3'] = df_train['y3'].astype('str')
df_validation['y3'] = df_validation['y3'].astype('str')
train_Gen = train_gen.flow_from_dataframe(df_train,directory=PATH,x_col='image_id',y_col=['y1','y2','y3'],target_size=(300,300),validate_filenames=False,batch_size=8,class_mode="multi_output")
training_Gen = train_generator(train_Gen,add_mask,8)
val_Gen = validation_gen.flow_from_dataframe(df_validation,directory=PATH,x_col='image_id',y_col=['y1','y2','y3'],target_size=(300,300),validate_filenames=False,class_mode="multi_output",batch_size=4)
validation_Gen = val_generator(val_Gen)

model.compile(keras.optimizers.Adam(lr=4e-4),loss={'root':'categorical_crossentropy','vowel':'categorical_crossentropy','consonant':'categorical_crossentropy'},loss_weights = {'root': 0.60,
                                'vowel': 0.20,
                                'consonant': 0.20},
              metrics={'root':['accuracy'],'vowel':['accuracy'],'consonant':['accuracy']})

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        '''
        change lr
        '''
        if epoch==0 or epoch==1 or epoch==3 or epoch==5 or epoch == 7 or epoch == 8 or epoch ==9 :
            lr = K.get_value(self.model.optimizer.lr)
            lr = lr * 0.4
            if lr >= 5e-6:
                K.set_value(self.model.optimizer.lr,lr)
        print(logs)

mycallback = myCallback()
callback2 = keras.callbacks.ModelCheckpoint('models/whole/1.{epoch:02d}-{val_loss:.2f}.hdf5',save_best_only=False)

model.fit_generator(training_Gen,steps_per_epoch=22510,epochs=15,validation_data=validation_Gen,validation_steps=1256,callbacks=[mycallback,callback2],verbose=0)
