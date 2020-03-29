import numpy as np
import pandas as pd
import time,gc
import cv2
import random
from tensorflow import keras
import matplotlib.image as mpimg
import tensorflow as tf
import matplotlib.pyplot as plt


train_df = pd.read_csv('bengaliai-cv19/train.csv')
test_df = pd.read_csv('bengaliai-cv19/test.csv')
class_map_df = pd.read_csv('bengaliai-cv19/class_map.csv')
sample_sub_df = pd.read_csv('bengaliai-cv19/sample_submission.csv')

#original image size
HEIGHT=137
WIDTH=236

print(train_df.shape)
print(test_df.shape)
print(class_map_df.shape)

Train = ['bengaliai-cv19/train_image_data_0.parquet','bengaliai-cv19/train_image_data_1.parquet',
        'bengaliai-cv19/train_image_data_2.parquet','bengaliai-cv19/train_image_data_3.parquet']
test = ['bengaliai-cv19/test_image_data_0.parquet','bengaliai-cv19/test_image_data_1.parquet',
       'bengaliai-cv19/test_image_data_2.parquet','bengaliai-cv19/test_image_data_3.parquet']

def read_and_save(train_dir,values,org_width,org_height,new_width,new_height,image_id):
    '''
    save images in png format (turn pd dataframe to png images)
    '''
    img = values.reshape((org_height,org_width)).astype(np.int16)
    cv2.imwrite(train_dir+str(image_id)+'.png',img)

def generate_images(df,train_dir,org_width,org_height,new_width,new_height):
    '''
    read images from dataframe
    '''

    image_ids = df['image_id'].values
    df = df.drop(['image_id'],axis=1)  #already drop the first column
    for image_id ,index in zip(image_ids,range(df.shape[0])):
        read_and_save(train_dir,df.iloc[index].values,org_width,org_height,new_width,new_height,image_id)


##After reading all images and save them to png files
##get training set and validation set based on the ratio of 9:1
df = pd.DataFrame(columns=['image_id','y1','y2','y3'],index=range(train_df.shape[0]))
df['image_id'] = train_df['image_id']
df['y1'] = train_df['grapheme_root']
df['y2'] = train_df['vowel_diacritic']
df['y3'] = train_df['consonant_diacritic']
df['image_id'] = df['image_id']  + '.png'

infer = np.random.permutation(range(200840))[:20084]
df_validation = df.iloc[infer]  #validation df
df_train = df.drop(infer,axis=0) #training df
#save them 
df_validation.to_csv('df_validation_0.csv')
df_train.to_csv('df_train_0.csv')
