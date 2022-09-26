#import os
#import csv
#import cv2
#import matplotlib.pyplot as plt
#import random
#import pprint

#import numpy as np
#from numpy import expand_dims

#%tensorflow_version 1.x
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras import backend as K
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense, Dropout, SpatialDropout2D
#from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau
#from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array 
   
  
#import sklearn
#from sklearn.model_selection import train_test_split
#import pandas as pd


class Network:
    def __init__(self):
        self.image_size = (420,280,3)
        self.model_path = '/home/min/sc/model'
        self.filepath = self.model_path + "/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        self.checkpoint = ModelCheckpoint(self.filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
        try:
            self.model_path_full = '/home/min/sc/model/model_8+8_laps_curves.h5'
            self.model = load_model(self.model_path_full)
            print("successfully load model")
        except:
            print("load model failed")
    
    def angle_predict(self,img):
        self.predict_angle = self.model.predict(img,batch_size = 32,verbose = 1)
        return self.predict_angle

