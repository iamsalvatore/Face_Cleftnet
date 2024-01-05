import numpy as np
import io
from keras.utils import np_utils
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
import pickle
import dlib
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import train_test_split
from itertools import zip_longest


def build_traindata():
    data_path='/disk/scratch/dataset/final_dataset_preprocessed'
    categories=os.listdir(data_path)
    labels=[i for i in range(len(categories))]
    label_dict=dict(zip(categories,labels)) #empty dictionary
    DATADIR = "/disk/scratch/dataset/final_dataset_preprocessed/"
    CATEGORIES = ["normal", "cleft"] 
    data=[]
    target=[]
    img_size=512
    for category in categories:
        path = os.path.join(DATADIR,category)
        for img in tqdm(os.listdir(path)):
                img_array = cv2.imread(os.path.join(path,img))
                width = 512
                height = 512
                dim = (width, height)
                img = cv2.resize(img_array, dim, interpolation = cv2.INTER_AREA)  
                new_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)          
                data.append(new_img)
                target.append(label_dict[category])
    return data,target

data,target=build_traindata()

def reshape(data,target):
    data=np.array(data)
    data=np.reshape(data,(data.shape[0],512,512,3))
    target=np.array(target)
    # target=np_utils.to_categorical(target)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)
    # normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    # X_train = np.reshape(X_train,(X_train.shape[0],img_size,img_size,3))
    # X_test = np.reshape(X_test,(X_test.shape[0],img_size,img_size,3))
    return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test=reshape(data,target)


def compile_fit_model(X_train,y_train):
    model = ResNet50(include_top = False, weights = 'imagenet', input_shape = (512,512, 3))

    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = 'res_net_50-'+timestr #
    # checkpoint_path = "checkpoints/"+name+"/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # os.system('mkdir {}'.format(checkpoint_dir))

    tensorboard_callback = TensorBoard(
    log_dir='tensorboard_logs/'+name,
    histogram_freq=1)

    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    history = model.fit(x=X_train,y=y_train,epochs=10,verbose=1,validation_split=0.2,batch_size=32,callbacks=[tensorboard_callback])
    model.save("ResNet50_model")

compile_fit_model(X_train,y_train)

