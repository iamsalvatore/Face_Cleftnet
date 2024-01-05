
import matplotlib.pyplot as plt
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.layers.experimental import preprocessing
import pickle
from itertools import zip_longest
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input
AUTOTUNE = tf.data.experimental.AUTOTUNE

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session =tf.compat.v1.InteractiveSession(config=config)
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)



def training_data ():

    batch_size = 2
    input_height = 512
    input_width = 512

    train_directory = '/disk/scratch/dataset/Lahshal_categories_preprocessed_tts/Train/'
    valid_directory =  '/disk/scratch/dataset/Lahshal_categories_preprocessed_tts/Validate/'
    test_directory = '/disk/scratch/dataset/Lahshal_categories_preprocessed_tts/Test/'

    # All images will be rescaled by 1./255, 
    # and random augmentation are added to the training generator
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Note that validation and test generators must not use augmentations!
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_directory,
        # All images will be resized to 150x150
        target_size=(input_height, input_width),
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')


    validation_generator = validation_datagen.flow_from_directory(
        valid_directory,
        target_size=(input_height, input_width),
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_directory,
        target_size=(input_height, input_width),
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

    return train_generator, validation_generator, test_generator


train_generator, validation_generator,test_generator = training_data()



def InceptionV3_model():

    inputs = keras.Input(shape=(512, 512, 3))
    # outputs = rest_of_the_model(x)
    # model = keras.Model(inputs, outputs)
    # input_tensor = Input(shape=(512, 512, 3))


    model = InceptionV3(input_tensor=inputs, weights='imagenet', include_top=False)



    x= tf.cast(inputs, tf.float32)

    x = tf.keras.applications.inception_v3.preprocess_input(x)
    
    x = model.output
    # x = data_augmentation
    # x = preprocessing.Rescaling(1.0 / 255)(inputs)
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    x = layers.Dropout(0.5)(x)

    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(4, activation='softmax')(x)

    # this is the model we will train
    model = keras.Model(inputs=inputs, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    # for layer in model.layers:
    #     layer.trainable = False

    model.summary()

    return model

model = InceptionV3_model ()



def compile_fit_model(train_generator, validation_generator,model):

    model.compile(optimizer = RMSprop(lr=.0001), loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics = ['accuracy'])

        # Fit model using the generator method.
    history= model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        workers=6)
    
    # history = model.fit(train_ds, epochs=30, validation_data=val_ds, batch_size=2)
    
    model.save("InceptionV3_categorical_model")
    
    return history

history = compile_fit_model(train_generator, validation_generator,model)



def plot_loss():
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    # plt.show()
    plt.savefig("preprocessing_categorical_loss_inception.svg")
plot_loss()
    

def plot_acc():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(len(acc))
    plt.figure(figsize=(12, 8))
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig("preprocessing_categorical_acc_inception.svg")
plot_acc()



def evaluate_model(test_generator):

    loss, acc = model.evaluate(test_generator)
    print("Accuracy", acc)
    print("Loss",loss)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])
evaluate_model(test_generator)






