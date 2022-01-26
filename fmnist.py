#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import mean
import sys, time, os, warnings 
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# In[2]:


class CNNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        model.add(Conv2D(32, (3, 3), padding="same",
        input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model


# In[3]:


# load dataset
def load_dataset():
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# In[4]:


def pixels_conversion(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


# In[5]:


def compile_model():
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model = CNNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[6]:


# evaluate a model
def start():
    NUM_EPOCHS=20
    trainX, trainY, testX, testY = load_dataset()
    print('FMNIST Dataset Loaded')
    trainX, testX = pixels_conversion(trainX, testX)
    print('FMNIST pixels are prepared.')
    print('FMNIST Model fitting is started.')
    model = compile_model()
    start = time.time()
    H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=NUM_EPOCHS, batch_size=512, shuffle=True, verbose=0)
    end = time.time()
    print("FMNIST Model is completed in {:3.2f}MIN".format((end - start )/60))
    model.save('final_model.h5')
    labelNames = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    preds = model.predict(testX)
    # show a nicely formatted classification report
    print("FMNIST evaluation report...")
    print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=labelNames))
    # plot the training loss and accuracy
    N = NUM_EPOCHS
    plt.style.use("ggplot")
    #plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")
    return model, testX, testY, labelNames


# In[7]:


# Run the model
def run():
    model, testX, testY, labelNames = start()


# In[8]:


# run test
run()


# In[ ]:




