# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:17:33 2018

@author: Adrin
"""

from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from keras.models import Sequential
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import os,cv2
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('th')      

from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam


#%%s

num_classes = 2 
num_epoch = 2
labelNames = ['airplane','notAirplane']
img_rows=256
img_cols=256    

''' hyperopt is reading the data from only these 2 functions data() and create_model() so 
whatever u want to initialise, do in these functions only '''

def data():
    using_grayscale_or_color_flag = "grayscale"    
    data_path = "C:/Users/Adrin/Desktop/keras_bin/airplaneNotairplane/data/train"
    data_dir_list = os.listdir(data_path) 
    img_data_list=[]  
    for dataset in data_dir_list:
    	img_list=os.listdir(data_path+'/'+ dataset)
    	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    	for img in img_list:
    		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
    		if using_grayscale_or_color_flag == "grayscale":
    		    input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    		input_img_resize=cv2.resize(input_img,(256,256))
    		img_data_list.append(input_img_resize)
    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    img_data= np.expand_dims(img_data, axis=1) 
    
    num_of_samples = img_data.shape[0]
    labels = np.ones((num_of_samples,),dtype='int64')
    labels[0:296]=0
    labels[296:]=1 
    Y = np_utils.to_categorical(labels, 2)
    x,y = shuffle(img_data,Y, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    return X_train, y_train, X_test, y_test, img_data


#%%
def create_model(X_train, y_train, X_test, y_test, img_data):
    input_shape=img_data[0].shape
    model = Sequential()
    
    model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense({{choice([128, 256, 512])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(Dense({{choice([128, 256, 512])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    
    model.fit(X_train, y_train, batch_size={{choice([64, 128])}}, epochs=5, verbose=2, validation_split=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

    
if __name__ == '__main__': 
    X_train, y_train, X_test, y_test, img_data = data()
    
    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=5, trials=Trials())
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

