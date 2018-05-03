# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:17:33 2018

@author: Adrin
"""

import os,cv2
import multiprocessing
os.environ['THEANO_FLAGS'] = "floatX=float32,openmp=True" 
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())

# Use scikit-learn to grid search the batch size and epochs
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential

from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier


import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
''' change this as per your preference of use of backend to th/tf'''
K.set_image_dim_ordering('th')      

from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam

#%%
'''*** change this as per your images ***'''
using_grayscale_or_color_flag = "grayscale"     #or color
using_callbacks_flag = True         #True or False

#PATH = os.getcwd()
# Define data path
#data_path = PATH + '/data'
data_path = "C:/Users/Adrin/Desktop/keras_bin/airplaneNotairplane/data/train"
data_dir_list = os.listdir(data_path)

img_rows=256
img_cols=256
#num_channel=1               #ie. for multispectral -- 3 .... for grayscale -- 1
if using_grayscale_or_color_flag == "grayscale":
    num_channel=1            
else:
    num_channel=3

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        #use it for grayscale images
		if using_grayscale_or_color_flag == "grayscale":
		    input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(256,256))
		img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)
		
#%%
# create model
def create_model(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True):
    input_shape=img_data[0].shape
    model = Sequential()
    
    model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))
    
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    '''THIS'''
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    
    return model
#%%
# Define the number of classes
num_classes = 2
batch_size = 32
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:296]=0
labels[296:]=1

labelNames = ['airplane','notAirplane']
	  
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)

'''If this train_test_split is used then use it in fitting the model '''
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)



#%%

# create model
model = KerasClassifier(build_fn=create_model, verbose=1)
# define the grid search parameters
batch_size = [32, 64]   #, 40, 60, 80, 100
epochs = [2, 5]   #for epochs better use earlyStopping method
lrs=[0.01, 0.001]      #1e-2
decays=[1e-6]
momentums=[0.8, 0.9]     #0.8,
nesterovs=[True]

param_grid = dict(batch_size=batch_size, epochs=epochs, momentum=momentums, decay=decays)

#if __name__ == '__main__':       #not working :(
grid = GridSearchCV(estimator=model,  cv=KFold(2), param_grid=param_grid, verbose=20, n_jobs=1)   #n_jobs=-1 means use all cores 
grid_result = grid.fit(X_train, y_train)

#hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_split=0.2)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))