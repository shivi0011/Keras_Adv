"""
Created on Fri Mar 16 10:00:00 2018

@author: Adrin
"""
# Import libraries
import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
''' change this as per your preference of use of backend to th/tf'''
K.set_image_dim_ordering('th')      

from keras.utils import np_utils
from keras.models import Sequential
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

num_epoch=5

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
		
        
# =============================================================================
# '''first save the things which u want to change and use in model then retrieve those info from json file '''
# #save below as conf.json
# {
# "model" : "inceptionv3",
# "weights" : "None",
# "include_top" : false,
# "train_path" : "C:/Users/Adrin/Desktop/keras_bin/Inception/image_recognition/dataset/train",
# "test_path" : "C:/Users/Adrin/Desktop/keras_bin/Inception/image_recognition/dataset/test",
# "features_path" : "C:/Users/Adrin/Desktop/keras_bin/Inception/image_recognition/output/features.h5",
# "labels_path" : "C:/Users/Adrin/Desktop/keras_bin/Inception/image_recognition/output/labels.h5",
# "results" : "C:/Users/Adrin/Desktop/keras_bin/Inception/image_recognition/output/results.txt",
# "classifier_path" : "C:/Users/Adrin/Desktop/keras_bin/Inception/image_recognition/output/classifier.pickle",
# "model_path" : "C:/Users/Adrin/Desktop/keras_bin/Inception/image_recognition/output/model",
# "test_size" : 0.10,
# "seed" : 9,
# "num_classes" : 2
# }
# #use below in program
# import json
# import datetime
# import time
# # load the user configs
# with open("C:/Users/Adrin/Desktop/keras_bin/Inception/image_recognition/conf/conf.json") as f:    
#   config = json.load(f)
# 
# # config variables
# model_name    = config["model"]
# include_top   = config["include_top"]
# train_path    = config["train_path"]
# features_path   = config["features_path"]
# labels_path   = config["labels_path"]
# test_size     = config["test_size"]
# results     = config["results"]
# model_path    = config["model_path"]
# # start time
# print ("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
# start = time.time()
# 
# =============================================================================


#%%
USE_SKLEARN_PREPROCESSING=False

if USE_SKLEARN_PREPROCESSING:
	# using sklearn for preprocessing
	from sklearn import preprocessing
	print(">>>>>>using sklearn preprocessings<<<<<\n")
	
	def image_to_feature_vector(image, size=(256, 256)):
		# resize the image to a fixed size, then flatten the image into
		# a list of raw pixel intensities
		return cv2.resize(image, size).flatten()
	
	img_data_list=[]
	for dataset in data_dir_list:
		img_list=os.listdir(data_path+'/'+ dataset)
		print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
		for img in img_list:
			input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
			if using_grayscale_or_color_flag == "grayscale":
			    input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)     
			input_img_flatten=image_to_feature_vector(input_img,(256,256))
			img_data_list.append(input_img_flatten)
	
	img_data = np.array(img_data_list)
	img_data = img_data.astype('float32')
	print (img_data.shape)
	img_data_scaled = preprocessing.scale(img_data)
	print (img_data_scaled.shape)
	
	print (np.mean(img_data_scaled))
	print (np.std(img_data_scaled))
	
	print (img_data_scaled.mean(axis=0))
	print (img_data_scaled.std(axis=0))
	
	if K.image_dim_ordering()=='th':
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
		print (img_data_scaled.shape)
		
	else:
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
		print (img_data_scaled.shape)
	
	
	if K.image_dim_ordering()=='th':
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
		print (img_data_scaled.shape)
		
	else:
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
		print (img_data_scaled.shape)

if USE_SKLEARN_PREPROCESSING:
	img_data=img_data_scaled
#%%
# Assigning Labels

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
# Defining the model
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
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
'''OR THIS'''
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

# Viewing model_configuration

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable


#%%

'''Either This '''
# =======================================================================================
#Training
'''As we already splitted our data using train_test_split function so use it directly '''
if using_callbacks_flag == False:
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))
           
'''otherwise'''
#hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_split=0.2)
# =======================================================================================

'''Or This '''
# =======================================================================================
# Training with callbacks
'''-----put callbacks here-----'''
if using_callbacks_flag == True:
    from keras import callbacks
    filename="C:/Users/Adrin/Desktop/keras_bin/model_train_new.csv"
    csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
    
    early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    
    filepath="C:/Users/Adrin/Desktop/keras_bin/models_with_archi/model_with_callbacks/Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
    
    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    callbacks_list = [csv_log,early_stopping,checkpoint]
    
    hist = model.fit(X_train, y_train, batch_size=16, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)

# =======================================================================================
csv_fp = open(filename,"r")
linecount=0
for line in csv_fp:
    csv_fp.readline()
    linecount = linecount+1

epochs_ran = linecount-1
print("#f epochs ran are :",epochs_ran)

# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(epochs_ran)

plt.figure(1,figsize=(7,5))     
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True) 
plt.legend(['train','val'])
#print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
  

#%%

# Evaluating the model

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])

# Testing a new image
test_image = cv2.imread("C:/Users/Adrin/Desktop/keras_bin/airplaneTest/air_craft_31.jpg")
if using_grayscale_or_color_flag == "grayscale":
    test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)     
test_image=cv2.resize(test_image,(256,256))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
   
if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
# Predicting the test image
print((model.predict(test_image)))
print(model.predict_classes(test_image))

#%%

'''
One dangerous pitfall that can be easily noticed with this visualization is that some activation maps may be all zero
for many different inputs, which can indicate dead 􀀫lters, and can be a symptom of high learning ratesOne dangerous 
pitfall that can be easily noticed with this visualization is that some activation maps may be all zero
for many different inputs, which can indicate dead 􀀫lters, and can be a symptom of high learning rates.
'''

# Visualizing the intermediate layer

#
def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

layer_num=4
filter_num=0

activations = get_featuremaps(model, int(layer_num),test_image)

print (np.shape(activations))
feature_maps = activations[0][0]      
print (np.shape(feature_maps))

if K.image_dim_ordering()=='th':
	feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
print (feature_maps.shape)

fig=plt.figure(figsize=(6,6))
#if using_grayscale_or_color_flag == "grayscale":
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')        # for grayscale add --    ,cmap='gray'
#else:
#    plt.imshow(feature_maps[:,:,filter_num])
    
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.jpg')

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))	
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
	if using_grayscale_or_color_flag == "grayscale":
	    ax.imshow(feature_maps[:,:,i] ,cmap='gray')     # for grayscale use ,cmap='gray'
	else:
		ax.imshow(feature_maps[:,:,i])       
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')

#%%
# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pred)
target_names = ['class 0(airplane)', 'class 1(notAirplane)']
					
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix')
#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
plt.show()

#%%
# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

'''THIS'''
# =============================================================================
# # serialize model to JSON
# model_json = model.to_json()
# with open("C:/Users/Adrin/Desktop/keras_bin/modelWithOpenCV_gray.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("C:/Users/Adrin/Desktop/keras_bin/modelWithOpenCV_gray.h5")
# print("Saved model to disk")
# 
# # load json and create model
# json_file = open("C:/Users/Adrin/Desktop/keras_bin/modelWithOpenCV_gray.json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("C:/Users/Adrin/Desktop/keras_bin/modelWithOpenCV_gray.h5")
# print("Loaded model from disk")
# =============================================================================

'''::::::::::::::::::::::OR::::::::::::::::::::::::::::'''

'''THIS'''
'''model.save will save the weights along with architecture also'''
model.save("C:/Users/Adrin/Desktop/keras_bin/modelWithOpenCV_air_gray_v2_acc.hdf5")
loaded_model=load_model("C:/Users/Adrin/Desktop/keras_bin/modelWithOpenCV_air_gray_v2_acc.hdf5")

#%%
''' Testing new Image '''

from keras.preprocessing import image

def test_new_image(newimage, inf_fp):
    test_image = cv2.imread(newimage)
    #if using_grayscale_or_color_flag == "grayscale":
    test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)        #enable it for grayscale images
    test_image=cv2.resize(test_image,(256,256))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    print (test_image.shape)
   
    if num_channel==1:
        if K.image_dim_ordering()=='th':
            test_image= np.expand_dims(test_image, axis=0)
            test_image= np.expand_dims(test_image, axis=0)
            print (test_image.shape)
        else:
            test_image= np.expand_dims(test_image, axis=3) 
            test_image= np.expand_dims(test_image, axis=0)
            print (test_image.shape)
    else:
       	if K.image_dim_ordering()=='th':
               test_image=np.rollaxis(test_image,2,0)
               test_image= np.expand_dims(test_image, axis=0)
               print (test_image.shape)
        else:
            test_image= np.expand_dims(test_image, axis=0)
            print (test_image.shape)
       
        # Predicting the test image
    print((model.predict(test_image)))
    pred = model.predict_classes(test_image)
    print(pred)
    inf_fp.write(str(pred)+"\n")
    print("Predicted as : ",labelNames[int(pred)])

    
''' call test_new_image() fn to test any image '''
#test_new_image("C:/Users/Adrin/Desktop/keras_bin/newpredict_color/forPrediction.jpg") 

'''To test bunch of images, provide dir'''
dir_in_str = "C:/Users/Adrin/Desktop/keras_bin/airplaneTest"
dir = os.fsencode(dir_in_str)

inf_fp = open(dir_in_str+"/inference.txt","w+")

for file in os.listdir(dir):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        print("filename --> ",filename+"\n")
        inf_fp.write(filename+":")        
        test_new_image(dir_in_str+"/"+filename, inf_fp)
        continue
    else:
        continue

#%%
# =============================================================================
# for line in inf_fp:
# 	tilename = os.path.split(line)[1]
# 	r = tilename.split('_')[-2]
# 	c = tilename.split('_')[-1].split('.')[0]	
#     index = 
# 	labeltext=labelNames[index]        
# =============================================================================
    
    
# =============================================================================
# import pandas as pd
# predictions = model.predict_classes(X_test, verbose=0)
# predictions_df = pd.DataFrame (predictions,columns = ['Label'])
# predictions_df['ImageID'] = predictions_df.index + 1
# submission_df = predictions_df[predictions_df.columns[::-1]]
# submission_df.to_csv("C:/Users/Adrin/Desktop/keras_bin/submission.csv", index=False, header=True)
# submission_df.head()
# 
# =============================================================================
