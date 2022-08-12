# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 18:06:16 2022

@author: lenovo
"""

import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle
import time
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image


def get_label_name(flag):
    if flag == 0:
        return 'cat'
    else:
        return 'dog'



image_directory = "C:/Users/lenovo/Desktop/Project M.Tech/Machine Learning/Neural networks and Deep Learning/Cats Vs Dogs/train/train"
categories=['cat','dog']
size=100
data=[]
for category in categories:
    folder=os.path.join(image_directory,category)
    label=categories.index(category)
    for img in os.listdir(folder):
        img_path=os.path.join(folder,img)
        img_arr=cv2.imread(img_path)
        img_arr=cv2.resize(img_arr,(size,size))
        data.append([img_arr,label])
random.shuffle(data)
X=[]
Y=[]
for features, labels in data:
    X.append(features)
    Y.append(labels)        
X=np.array(X)
Y=np.array(Y)    
pickle.dump(X, open('X.pk1','wb'))
pickle.dump(Y, open('Y.pk1','wb'))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import TensorBoard

NAME=f'cat-vs-dog-prediction-{int(time.time())}'
tensorboard=TensorBoard(log_dir=f'logs\\{NAME}\\')

X=pickle.load(open('X.pk1','rb'))
Y=pickle.load(open('Y.pk1','rb'))

X = X/255   
Y = Y/255




model=Sequential()
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128, input_shape= X.shape[1:], activation='relu'))    
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history=model.fit(X,Y, epochs=5, validation_split=0.1, batch_size=32,callbacks=[tensorboard]) 
  
# evaluate model
_, acc = model.evaluate(X, Y)
print("Accuracy = ", (acc * 100.0), "%")


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
#acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


model.save('CatVsDog.h5')
# Load a model
loaded_model = tf.keras.models.load_model('CatVsDog.h5')
loaded_model.layers[0].input_shape


#Testing
batch_holder = np.zeros((50, size, size, 3))
img_dir='C:/Users/lenovo/Desktop/Project M.Tech/Machine Learning/Neural networks and Deep Learning/Cats Vs Dogs/test/test2'
for i,img in enumerate(os.listdir(img_dir)):
  img = image.load_img(os.path.join(img_dir,img), target_size=(size, size))
  batch_holder[i, :] = img
  
result=loaded_model.predict(batch_holder)

fig = plt.figure(figsize=(20, 20))

for i,img in enumerate(batch_holder):
  fig.add_subplot(4,5, i+1)
  plt.title(untitled3.get_label_name(int(result[i][0])))
  plt.imshow(img/256.)
  
plt.show()


'''
image_path="C:/Users/lenovo/Desktop/Project M.Tech/Machine Learning/Neural networks and Deep Learning/Cats Vs Dogs/test/test/52.jpg"
img = image.load_img(image_path, target_size=(size, size))

img = np.expand_dims(img, axis=0)



result=np.argmax(model.predict(img), axis=-1)
result = np.expand_dims(result, 1)
import untitled3
plt.imshow(np.squeeze(img))
plt.title(untitled3.get_label_name(int(result[0][0])))
plt.show()
'''    
        
    
    
    
    
    
    
    
    
    
    