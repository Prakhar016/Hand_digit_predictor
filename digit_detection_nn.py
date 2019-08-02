# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import random
import matplotlib.image as mpimg

np.random.seed(0)

data=mnist.load_data()

len(data[0])

(x_train,y_train),(x_test,y_test)=data

len(x_train)

len(y_test)

plt.imshow(x_train[0],cmap="gray")
y_train[0]

print(x_train.shape)
print(x_test.shape)

plt.imshow(x_train[0])

num_of_sample=[]
col=5
num_Classes=10

fig,axs=plt.subplots(nrows=num_Classes,ncols=col,figsize=(5,10))
fig.tight_layout()
for i in range(col):
  for j in range(num_Classes):
    x_selected=x_train[y_train==j]
    axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected)-1)),:,:],cmap=plt.get_cmap("gray"))
    axs[j][i].axis("off")
    if i==2:
      axs[j][i].set_title(str(j))
      num_of_sample.append(len(x_selected))

print(num_of_sample)
plt.figure(figsize=(12,4))
plt.bar(range(0,num_Classes),num_of_sample)
plt.title("Distribution of data")
plt.xlabel("Class number")
plt.ylabel("number of images")
plt.show()

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

y_train=to_categorical(y_train,10)
#y_test=to_categorical(y_test,10)

y_train[0]

x_train=x_train/255
x_test=x_test/255

y_train.shape

def le_net():
  model=Sequential()
  model.add(Conv2D(30,(5,5),input_shape=(28,28,1),activation="relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(15,(3,3),activation="relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten())
  model.add(Dense(500,activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dropout(0.5))
  model.add(Dense(num_Classes,activation="softmax"))
  model.compile(Adam(lr=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
  return model

lenet=le_net()
lenet.summary()

history=lenet.fit(x_train,y_train,epochs=8,validation_split=0.1,batch_size=400,verbose=1,shuffle=1)

lenet.save("digit.h5")

!ls

from google.colab import files

files.download("digit.h5")

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.title("Loss")
plt.xlabel("epoch")

plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.legend(["acc","val_acc"])
plt.title("acc")
plt.xlabel("epoch")

import requests as r
from PIL import Image
response=r.get("https://kx.com/images/03_IMAGES/160520-8.png",stream=True)
img=Image.open(response.raw).convert("L")
plt.imshow(img,cmap="gray")

img.size

import cv2
img_array=np.asarray(img)
res=cv2.resize(img_array,(28,28))
plt.imshow(res,cmap="gray")

res=res/255
res=res.reshape(1,28,28,1)

lenet.predict(res)

lenet.predict_classes(res)[0]



