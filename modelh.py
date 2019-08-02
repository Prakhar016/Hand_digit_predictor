# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 18:35:51 2019

@author: Prakhar
"""
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
class HandleModel(object):
    def __init__(self,arg):
        self.model_path=arg
        self.__model=load_model(self.model_path)
    def preprocess(self,path):
        self._img=Image.open(path).convert("L")
        self._img_array=np.asarray(self._img)
        self._res=cv2.resize(self._img_array,(28,28))
        self._res=self._res/255
        self._res=self._res.reshape(1,28,28,1)
        return self._res
    def predict(self,imgrray):
        return self.__model.predict_classes(imgrray)[0]
