# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 18:33:56 2019

@author: Prakhar
"""

from modelh import HandleModel as hm

def main():
    path= input("enter a Image :")
    model=hm("digit.h5")
    print("output is :", model.predict(model.preprocess(path)))

if __name__ =="__main__":
    main()
