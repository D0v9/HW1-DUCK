# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:04:32 2022

@author: inkzs
"""
import numpy as np
import pandas as pd
import cv2

#load img
img=cv2.imread("background6.jpg")
#出來的結果是BGR
height,weight,channels=img.shape
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#bgr to rgb
img_array=np.reshape(img, (height*weight, 3))


img_array=pd.DataFrame(img_array)
img_array.to_csv("RGBfeature.csv")#將rgb向量輸出成csv