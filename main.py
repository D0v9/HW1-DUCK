import numpy as np
import pandas as pd
import cv2
from PIL import Image
from datetime import datetime as dt
import math
import matplotlib.pyplot as plt




def summarize(dataset): 
    summaries = [(np.mean(attribute), np.std(attribute)) for attribute in
                 zip(*dataset[1:-2])]  
    del summaries[-1]
    return summaries

###將label=1or2進行分類
def summarizeByClass(dataset):  
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):  
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

###高斯分布
def calculateProbability(x, mean, std):  
    exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2)))) 
    return (1 / (math.sqrt(2 * math.pi) * std)) * exp

###計算可
def calculateClassProbabilities(summaries, inputVector): 
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, std = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, std)
    return probabilities

###預測是label=1還是=2時機率比較高，
def predict(summaries, inputVector): 
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel




def main():
    ###輸入train資料
    training_data=pd.read_csv("pixel.csv")
    training_data=training_data.values.tolist()

    ###輸入測試資料，並將他轉成RGB
    test_img=cv2.imread("full_duck.jpg")#bgr

    height,weight,channels=test_img.shape
    test_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)#bgr to rgb
    test_img_array=np.reshape(test_img, (height*weight, 3))


    test_img_array=pd.DataFrame(test_img_array)
    test_data=test_img_array.values.tolist()

    ############################################################ 
    summaries = summarizeByClass(training_data)  

    predictions = []
    for i in range(len(test_data)):
        result = predict(summaries, test_data[i])
        predictions.append(result)
    
    ###如果分類為鴨子，則填上白點，若不是則是黑點
    Images =[]  
    for i in range(len(predictions)):
        if(1 == int(predictions[i])):
            Images.append([255,255,255])
        else:
            Images.append(([0,0,0]))
    Images = np.array(Images)
    size = test_img.shape
    array = np.reshape(Images, (size[0], -1))

    ###輸出並儲存圖片
    output_img = Image.fromarray(array)
    output_img = output_img.resize((size[0], size[1]))
    output_img.convert('RGB').save("D://T//IP//HW1-DUCK//result.jpg")
    
    # output_img.save("D://T//IP//HW1-DUCK//result.png")
    print("Successful create new images")
   
 
    
main()
