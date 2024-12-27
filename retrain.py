#Import needed libraries
import os
from ultralytics import YOLO
import cv2
import numpy as np 
import pandas as pd

def retrain():
    model = YOLO('yolo11n.yaml').load('C:/Users/THIS PC/Downloads/CAD/runs/detect/train5/weights/best.pt') # build from YAML and transfer weights
    model.train(data = 'C:/Users/THIS PC/Downloads/CAD/YOLO/data.yaml',
                          batch = 0.5, # let the model eat all the RAM
                          epochs = 4, 
                          optimizer = "auto", 
                          save = True, 
                          seed = 21522592, 
                          # cache = True, 
                          dropout = 0.2,
                          plots = True)
    results = model('C:/Users/THIS PC/Downloads/CAD/165018059_407148354_4.jpg',save = True, show = True)
    print(results)

#Define main function Create streamlit app
def main():
    os.system("cls")
    retrain()
if __name__ == "__main__":
    main()