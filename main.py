#Import needed libraries
import os
from ultralytics import YOLO
import cv2
import torch
import numpy as np 
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

def plot_bboxes(results):
    img = results[0].orig_img # original image
    names = results[0].names # class names dict
    scores = results[0].boxes.conf.numpy() # probabilities
    classes = results[0].boxes.cls.numpy() # predicted classes
    boxes = results[0].boxes.xyxy.numpy().astype(np.int32) # bboxes
    for score, cls, bbox in zip(scores, classes, boxes): # loop over all bboxes
        class_label = names[cls] # class name
        label = f"{class_label} : {score:0.2f}" # bbox label
        lbl_margin = 3 #label margin
        img = cv2.rectangle(img, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            color=(0, 0, 255),
                            thickness=1)
        label_size = cv2.getTextSize(label, # labelsize in pixels 
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                     fontScale=1, thickness=1)
        lbl_w, lbl_h = label_size[0] # label w and h
        lbl_w += 2* lbl_margin # add margins on both sides
        lbl_h += 2*lbl_margin
        img = cv2.rectangle(img, (bbox[0], bbox[1]), # plot label background
                             (bbox[0]+lbl_w, bbox[1]-lbl_h),
                             color=(0, 0, 255), 
                             thickness=-1) # thickness=-1 means filled rectangle
        cv2.putText(img, label, (bbox[0]+ lbl_margin, bbox[1]-lbl_margin), # write label to the image
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(255, 255, 255 ),
                    thickness=1)
    return img

#Training model
def train():
    #dcm file to png 
    # Load a COCO-pretrained YOLOv8n model
    model = YOLO("yolo11n.yaml")
    # Display model information (optional)
    model.info()
    results = model.train(data = 'C:/Users/THIS PC/Downloads/CAD/YOLO/data.yaml',
                          batch = 0.8, # let the model eat all the RAM
                          epochs = 10, 
                          optimizer = "auto", 
                          save = True, 
                          seed = 21522592, 
                          cache = True, 
                          dropout = 0.2,
                          plots = True)
    #test results
    results = model('C:/Users/THIS PC/Downloads/CAD/165018059_407148354_4.jpg',save = True, show = True)
    img = plot_bboxes(results)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Define main function Create streamlit app
def main():
    os.system("cls")
    train()
if __name__ == "__main__":
    main()