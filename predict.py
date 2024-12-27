import torch 
import tqdm
import sys, os
import numpy as np
from PIL import Image
from ultralytics import YOLO
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(path: str):
    model = YOLO(path)
    model.fuse()
    # Ensure model using GPU or else it got error
    model.to('cuda')
    return model

# # got from https://medium.com/@telega.slawomir.ai/yolo8-basics-plotting-bboxes-e50a8f3e5227
# # def plot_bboxes(results):
#     img = results[0].orig_img # original image
#     names = results[0].names # class names dict
#     scores = results[0].boxes.conf.numpy() # probabilities
#     classes = results[0].boxes.cls.numpy() # predicted classes
#     boxes = results[0].boxes.xyxy.numpy().astype(np.int32) # bboxes
#     for score, cls, bbox in zip(scores, classes, boxes): # loop over all bboxes
#         class_label = names[cls] # class name
#         label = f"{class_label} : {score:0.2f}" # bbox label
#         lbl_margin = 3 #label margin
#         img = cv2.rectangle(img, (bbox[0], bbox[1]),
#                             (bbox[2], bbox[3]),
#                             color=(0, 0, 255),
#                             thickness=1)
#         label_size = cv2.getTextSize(label, # labelsize in pixels 
#                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
#                                      fontScale=1, thickness=1)
#         lbl_w, lbl_h = label_size[0] # label w and h
#         lbl_w += 2* lbl_margin # add margins on both sides
#         lbl_h += 2*lbl_margin
#         img = cv2.rectangle(img, (bbox[0], bbox[1]), # plot label background
#                              (bbox[0]+lbl_w, bbox[1]-lbl_h),
#                              color=(0, 0, 255), 
#                              thickness=-1) # thickness=-1 means filled rectangle
#         cv2.putText(img, label, (bbox[0]+ lbl_margin, bbox[1]-lbl_margin), # write label to the image
#                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=1.0, color=(255, 255, 255 ),
#                     thickness=1)
#     return img
# def plot_bboxes(results):
#     img = results[0].orig_img # original image
#     names = results[0].names # class names dict
    
#     # Chuyển tensor từ GPU về CPU trước khi chuyển sang numpy
#     scores = results[0].boxes.conf.cpu().numpy() # probabilities
#     classes = results[0].boxes.cls.cpu().numpy() # predicted classes
#     boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32) # bboxes
    
#     for score, cls, bbox in zip(scores, classes, boxes): # loop over all bboxes
#         class_label = names[cls] # class name
#         label = f"{class_label} : {score:0.2f}" # bbox label
#         lbl_margin = 3 #label margin
#         img = cv2.rectangle(img, (bbox[0], bbox[1]),
#                             (bbox[2], bbox[3]),
#                             color=(0, 0, 255),
#                             thickness=1)
#         label_size = cv2.getTextSize(label, # labelsize in pixels 
#                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
#                                      fontScale=1, thickness=1)
#         lbl_w, lbl_h = label_size[0] # label w and h
#         lbl_w += 2* lbl_margin # add margins on both sides
#         lbl_h += 2*lbl_margin
#         img = cv2.rectangle(img, (bbox[0], bbox[1]), # plot label background
#                              (bbox[0]+lbl_w, bbox[1]-lbl_h),
#                              color=(0, 0, 255), 
#                              thickness=-1) # thickness=-1 means filled rectangle
#         cv2.putText(img, label, (bbox[0]+ lbl_margin, bbox[1]-lbl_margin), # write label to the image
#                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=1.0, color=(255, 255, 255 ),
#                     thickness=1)
#     return img

# # def reference(image: np.ndarray,model):
# #     # Still we need to check the image
# #     # If image in RGB format
# #     if len(image.shape) == 3:
# #         # Convert to gray scale 
# #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     # if image already grayscale
# #     if len(image.shape) < 3:
# #         result = model(image)
# #         temp = plot_bboxes(result)
# #         # Display 1 image first
# #         cv2.imshow('Image with Bouding Box', temp) # show annotated image
# #         cv2.waitKey(0) # wait for a keypressed
# #         cv2.destroyAllWindows() # clear windows
# #         # Save image
# #         temp.save("out.png")

# def reference(image: np.ndarray, model):

#     # Thực hiện dự đoán trên ảnh
#     result = model.predict(image, conf = 0.25)
#     print(result.boxes.xyxy)
#     # Vẽ bounding boxes trên ảnh gốc
#     annotated_image = plot_bboxes(result)
    
#     # # Chuyển đổi từ PIL Image sang định dạng OpenCV nếu cần
#     # if isinstance(annotated_image, Image.Image):
#     #     annotated_image = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
    
#     # Hiển thị ảnh
#     cv2.imshow('Image with Bounding Box', annotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     # Lưu ảnh
#     if isinstance(annotated_image, np.ndarray):
#         cv2.imwrite("out.png", annotated_image)
#     else:
#         annotated_image.save("out.png")
        
# def predict():
#     # get image path from command line 
#     # image_link = str(sys.argv[1])
#     # model_link = str(sys.argv[0])
#     model = load_model('C:/Users/THIS PC/Downloads/CAD/runs/detect/train4/weights/best.pt')
#     image = cv2.imread("C:/Users/THIS PC/Downloads/CAD/165018059_407148354_4.jpg")
#     reference(image=image,model=model)

#Define main function Create streamlit app
def main():
    os.system("cls")
    text = []
    # model = YOLO("yolo11n.yaml").load("C:/Users/THIS PC/Downloads/CAD/runs/detect/train/weights/best.pt")
    model = YOLO("C:/Users/THIS PC/Downloads/CAD/runs/detect/train/weights/best.pt")
    # results = model.predict(source = "./2581421047_3385717602_6.jpg", show = True, save = True)
    # for result in results:
    #     if result.boxes.xyxy.shape[0] == 0:
    #         if result.boxes.id is None:
    #             print("True")
    #     else:
    #         print("Fasle")
    directory = os.fsencode("YOLO/train/images")
    print('Data load initiated')
    for subdir, dirs, files in os.walk(directory):
        # files = [f for f in files if f.startswith('Test')]
        for file in tqdm.tqdm(files):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Predit the image
                results = model("YOLO/train/images/" +filename)
                for r in results:
                    if r.boxes.xyxy.shape[0] != 0 and r.boxes.id is not None:
                        # print(filename)
                        text.append(filename)
                    else:
                        pass
        # Save text file
        with open("save1.txt","a") as file:
            for line in text:
                file.write(f"{line}\n")
if __name__ == "__main__":
    main()