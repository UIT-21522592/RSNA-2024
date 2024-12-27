import streamlit as st
import pandas as pd
import SimpleITK as sitk
import pydicom
import tempfile
import matplotlib.pyplot as plt
import cv2
import numpy as np
from io import StringIO
from PIL import Image
from IPython.display import Image as IPyImage, display
#Set page config
st.set_page_config(page_title="RSNA Spine condition")
# Load the custom CSS file into the Streamlit app
def plot_spine_predictions(image_path, model_path, 
                         conf_threshold=0.25, iou_threshold=0.45, img_size=384):
    """
    Plot YOLO predictions for spine levels on a DICOM image
    
    Args:
        image_path: Path to DICOM image
        model_path: Path to saved YOLO model
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IOU threshold for NMS
        img_size: Image size for model input
    """
    # Load model
    model = YOLO(model_path)
    
    # Read DICOM
    ds = pydicom.dcmread(image_path)
    image = ds.pixel_array
    
    # Normalize and resize
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    image_resized = cv2.resize(image_normalized, (img_size, img_size))
    
    # Convert grayscale to RGB
    image_rgb = np.stack([image_resized] * 3, axis=-1)
    
    # Create figure
    fig = plt.figure(figsize=(15, 7))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_resized, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    # st.pyplot(fig)

    
    # Plot image with predictions
    plt.subplot(1, 2, 2)
    plt.imshow(image_resized, cmap='gray')
    plt.title('Predictions')
    plt.axis('off')
    # st.pyplot(fig)

    # Get predictions
    results = model.predict(
        source=image_rgb,
        conf=conf_threshold,
        iou=iou_threshold
    )
    
    # Define colors for each level
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    level_names = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.cpu().numpy()
        
        # Sort boxes by y-coordinate to display levels in order
        
        box_data = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            x1, y1, x2, y2 = box.xyxy[0]
            box_data.append((y1, cls_id, conf, x1, y1, x2, y2))

        box_data.sort()  # Sort by y1 coordinate
        
        # Plot each detection
        for i, (_, cls_id, conf, x1, y1, x2, y2) in enumerate(box_data):
            color = colors[cls_id]
            
            level_name = level_names[cls_id]
            
            # Draw bounding box
            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, color=color, linewidth=2
            ))
            
            # Add label
            plt.text(
                x2 + 5, (y1 + y2) / 2, 
                f'{level_name}: {conf:.2f}',
                color=color, fontsize=8, verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
            
            # Print detection info
            print(f"Found {level_name} with confidence {conf:.2f}")
    else:
        print("No detections found")
    
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

# def plot_spine_predictions_condition(image_path, model_path_1, model_path_2, model_path_3, conf_threshold=0.25, iou_threshold=0.45, img_size=384, axial = 0):
#     """
#     Plot YOLO predictions for spine levels on a DICOM image
    
#     Args:
#         image_path: Path to DICOM image
#         model_path: Path to saved YOLO model
#         conf_threshold: Confidence threshold for predictions
#         iou_threshold: IOU threshold for NMS
#         img_size: Image size for model input
#     """
#     # Load model
#     model_level = YOLO(model_path_1)
#     model_condition = YOLO(model_path_2)
#     model_target = YOLO(model_path_3)
#     # Read DICOM
#     ds = pydicom.dcmread(image_path)
#     image = ds.pixel_array
    
#     # Normalize and resize
#     image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#     image_resized = cv2.resize(image_normalized, (img_size, img_size))
    
#     # Convert grayscale to RGB
#     image_rgb = np.stack([image_resized] * 3, axis=-1)
    
#     # Create figure
#     fig = plt.figure(figsize=(15, 7))
    
#     # Plot original image
#     plt.subplot(1, 2, 1)
#     plt.imshow(image_resized, cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')
#     # st.pyplot(fig)

    
#     # Plot image with predictions
#     plt.subplot(1, 2, 2)
#     plt.imshow(image_resized, cmap='gray')
#     plt.title('Predictions')
#     plt.axis('off')
#     # st.pyplot(fig)

#     # Get predictions
#     results_level = model_level.predict(
#         source=image_rgb,
#         conf=conf_threshold,
#         iou=iou_threshold
#     )
    
#     results_condition = model_condition.predict(
#         source=image_rgb,
#         conf=conf_threshold,
#         iou=iou_threshold
#     )
#     results_target = model_target.predict(
#         source=image_rgb,
#         conf=conf_threshold,
#         iou=iou_threshold
#     )
#     # Define colors for each level
#     colors = ['red', 'green', 'blue', 'yellow', 'purple']
#     if axial == 0:
#         level_names = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1'] # For sagittal
#     else:
#         level_names = ['left', 'right'] # For axial
#     conditions_name = ['Spinal Canal Stenosis', 'Right Neural Foraminal Narrowing', 'Left Neural Foraminal Narrowing', 'Left Subarticular Stenosis', 'Right Subarticular Stenosis']
#     target = ['Normal/Mild','Moderate','Severe']

#     if results_level[0].boxes is not None and results_condition[0].boxes is not None and results_target[0].boxes is not None:
#         boxes = results_level[0].boxes.cpu().numpy()
#         condition_boxes = results_condition[0].boxes.cpu().numpy()
#         target_boxes = results_condition[0].boxes.cpu().numpy()

#         # Sort boxes by y-coordinate to display levels in order
        
#         box_data = []
#         for box in boxes:
#             cls_id = int(box.cls[0])
#             conf = box.conf[0]
#             x1, y1, x2, y2 = box.xyxy[0]
#             box_data.append((y1, cls_id, conf, x1, y1, x2, y2))

#         box_data.sort()  # Sort by y1 coordinate

#         box_condition_data = []
#         for box in condition_boxes:
#             cls_id = int(box.cls[0])
#             conf = box.conf[0]
#             x1, y1, x2, y2 = box.xyxy[0]
#             box_data.append((y1, cls_id, conf, x1, y1, x2, y2))

#         box_condition_data.sort()  # Sort by y1 coordinate

#         box_target_data = []
#         for box in target_boxes:
#             cls_id = int(box.cls[0])
#             conf = box.conf[0]
#             x1, y1, x2, y2 = box.xyxy[0]
#             box_data.append((y1, cls_id, conf, x1, y1, x2, y2))

#         box_target_data.sort()  # Sort by y1 coordinate
        
#         # Plot each level detection
#         for i, (_, cls_id, conf, x1, y1, x2, y2) in enumerate(box_data):
#             color = colors[cls_id]
            
#             level_name = level_names[cls_id]
            
#             # Draw bounding box
#             plt.gca().add_patch(plt.Rectangle(
#                 (x1, y1), x2-x1, y2-y1,
#                 fill=False, color=color, linewidth=2
#             ))
            
#             # Add label
#             plt.text(
#                 x2 + 5, (y1 + y2) / 2, 
#                 f'{level_name}: {conf:.2f}',
#                 color=color, fontsize=8, verticalalignment='center',
#                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
#             )
            
#             # Print detection info
#             print(f"Found {level_name} with confidence {conf:.2f}")
#         # Plot each condition detection
#         for i, (_, cls_id, conf, x1, y1, x2, y2) in enumerate(box_condition_data):
#             color = colors[cls_id]
            
#             conditions_name = conditions_name[cls_id]
            
#             # Draw bounding box
#             plt.gca().add_patch(plt.Rectangle(
#                 (x1, y1), x2-x1, y2-y1,
#                 fill=False, color=color, linewidth=2
#             ))
            
#             # Add label
#             plt.text(
#                 x2 + 5, (y1 + y2) / 2, 
#                 f'{conditions_name}: {conf:.2f}',
#                 color=color, fontsize=8, verticalalignment='center',
#                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
#             )
            
#             # Print detection info
#             print(f"Found {conditions_name} with confidence {conf:.2f}")
#         # Plot each target detection
#         for i, (_, cls_id, conf, x1, y1, x2, y2) in enumerate(box_target_data):
#             color = colors[cls_id]
            
#             target = target[cls_id]
            
#             # Draw bounding box
#             plt.gca().add_patch(plt.Rectangle(
#                 (x1, y1), x2-x1, y2-y1,
#                 fill=False, color=color, linewidth=2
#             ))
            
#             # Add label
#             plt.text(
#                 x2 + 5, (y1 + y2) / 2, 
#                 f'{target}: {conf:.2f}',
#                 color=color, fontsize=8, verticalalignment='center',
#                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
#             )
            
#             # Print detection info
#             print(f"Found {target} with confidence {conf:.2f}")
#     else:
#         print("No detections found")
    
#     plt.axis('off')
#     plt.tight_layout()
#     st.pyplot(fig)

def plot_spine_predictions_condition(image_path, model_path_1, model_path_2, model_path_3, 
                                  conf_threshold=0.25, iou_threshold=0.45, img_size=384, axial=0):
    # Load models
    model_level = YOLO(model_path_1)
    model_condition = YOLO(model_path_2) 
    model_target = YOLO(model_path_3)

    # Read and process image
    ds = pydicom.dcmread(image_path)
    image = ds.pixel_array
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    image_resized = cv2.resize(image_normalized, (img_size, img_size))
    image_rgb = np.stack([image_resized] * 3, axis=-1)

    # Create figure
    fig = plt.figure(figsize=(15, 7))
    
    # Plot original and prediction images
    plt.subplot(1, 2, 1)
    plt.imshow(image_resized, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_resized, cmap='gray')
    plt.title('Predictions')
    plt.axis('off')

    # Get predictions
    results_level = model_level.predict(source=image_rgb, conf=conf_threshold, iou=iou_threshold)
    results_condition = model_condition.predict(source=image_rgb, conf=conf_threshold, iou=iou_threshold)
    results_target = model_target.predict(source=image_rgb, conf=conf_threshold, iou=iou_threshold)

    # Define colors and names
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    level_names = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1'] if axial == 0 else ['left', 'right']
    condition_names = ['Spinal Canal Stenosis', 'Right Neural Foraminal Narrowing', 
                      'Left Neural Foraminal Narrowing', 'Left Subarticular Stenosis', 
                      'Right Subarticular Stenosis']
    target_names = ['Normal/Mild', 'Moderate', 'Severe']

    # Plot predictions
    offset = 0  # Vertical offset for text placement
    
    # Plot level predictions
    if results_level[0].boxes is not None:
        boxes = results_level[0].boxes.cpu().numpy()
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            x1, y1, x2, y2 = box.xyxy[0]
            
            # Draw box
            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, color=colors[cls_id], linewidth=2
            ))
            
            # Add label
            plt.text(
                x2 + 5, (y1 + y2) / 2,
                f'Level {level_names[cls_id]}: {conf:.2f}',
                color=colors[cls_id], fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
            offset += 20
    
    # Plot condition predictions
    if results_condition[0].boxes is not None:
        boxes = results_condition[0].boxes.cpu().numpy()
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            x1, y1, x2, y2 = box.xyxy[0]
            
            # Draw box with dashed line
            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, color=colors[cls_id], linewidth=2, linestyle='--'
            ))
            
            # Add label
            plt.text(
                x2 + 5, (y1 + y2) / 2 + 10,
                f'Condition {condition_names[cls_id]}: {conf:.2f}',
                color=colors[cls_id], fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
            offset += 20

    # Plot target predictions
    if results_target[0].boxes is not None:
        boxes = results_target[0].boxes.cpu().numpy()
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            x1, y1, x2, y2 = box.xyxy[0]
            
            # Draw box with dotted line
            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, color=colors[cls_id], linewidth=2, linestyle=':'
            ))
            
            # Add label
            plt.text(
                x2 + 5, (y1 + y2) / 2,
                f'Target {target_names[cls_id]}: {conf:.2f}',
                color=colors[cls_id], fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
            offset += 20

    plt.tight_layout()
    st.pyplot(fig)


def load_css():
    with open("./style/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Apply the CSS file
load_css()
# Thematic Header
html_templ = """
    <div style="background-color:#2D3E50; padding:20px; text-align:center; border-radius:10px;">
        <h1 style="color:white; font-family: 'Arial', sans-serif; font-size: 36px;">
            <span style="font-size: 50px; vertical-align: middle;">&#9748;</span> 
            RSNA Spine Prediction <span style="color:#ffeb3b;">&#9748;</span>
        </h1>
        <p style="color:white; font-size: 18px; font-family: 'Arial', sans-serif; font-weight: 300;">
            A comprehensive web application for predicting spine condition and plotting.
        </p>
    </div>
    """
st.markdown(html_templ, unsafe_allow_html=True)
# Sidebar context with improvements for design
st.sidebar.title("Spine condition Prediction Sidebar")
st.sidebar.header("Choose the content")
content_choice = st.sidebar.selectbox("Choose the state", ("Gallery", "Spine Prediction"))
# Read data
train_df = pd.read_csv("./train.csv")

from function import visualize_3d_plotly
from ultralytics import YOLO

# Import the model
model_axial = "./YOLOv8model/axial_t2_spine_detector.pt"
model_sagittal_T1 = "./YOLOv8model/sagittal_t1_spine_detector.pt"
model_sagittal_T2 = "./YOLOv8model/sagittal_t2_spine_detector.pt"
model_axial_condition = "./YOLOv8model/axial_t2_spine_condition_detector.pt"
model_sagittal_T1_condition = "./YOLOv8model/sagittal_t1_spine_condition_detector.pt"
model_sagittal_T2_condition = "./YOLOv8model/sagittal_t2_spine_condition_detector.pt"
model_sagittal_T1_target = "./YOLOv8model/sagittal_t1_spine_target_detector.pt"
model_sagittal_T2_target = "./YOLOv8model/sagittal_t2_spine_target_detector.pt"
model_axial_target = "./YOLOv8model/axial_t2_spine_target_detector.pt"
# Read data
train_df = pd.read_csv("./train.csv")
# Add custom CSS for font size styling for selectbox components
st.markdown("""
    <style>
    .stSelectbox>div>div>div>label {
        font-size: 18px !important; /* Font size for selectbox label */
    }
    .stSelectbox>div>div>div>div>div>div {
        font-size: 16px !important; /* Font size for selectbox options */
    }
    .stButton>button {
        font-size: 16px !important; /* Button font size */
    }
    .stMarkdown {
        font-size: 16px !important; /* Markdown text font size */
    }
    </style>
    """, unsafe_allow_html=True)
if content_choice == "Gallery":
    st.subheader("Gallery")
    study_ids = train_df.study_id.unique()
    study_id = st.selectbox('Select a study id to view', options=study_ids )
    left, middle = st.columns(2)
    if left.button("OK", use_container_width=True):
        left.markdown("You clicked the plain button.")
        # Select study_id


        # study_id = study_ids[0]
        sample_df = train_df[train_df['study_id']==study_id]

        sagittal_files = sample_df['image_path'][sample_df['series_description']!='Axial T2'].unique() 
        axial_files = sample_df['image_path'][sample_df['series_description']=='Axial T2'].unique() 

        sagittal_conditions = []
        for sagittal_file in sagittal_files:
            conditions = sample_df[sample_df['image_path'] == sagittal_file].condition_lr.unique()
            assert len(conditions)==1
            sagittal_conditions.append(conditions[0])

        axial_conditions = []
        for axial_file in axial_files:
            conditions = sample_df[sample_df['image_path'] == axial_file].condition_lr.unique()
            assert len(conditions)==1
            axial_conditions.append(conditions[0])

        TRAIN_PATH = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images'
        visualize_3d_plotly(sagittal_files, axial_files, sagittal_conditions, axial_conditions, sample_df, use_surfacecolor=True, scale_factor=4)
    if middle.button("CANCEL", icon="😃", use_container_width=True):
        middle.markdown("You clicked the emoji button.")
if content_choice == "Spine Prediction":  
    uploaded_files =  st.sidebar.file_uploader("Upload an Spine MRI Image (dicom)",type=['dcm'])
    if uploaded_files is not None:
        # plot_spine_predictions(uploaded_files,"./YOLOv8model/sagittal_t2_spine_detector.pt")
        plot_spine_predictions_condition(uploaded_files, model_sagittal_T1, model_sagittal_T1_condition, model_sagittal_T1_target, axial = 0)