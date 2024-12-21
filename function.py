import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
import pydicom
import nbformat
import matplotlib as plt
from ultralytics import YOLO
import cv2
from scipy.interpolate import griddata
TRAIN_PATH = "./train_images"
color_map = {
    'Normal/Mild': 'green',
    'Moderate': 'yellow',
    'Severe': 'red'
}
colormap = {'Spinal Canal Stenosis':'red',
            'Neural Foraminal Narrowing': 'green',
            'Subarticular Stenosis': 'blue'}
def load_dicom_image(dicom_file_path):
    dicom_image = pydicom.dcmread(f'{TRAIN_PATH}/{dicom_file_path}')
    pixel_array = dicom_image.pixel_array
    position = np.array(dicom_image.ImagePositionPatient)
    orientation = np.array(dicom_image.ImageOrientationPatient)
    spacing = np.array(dicom_image.PixelSpacing)
    slice_thickness = dicom_image.SliceThickness if 'SliceThickness' in dicom_image else 1.0
    return pixel_array, position, orientation, spacing, slice_thickness

def load_and_resize_image(dicom_file_path, scale_factor=0.25):
    dicom_image = pydicom.dcmread(f'{TRAIN_PATH}/{dicom_file_path}')
    pixel_array = dicom_image.pixel_array
    resized_array = cv2.resize(pixel_array, (0, 0), fx=scale_factor, fy=scale_factor)
    position = np.array(dicom_image.ImagePositionPatient)
    orientation = np.array(dicom_image.ImageOrientationPatient)
    spacing = np.array(dicom_image.PixelSpacing) * (1 / scale_factor)
    slice_thickness = dicom_image.SliceThickness if 'SliceThickness' in dicom_image else 1.0
    return resized_array, position, orientation, spacing, slice_thickness

# Calculate coordinates
def get_corners(position, orientation, spacing, pixel_array_shape):
    rows, cols = pixel_array_shape
    row_dir = orientation[:3]
    col_dir = orientation[3:]

    top_left = position
    top_right = position + col_dir * (cols - 1) * spacing[1]
    bottom_left = position + row_dir * (rows - 1) * spacing[0]
    bottom_right = position + row_dir * (rows - 1) * spacing[0] + col_dir * (cols - 1) * spacing[1]

    return top_left, top_right, bottom_left, bottom_right


def create_surface(x_coords, y_coords, z_coords, image):
    # Create a grid for interpolation
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, image.shape[1]), np.linspace(0, 1, image.shape[0]))

   # Interpolate x, y, and z values
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    
    grid_x_coords = griddata(points, np.array(x_coords), (grid_x, grid_y), method='linear')
    grid_y_coords = griddata(points, np.array(y_coords), (grid_x, grid_y), method='linear')
    grid_z_coords = griddata(points, np.array(z_coords), (grid_x, grid_y), method='linear')

    return grid_x_coords, grid_y_coords, grid_z_coords

def pixel_to_physical(x_pixel, y_pixel, position, orientation, spacing):
    x_physical = position[0] + x_pixel * orientation[0] * spacing[0] + y_pixel * orientation[3] * spacing[1]
    y_physical = position[1] + x_pixel * orientation[1] * spacing[0] + y_pixel * orientation[4] * spacing[1]
    z_physical = position[2] + x_pixel * orientation[2] * spacing[0] + y_pixel * orientation[5] * spacing[1]
    return x_physical, y_physical, z_physical

def visualize_3d_plotly(sagittal_files, axial_files, sagittal_conditions, axial_conditions, label_data, use_surfacecolor=False, scale_factor=4):
    fig = go.Figure()

    # Load and plot the sagittal images
    for sagittal_file, condition in zip(sagittal_files,sagittal_conditions):
        #sagittal_image, sagittal_position, sagittal_orientation, sagittal_spacing, _ = load_dicom_image(sagittal_file) 
        sagittal_image, sagittal_position, sagittal_orientation, sagittal_spacing, _ = load_and_resize_image(sagittal_file,scale_factor=1/scale_factor) 
        corners = get_corners(sagittal_position, sagittal_orientation, sagittal_spacing, sagittal_image.shape)
        
        x_coords = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]]
        y_coords = [corners[0][1], corners[1][1], corners[2][1], corners[3][1]]
        z_coords = [corners[0][2], corners[1][2], corners[2][2], corners[3][2]]
        
        x_grid, y_grid, z_grid = create_surface(x_coords, y_coords, z_coords, sagittal_image)
        
        # Rotate image
        sagittal_image = np.flipud(np.rot90(sagittal_image))

        color = colormap[condition]
        
        if use_surfacecolor:
            fig.add_trace(go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                surfacecolor=sagittal_image,
                colorscale='Gray',
                showscale=False,
                name='Sagittal Slice'
            ))
        else:
            fig.add_trace(go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale=[[0, color], [1,color]],
                showscale=False,
                name='Sagittal Slice'
            ))
            
        # Add cross marks
        image_labels = label_data[label_data['image_path'] == sagittal_file]
        for _, row in image_labels.iterrows():
            cross_color = 'green' if row['label'] == 0 else 'red'
            x_pixel = row['x']//scale_factor
            y_pixel = row['y']//scale_factor
            condition = row['target']
            color = color_map[condition]
            x_physical, y_physical, z_physical = pixel_to_physical(x_pixel, y_pixel, sagittal_position, sagittal_orientation, sagittal_spacing)
            fig.add_trace(go.Scatter3d(
                x=[x_physical], y=[y_physical], z=[z_physical],
                mode='markers',
                # marker=dict(symbol='x', size=4, color=cross_color, line=dict(width=2)),
                marker=dict(symbol='x', size=4, color=color, line=dict(width=2)),
                name= str(condition)
            ))

    # Load and plot the axial images
    for axial_file, condition in zip(axial_files,axial_conditions):
        #axial_image, axial_position, axial_orientation, axial_spacing, _ = load_dicom_image(axial_file)
        axial_image, axial_position, axial_orientation, axial_spacing, _ = load_and_resize_image(axial_file)
        corners = get_corners(axial_position, axial_orientation, axial_spacing, axial_image.shape)
        
        x_coords = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]]
        y_coords = [corners[0][1], corners[1][1], corners[2][1], corners[3][1]]
        z_coords = [corners[0][2], corners[1][2], corners[2][2], corners[3][2]]
        
        
        x_grid, y_grid, z_grid = create_surface(x_coords, y_coords, z_coords, axial_image)
        # Rotate image
        axial_image = np.rot90(axial_image, 1)
        
        color = colormap[condition]
        
        if use_surfacecolor:
            fig.add_trace(go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                surfacecolor=axial_image,
                colorscale='Gray',
                showscale=False,
                name='Axial Slice'
            ))
            
        else:
            fig.add_trace(go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale=[[0, color], [1,color]],
                showscale=False,
                name='Axial Slice'
            ))
        # Add cross marks
        image_labels = label_data[label_data['image_path'] == axial_file]
        for _, row in image_labels.iterrows():
            cross_color = 'green' if row['label'] == 0 else 'red'
            x_pixel = row['x']//scale_factor
            y_pixel = row['y']//scale_factor
            x_physical, y_physical, z_physical = pixel_to_physical(x_pixel, y_pixel, axial_position, axial_orientation, axial_spacing)
            
            fig.add_trace(go.Scatter3d(
                x=[x_physical], y=[y_physical], z=[z_physical],
                mode='markers',
                marker=dict(symbol='circle', size=8, color=cross_color, line=dict(width=2)),
                name='Circle'
            ))
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        )
    )

    # fig.show(renderer='iframe')
    fig.show()

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
    plt.figure(figsize=(15, 7))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_resized, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot image with predictions
    plt.subplot(1, 2, 2)
    plt.imshow(image_resized, cmap='gray')
    plt.title('Predictions')
    
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
    plt.show()

