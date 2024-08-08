import streamlit as st
import cv2
import os
import base64
import math
import numpy as np
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO('best.pt')

# Define folder paths
input_images_folder = "input-images"
output_images_folder = "output-images"

# Ensure the existence of the folders
os.makedirs(input_images_folder, exist_ok=True)
os.makedirs(output_images_folder, exist_ok=True)

# Function to perform object detection
def detect_objects(frame):
    results = model.predict(source=frame, conf=0.5)  # You can adjust the confidence threshold as needed
    classNames = results[0].names

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # class name
            cls = int(box.cls[0])

            # Display name, confidence, and bounding box info
            text = f"{classNames[cls]} : {confidence:.2f}"
            org = (x1, y1 - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(frame, text, org, font, fontScale, color, thickness)

    return frame

# Streamlit App
st.markdown(
    """
    <style>
    /* Background image */
    .stApp {
        background-image: url("background-image.jpeg"); /* Ensure the image is in the same directory */
        background-size: cover; /* Cover the entire page */
        background-position: center; /* Center the image */
        height: 100vh; /* Full height */
    }
    
    /* Title styling */
    .title {
        text-align: center;
        color: #000000; /* Black color */
        font-size: 36px; /* Adjust the size as needed */
        font-weight: bold; /* Make the text bold */
        margin-top: 20px; /* Add some margin from the top */
    }
    </style>
    <div class="title">RSD26</div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the file as an image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Process the image
    output_img = detect_objects(img)

    # Save the output image
    output_image_path = os.path.join(output_images_folder, "output_" + uploaded_file.name)
    cv2.imwrite(output_image_path, output_img)

    # Display the images
    def image_to_base64(image_path):
        with open(image_path, "rb") as file:
            return base64.b64encode(file.read()).decode('utf-8')

    input_image_base64 = base64.b64encode(file_bytes).decode('utf-8')
    output_image_base64 = image_to_base64(output_image_path)

    st.image(f"data:image/jpeg;base64,{input_image_base64}", caption='Uploaded Image', use_column_width=True)
    st.image(f"data:image/jpeg;base64,{output_image_base64}", caption='Processed Image', use_column_width=True)
