import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt

# Load trained YOLOv8 model
model = YOLO("runs/detect/train9/weights/best.pt")  # Update with your trained model path

#Streamlit UI
st.title("💊 Counterfeit Drug Detection")
st.write("Upload an image of drug packaging to detect authenticity.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Run YOLOv8 object detection
    results = model(image)

    # Display results
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("📌 **Detection Results:**")

    # Show detected objects
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            conf = box.conf[0].item()
            st.write(f"🟢 **{class_name}** detected with confidence {conf:.2f}")

    # Apply OCR (Extract text)
    st.write("📌 **Extracted Text (OCR):**")
    text = pytesseract.image_to_string(image)
    st.write(text)

# # Streamlit UI
# st.set_page_config(page_title="Counterfeit Drug Detection", layout="wide")

# # Sidebar for file upload
# st.sidebar.title("💊 Counterfeit Drug Detection")
# st.sidebar.write("Upload an image of drug packaging to detect authenticity.")

# uploaded_file = st.sidebar.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

# # Set theme
# st.markdown(
#     """
#     <style>
#     body { background-color: #f5f5f5; }
#     .reportview-container { padding: 20px; }
#     .sidebar .sidebar-content { background-color: #e6e6e6; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Function to display detection results
# def draw_bounding_boxes(image, results):
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#             class_name = model.names[int(box.cls)]
#             conf = box.conf[0].item()

#             # Draw bounding box
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     return image

# # Process uploaded image
# if uploaded_file:
#     image = Image.open(uploaded_file)
#     image_np = np.array(image)

#     # YOLOv8 Detection
#     results = model(image_np)

#     # Draw bounding boxes
#     image_with_boxes = draw_bounding_boxes(image_np.copy(), results)

#     # Display image with detections
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(image, caption="📌 Uploaded Image", use_column_width=True)

#     with col2:
#         st.image(image_with_boxes, caption="✅ Detection Results", use_column_width=True)

#     # Display detection confidence
#     st.subheader("📊 Detection Confidence")
#     for result in results:
#         for box in result.boxes:
#             class_name = model.names[int(box.cls)]
#             conf = box.conf[0].item()
#             st.progress(conf)
#             st.write(f"**{class_name}** detected with confidence **{conf:.2f}**")

#     # Apply OCR (Extract text)
#     extracted_text = pytesseract.image_to_string(image)

#     # Display OCR results
#     with st.expander("📖 Extracted Text (OCR)"):
#         st.text(extracted_text)

#     st.success("✅ Process completed successfully!")
