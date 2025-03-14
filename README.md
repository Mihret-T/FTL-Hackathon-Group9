# FTL-Hackathon-Group9
Counterfeit Drug Detection Using YOLOv8 &amp; OCR.This project is a Streamlit web app that detects counterfeit vs. authentic drugs using YOLOv8 object detection and OCR (Tesseract) for text extraction from drug packaging. 

# Requirements to Run the Counterfeit Drug Detection App
1. Install Python
Ensure you have Python 3.8 or later installed.
To check your Python version, run:
python --version
If Python is not installed, download it from python.org.

# 2. Install Required Libraries
Install all necessary dependencies using:
pip install -r requirements.txt
If requirements.txt is missing, manually install the following:
pip install streamlit numpy opencv-python ultralytics pytesseract pillow matplotlib

# Library Details
Library
Purpose
streamlit
Web app framework
numpy
Numerical computations
opencv-python
Image processing
ultralytics
YOLOv8 model for object detection
pytesseract
OCR for text extraction from images
pillow
Image handling
matplotlib
Visualization

# 3. Ensure YOLOv8 Model is Available
Ensure the trained model best.pt is inside the correct path:
runs/detect/train9/weights/best.pt
If missing, download or retrain the model before running the app.

# 4. Run the App
Navigate to the project folder and start Streamlit:
cd /path/to/project/
streamlit run app.py
Once running, open the provided localhost link in your browser.

# 5. Troubleshooting

A. Module Not Found Error
If you see ModuleNotFoundError, try:
pip install missing-library-name

B. YOLOv8 Model Not Found
Make sure best.pt is correctly placed inside runs/detect/train9/weights/.
