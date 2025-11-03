# Automatic Number Plate Recognition with YOLOv8

## 🔍 Overview

This project performs automatic vehicle and number plate detection using YOLOv8, SORT for object tracking, and EasyOCR for text recognition. It processes video input, tracks vehicles, detects license plates, and overlays recognized plate numbers onto the output video.

---

## 🕸️ Webpage
![preview](https://github.com/Faris-as/licenseplate-ocr-detection/blob/main/demo/top.png)

## 🎬 Output
[https://github.com/user-attachments/assets/874be1f3-15eb-4d4e-8896-c5c22922a411](https://github.com/user-attachments/assets/874be1f3-15eb-4d4e-8896-c5c22922a411)

---

## 📹 Sample Video

The sample video used in this project can be downloaded from:  
👉 [Pexels Traffic Flow Video](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/)

---

## 🧠 Models Used

- **Vehicle Detection**: YOLOv8n (pre-trained by Ultralytics)
- **License Plate Detection**: A custom YOLOv8 model trained on [this Roboflow dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)

📥 [Download the trained license plate detector model](https://drive.google.com/file/d/1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw/view?usp=sharing)

---

## ⚠️ Important Note

This project was tested with specific versions of `torch`, `torchvision`, `easyocr`, and `numpy`.  
Some libraries (like `torch`, `numpy`, `Pillow`, etc.) may require specific versions to work correctly due to breaking changes in newer releases.

🛠️ If you encounter issues like:
- `ANTIALIAS` attribute error from `Pillow`
- NumPy 2.x compatibility errors
- Unpickling errors from PyTorch

👉 You may need to **downgrade to:**
```bash
pip install numpy<2.0 torch==2.1.0 torchvision==0.16.0 Pillow==9.5.0 easyocr==1.6.1
```

## 🔧 Dependencies

Clone the SORT tracking module (required):

```bash
git clone https://github.com/abewley/sort
```

## ⚙️ Project Setup
1️⃣ Create a virtual environment (Python 3.10)
```bash
conda create --prefix ./env python=3.10 -y
conda activate ./env
```
2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
Or manually install with specific versions:

```bash
pip install ultralytics==8.0.20
pip install opencv-python
pip install easyocr==1.6.1
pip install scipy pandas matplotlib
pip install torch==2.1.0 torchvision==0.16.0 numpy<2.0 Pillow==9.5.0
```
## 🚀 Run the Pipeline
➤ Step 1: Detect and track vehicles
```bash
python main.py
```
➤ Step 2: Interpolate missing frame data
```bash
python add_missing_data.py
```
➤ Step 3: Visualize the final result
```bash
python visualize.py
```
This will generate out.mp4 with bounding boxes and overlaid license plate numbers.

### OR

You can run the code 
```bash
python runner.py --run all
```
This is used to run all the files in the order.
But if you want you can use runner.py to run each file individually by using 'detect', 'interpolate' & 'visualise' instead of all.

OR 
Best Way
Just run the page code
```bash
streamlit run app.py
```
🗃️ This will open a site with a input where you can upload the video directly.


![preview](https://github.com/Faris-as/licenseplate-ocr-detection/blob/main/demo/top.png)

The system will do a running symbol like this 🏃 to understand that this is working.
After processing one can download the video as you can see.

![view](https://github.com/Faris-as/licenseplate-ocr-detection/blob/main/demo/bottom.png)

## 📁 Project Structure
```bash
├── sample.mp4
├── main.py
├── add_missing_data.py
├── visualize.py
├── test.csv
├── test_interpolated.csv
├── license_plate_detector.pt
├── sort/
│   └── ...
└── assets/
    └── demo_preview.mp4
```
## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [SORT Object Tracker](https://github.com/abewley/sort)
- [Roboflow License Plate Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)
