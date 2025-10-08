# Facebased-Attendence-Project
A real-time attendance system using facial recognition to automate the process of marking attendance. This system captures faces, trains a model, and recognizes individuals to log their attendance in CSV files.

---

## Features ✨
- **Real-time Face Detection**: Uses Haar Cascade classifier for face detection via webcam.
- **Attendance Automation**: Logs attendance with timestamps directly into CSV files.
- **Simple CLI Menu**: Easy-to-use command-line interface with 5 options.
- **Multi-User Support**: Capture and recognize multiple individuals.
- **Confidence-Based Recognition**: Only marks attendance if recognition confidence > 67%.
  
---

## Technologies Used 🛠️
- OpenCV (Face detection and LBPH recognition)
- Pandas (CSV data handling)
- Python Threading (For smoother training process)

---

## Installation ⚙️

### 1. Clone Repository
```bash
git clone [https://github.com/Faris-as/Facebased-Attendence-Project]
cd face-recognition-attendance-system
```
### 2. Install Dependencies
```bash
pip install opencv-python pandas Pillow
```
### 3. Setup Directories
Create these folders manually:

```bash
mkdir TrainingImage StudentDetails TrainingImageLabel Attendance
```

---

## Usage 🚀
Start the System

```bash
python homge.py
```

## Notes 📝
- 💡 ID Format: Must be numeric (e.g., 101, 202)

- 💡 Name Format: Must be alphabetical (e.g., "John_Doe")

- 💻 Webcam required (tested with default laptop cameras)

- ⚠️ First-time users should run "Check Camera" (Option 1) before capture

- 📊 Attendance files are saved in ``` Attendance/ ``` with format:
   ```Attendance_YYYY-MM-DD_HH-MM-SS.csv```
