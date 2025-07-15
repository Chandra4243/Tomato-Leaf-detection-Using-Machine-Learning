# 🍅 Tomato Leaf Disease Detection

A deep learning–powered web app that detects 10 types of **tomato leaf diseases** from uploaded images using a trained CNN model.  
Built with **Flask**, **TensorFlow**, and **Keras**, this project provides fast diagnosis and disease-specific recommendations via a web interface.

---

## 📸 Supported Disease Classes

- Tomato Bacterial Spot  
- Tomato Early Blight  
- Tomato Healthy  
- Tomato Late Blight  
- Tomato Leaf Mold  
- Tomato Septoria Leaf Spot  
- Tomato Target Spot  
- Tomato Yellow Leaf Curl Virus  
- Tomato Mosaic Virus  
- Tomato Two-Spotted Spider Mite  

---

## 🧪 How to Run This Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/tomato-leaf-disease-detection.git
cd tomato-leaf-disease-detection


2. Install Requirements
It's recommended to use a virtual environment.

pip install -r requirements.txt

3. Run the Web App
     app.py

4. Project Structure
Tomato Leaf Disease Detection/
├── app.py                     # Main Flask app
├── model.h5                   # Trained CNN model
├── Training.py                # Training script (optional)
├── static/
│   └── upload/                # Folder for uploaded images
├── templates/
│   ├── index.html             # Upload page
│   └── Tomato-*.html          # Disease-specific result pages
├── Dataset/                   # Sample/test images
├── requirements.txt           # Dependencies
└── README.md                  # You're reading it!

5.🧠 Model Details
Input size: 128x128 RGB images

Preprocessing: Normalization (/255)

Framework: TensorFlow/Keras

Output: Softmax over 10 classes

Architecture: Defined in Training.py

6. 🪪 License
This project is released for educational and research purposes only.

6. Author    Chandrabhushan Upadhyay 
Aspiring AI/ML Engineer | Deep Learning & Computer Vision Enthusiast
 contact me - chandr4243@gmail.com


															This project is released for educational and research purposes only.