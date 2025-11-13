# TrueSight: DeepFake Detection System

TrueSight is a full-stack web application designed to detect AI-generated deepfakes in both videos and images. It provides a simple, user-friendly interface for media analysis, supported by a powerful multi-modal AI backend.

This project was built as a final-year Computer Science project.

### 🔗 Live Demo

You can try the live application here:
**[huggingface.co/spaces/ShadowArtisan/TrueSight](https://huggingface.co/spaces/ShadowArtisan/TrueSight)**

---

### ✨ Features

* **Hybrid AI Backend:** Uses two different state-of-the-art models for specialized detection:
    * **Video Detection:** An **Xception+LSTM** model (built with TensorFlow/Keras) analyzes video frames over time for temporal inconsistencies.
    * **Image Detection:** A **Vision Transformer (ViT)** model (built with PyTorch) analyzes still images for manipulation artifacts.
* **Full-Stack Application:** A responsive web interface (HTML/CSS/JS) built on a Flask (Python) server.
* **User Authentication:** Secure user sign-up and sign-in functionality implemented using **Firebase Authentication**.
* **Real-Time Results:** Get instant classification ("REAL" or "FAKE") with a confidence score and a visual breakdown of analyzed frames.
* **Containerized Deployment:** The entire application is containerized using **Docker** for easy and reliable deployment on platforms like Hugging Face Spaces.

---

### 📷 Screenshots

*(Optional: You can add your screenshots here. Just drag and drop them onto the GitHub README editor.)*

| Home Page | Video Detection | Image Results |
| :---: | :---: | :---: |
| 

[Image of Homepage]
 |  | 

[Image of Image Results]
 |

---

### 💻 Technology Stack

* **Backend:** Python, Flask, Gunicorn
* **Frontend:** HTML, CSS, JavaScript
* **AI / Machine Learning:**
    * [cite_start]TensorFlow / Keras (for Xception+LSTM) [cite: 52]
    * [cite_start]PyTorch / Transformers (for ViT) [cite: 52]
    * [cite_start]MTCNN (for face detection) [cite: 52]
    * OpenCV (for video processing)
* **Authentication:** Firebase
* [cite_start]**Deployment:** Docker [cite: 1]

---

### 🚀 How to Run Locally

To run this project on your local machine, follow these steps:

**1. Clone the Repository:**
```bash
git clone [https://github.com/N25O14I18R13/TrueSight-App.git](https://github.com/N25O14I18R13/TrueSight-App.git)
cd TrueSight-App
