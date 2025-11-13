# TrueSight: DeepFake Detection System

TrueSight is a full-stack web application designed to detect AI-generated deepfakes in both videos and images. It provides a simple, user-friendly interface for media analysis, supported by a powerful multi-modal AI backend.

This project was built as a final-year Computer Science project.


### ✨ Features

* **Hybrid AI Backend:** Uses two different state-of-the-art models for specialized detection:
    * **Video Detection:** An **Xception+LSTM** model (built with TensorFlow/Keras) analyzes video frames over time for temporal inconsistencies.
    * **Image Detection:** A **Vision Transformer (ViT)** model (built with PyTorch) analyzes still images for manipulation artifacts.
* **Full-Stack Application:** A responsive web interface (HTML/CSS/JS) built on a Flask (Python) server.
* **User Authentication:** Secure user sign-up and sign-in functionality implemented using **Firebase Authentication**.
* **Real-Time Results:** Get instant classification ("REAL" or "FAKE") with a confidence score and a visual breakdown of analyzed frames.
* **Containerized Deployment:** The entire application is containerized using **Docker** for easy and reliable deployment on platforms like Hugging Face Spaces.


### 💻 Technology Stack

* **Backend:** Python, Flask, Gunicorn
* **Frontend:** HTML, CSS, JavaScript
* **AI / Machine Learning:**
    * TensorFlow / Keras (for Xception+LSTM) 
    * PyTorch / Transformers (for ViT)
    * MTCNN (for face detection)
    * OpenCV (for video processing)
* **Authentication:** Firebase
* **Deployment:** Docker

---
