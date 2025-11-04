import os
import time
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.xception import preprocess_input
from mtcnn import MTCNN
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import shutil

import torch
from transformers import ViTForImageClassification, ViTImageProcessor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TIME_STEPS = 30
HEIGHT, WIDTH = 299, 299

VIDEO_MODEL_PATH = 'models/Video_model.keras'
IMAGE_MODEL_PATH = 'models/'

app = Flask(__name__)

detector = None
video_model = None
image_model = None
image_processor = None

from threading import Lock
detector_lock = Lock()
video_model_lock = Lock()
image_model_lock = Lock()

def get_detector():
    global detector
    with detector_lock:
        if detector is None:
            try:
                print("Loading MTCNN detector...")
                detector = MTCNN()
                print("✅ MTCNN face detector initialized.")
            except Exception as e:
                print(f"Warning: Could not initialize MTCNN. Error: {e}")
        return detector

def get_video_model():
    global video_model
    with video_model_lock:
        if video_model is None:
            try:
                if not os.path.exists(VIDEO_MODEL_PATH):
                    print(f"Warning: Video model file not found at {VIDEO_MODEL_PATH}")
                else:
                    print("Loading VIDEO (Xception+LSTM) model...")
                    video_model = build_video_model()
                    video_model.load_weights(VIDEO_MODEL_PATH)
                    print("✅ VIDEO (Xception+LSTM) model loaded successfully.")
            except Exception as e:
                print(f"Error loading video model: {e}")
        return video_model

def get_image_model():
    global image_model, image_processor
    with image_model_lock:
        if image_model is None or image_processor is None:
            try:
                if not os.path.exists(os.path.join(IMAGE_MODEL_PATH, 'pytorch_model.bin')):
                    print(f"Warning: Image model files not found in directory: {IMAGE_MODEL_PATH}")
                else:
                    print("Loading IMAGE (Vision Transformer) model...")
                    image_processor = ViTImageProcessor.from_pretrained(IMAGE_MODEL_PATH)
                    image_model = ViTForImageClassification.from_pretrained(IMAGE_MODEL_PATH)
                    print("✅ IMAGE (Vision Transformer) model loaded successfully.")
            except Exception as e:
                print(f"Error loading image model: {e}.")
        return image_model, image_processor

def build_video_model(lstm_hidden_size=256, num_classes=2, dropout_rate=0.5):
    with tf.keras.backend.name_scope('model'):
        inputs = layers.Input(shape=(TIME_STEPS, HEIGHT, WIDTH, 3))
        base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, pooling='avg')
        base_model.trainable = False
        x = layers.TimeDistributed(base_model)(inputs)
        x = layers.LSTM(lstm_hidden_size, return_sequences=False)(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)
        return model

def preprocess_video_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    image = preprocess_input(image)
    return image

def extract_faces_from_video(video_path, start_time=0, duration=2, num_frames=TIME_STEPS):
    local_detector = get_detector()
    if local_detector is None: raise Exception("MTCNN detector is not initialized.")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None
    fps = cap.get(cv2.CAP_PROP_FPS);
    if fps == 0: fps = 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); total_duration = frame_count / fps if fps > 0 else 0
    if start_time >= total_duration and total_duration > 0: return None, None
    start_frame = int(start_time * fps); end_frame = min(int((start_time + duration) * fps), frame_count)
    if end_frame <= start_frame: end_frame = frame_count
    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = local_detector.detect_faces(frame_rgb)
        face_processed = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

        if detections:
            x, y, w, h = detections[0]['box']
            x, y = max(0, x), max(0, y)
            face = frame_rgb[y:y+h, x:x+w]
            try:
                face_image = Image.fromarray(face)
                face_processed = preprocess_video_image(face_image)
            except: pass
        frames.append(face_processed)
        if len(frames) == num_frames: break
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32))
    return np.expand_dims(np.array(frames), axis=0), frames

def run_image_detection_vit(image_path):
    local_image_model, local_image_processor = get_image_model()
    if local_image_model is None or local_image_processor is None:
        raise Exception("Image model (ViT) or processor is not loaded.")
    
    image = Image.open(image_path).convert("RGB")
    inputs = local_image_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = local_image_model(**inputs)
        
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    prediction_label = local_image_model.config.id2label[predicted_class_idx]
    
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probabilities[0][predicted_class_idx].item()
    
    return prediction_label, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect')
def detect_page():
    return render_template('detect.html')

@app.route('/detect-image')
def detect_image_page():
    return render_template('detect-image.html')

@app.route('/api/analyze-video', methods=['POST'])
def analyze_video_api():
    local_video_model = get_video_model()
    if local_video_model is None:
        return jsonify({"error": "Video model is not loaded."}), 500

    if 'video' not in request.files: return jsonify({"error": "No video file provided"}), 400
    file = request.files['video'];
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    temp_dir = tempfile.mkdtemp(); filename = secure_filename(file.filename)
    video_path = os.path.join(temp_dir, filename); file.save(video_path)
    start_process_time = time.time()
    
    try:
        video_array, display_frames = extract_faces_from_video(video_path, start_time=0)
        if video_array is None or video_array.shape[1] != TIME_STEPS:
            return jsonify({"error": "Unable to process video segment. No faces detected or video error."}), 500

        predictions = local_video_model.predict(video_array, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        probabilities = predictions[0]
        end_process_time = time.time()
        
        prediction_text = "REAL" if predicted_class == 0 else "FAKE"
        confidence_score = f"{probabilities[predicted_class]*100:.2f}"
        processing_time_str = f"{end_process_time - start_process_time:.2f}s"

        frame_paths = []
        for i, frame_data in enumerate(display_frames[:8]):
            img_data = np.clip((frame_data + 1) / 2 * 255, 0, 255).astype(np.uint8)
            img_data_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB_BGR)
            frame_filename = f"frame_{i}.jpg"; save_path = os.path.join('static', 'frames', frame_filename)
            cv2.imwrite(save_path, img_data_bgr); frame_paths.append(f"/static/frames/{frame_filename}")
            
        response_data = {
            "prediction": prediction_text, "confidence": confidence_score,
            "framesAnalyzed": TIME_STEPS, "processingTime": processing_time_str,
            "frames": frame_paths
        }
        return jsonify(response_data)
    except Exception as e: return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    finally:
        try:
            if os.path.exists(video_path): os.remove(video_path)
            if os.path.exists(temp_dir): os.rmdir(temp_dir)
        except Exception as e: print(f"Warning: Could not clean up temp file {video_path}. Error: {e}")

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image_api():
    local_image_model, _ = get_image_model()
    if local_image_model is None:
        return jsonify({"error": "Image model (ViT) is not loaded."}), 500

    if 'image' not in request.files: return jsonify({"error": "No image file provided"}), 400
    file = request.files['image'];
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    temp_dir = tempfile.mkdtemp(); filename = secure_filename(file.filename)
    image_path = os.path.join(temp_dir, filename); file.save(image_path)
    start_process_time = time.time()
    
    try:
        prediction_label, confidence = run_image_detection_vit(image_path)
        end_process_time = time.time()

        confidence_score = f"{confidence*100:.2f}"
        
        preview_filename = f"preview_{filename}"
        preview_save_path = os.path.join('static', 'frames', preview_filename)
        shutil.copy(image_path, preview_save_path)
        
        response_data = {
            "prediction": prediction_label.toUpperCase(),
            "confidence": confidence_score,
            "image_url": f"/static/frames/{preview_filename}"
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    finally:
        try:
            if os.path.exists(image_path): os.remove(image_path)
            if os.path.exists(temp_dir): os.rmdir(temp_dir)
        except Exception as e: print(f"Warning: Could not clean up temp file {image_path}. Error: {e}")