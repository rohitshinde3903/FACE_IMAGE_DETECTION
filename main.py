import PIL
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from PIL import Image
import os

app = Flask(__name__)

# Load model
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json, custom_objects={"Sequential": Sequential})
model.load_weights("facialemotionmodel.h5")

# Haar Cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Labels
labels = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

# Feature extraction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# ----- Webcam feed disabled -----
# def generate_frames():
#     webcam = cv2.VideoCapture(0)
#     while True:
#         success, frame = webcam.read()
#         if not success:
#             break
#         else:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#             for (x, y, w, h) in faces:
#                 roi_gray = gray[y:y+h, x:x+w]
#                 roi_gray = cv2.resize(roi_gray, (48, 48))
#                 roi = extract_features(roi_gray)
#                 pred = model.predict(roi)
#                 label = labels[pred.argmax()]
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#                 cv2.putText(frame, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

# ----- Webcam route disabled -----
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    img = Image.open(file.stream).convert('L')  # grayscale
    img = img.resize((48, 48))
    img_array = np.array(img)
    img_array = extract_features(img_array)
    pred = model.predict(img_array)
    label = labels[pred.argmax()]
    return render_template('result.html', prediction=label)

import google.generativeai as genai
genai.configure(api_key="AIzaSyBJqnwsEDyhXkD73L04O7wxYHIxfCJLETU")
import PIL.Image
import io

@app.route('/gemini_upload', methods=['POST'])
def gemini_upload():
    if 'gemini_image' not in request.files:
        return "No Gemini image uploaded", 400

    file = request.files['gemini_image']
    if file.filename == '':
        return "No Gemini image selected", 400

    image_bytes = file.read()

    try:
        model = genai.GenerativeModel('gemini-pro-vision')
        img = PIL.Image.open(io.BytesIO(image_bytes))

        prompt = (
            "Based on the visual appearance of the person(s) in this image, please provide your best "
            "*estimation* of their age range (e.g., child, teenager, young adult, middle-aged, senior) and "
            "gender (e.g., male, female, or prefer not to specify). Acknowledge that this is an estimation "
            "based on visual cues and may not be accurate."
        )

        response = model.generate_content([prompt, img])
        result = response.text

        return render_template('gemini_result.html', gemini_prediction=result)

    except Exception as e:
        return f"Gemini processing failed: {e}", 500



if __name__ == "__main__":
    app.run()
