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

if __name__ == "__main__":
    app.run()
