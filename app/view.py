from flask import render_template, Response, request
import tensorflow as tf
import numpy as np
import cv2
import os
import cvlib as cv

global capture, userface, prediction, switch
capture = 0
userface = 0
prediction = 0
switch = 0 

camera = cv2.VideoCapture(0)
emotion_model = tf.keras.models.load_model('app/static/model_cv.h5')
face_model = cv2.dnn.readNetFromCaffe('app/static/deploy.prototxt.txt', 'app/static/res10_300x300_ssd_iter_140000.caffemodel')

def detect_face(frame):
    global face_model
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    face_model.setInput(blob)
    detections = face_model.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame

def gen_frames():
    global capture, userface, prediction
    while True:
        success, frame = camera.read()
        if success:
            # take picture to predict emotion
            if capture:
                capture = 0
                userface = detect_face(frame)
                userface = cv2.resize(userface, (244,244))
                prediction = np.expand_dims(userface, axis=4)
                prediction = emotion_model.predict(prediction)
                print(np.argmax(prediction))

            # webcam stream
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass

def task():
    global capture, switch
    if request.method == 'POST':
        if request.form.get('click') == 'capture':
            capture = 1

        if request.form.get('stop') == 'Stop/Start':
            if (switch == 1):
                switch == 0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch == 1
    elif request.method == 'GET':
        return render_template('ourfeature.html')
    return render_template('ourfeature.html')
            
def home():
    return render_template('home.html')

def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
