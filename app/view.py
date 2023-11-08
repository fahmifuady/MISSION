from flask import render_template, Response, request
import tensorflow as tf
import numpy as np
import cv2
import os
import cvlib as cv

global capture, userface, prediction, switch, camera
camera = 0
capture = 0
userface = 0
prediction = 0
switch = 1

camera = cv2.VideoCapture(0)

def gen_frames():
    global capture, userface, prediction
    while True:
        success, frame = camera.read()
        if success:
            # take picture to predict emotion
            if capture:
                capture = 0
                # ToDo: add function from model.py to detect emotion on frame.

            # webcam stream
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print("Error nya itu " + e)
                pass
        else:
            pass

def task():
    global capture, switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'capture':
            capture = 1

        elif request.form.get('stop') == 'on/off':
            if (switch == 1):
                switch == 0
                camera.release()

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
