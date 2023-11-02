from flask import Flask, render_template, Response
from app import view

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return view.home()

@app.route('/video_feed')
def video_feed():
    return Response(view.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/newfeature',methods=['POST','GET'])
def task():
    return view.task()

