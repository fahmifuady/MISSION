from flask import Flask, render_template, Response
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

app = Flask(__name__)

from app import routes
