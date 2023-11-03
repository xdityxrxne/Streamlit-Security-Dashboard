from flask import Flask, render_template, request, redirect, url_for, jsonify
from pathlib import Path
from PIL import Image
import settings
import helper
import cv2
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/index.html')
def start():
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/analytics.html')
def end():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
