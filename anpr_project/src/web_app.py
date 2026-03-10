"""
ANPR Web Interface
Run in browser: http://localhost:5000
"""

import os
import sys
import uuid
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
os.environ['LD_PRELOAD'] = '/opt/rocm/lib/libamdhip64.so'

sys.path.insert(0, os.path.dirname(__file__))

from anpr_yolo_app import ANPR

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

anpr = ANPR()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ['jpg', 'jpeg', 'png', 'bmp']:
        return jsonify({'error': 'Invalid file type'}), 400
    
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        plate_text = anpr.predict(filepath)
        
        return jsonify({
            'success': True,
            'image': f'/static/uploads/{filename}',
            'plates': [plate_text],
            'count': 1
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting ANPR Web Interface...")
    print("Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=True)
