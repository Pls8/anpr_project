"""
ANPR Web Interface
Run in browser: http://localhost:5000

Features:
- Image upload for plate recognition
- Live video stream from webcam
"""

import os
import sys
import uuid
import cv2
import threading
import time
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
os.environ['LD_PRELOAD'] = '/opt/rocm/lib/libamdhip64.so'

sys.path.insert(0, os.path.dirname(__file__))

from anpr_yolo_app import ANPR

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

anpr = ANPR()

# Video streaming global state
video_streaming = False
video_capture = None
video_thread = None
current_video_frame = None
current_plate_text = "Waiting for camera..."
frame_lock = threading.Lock()


def video_capture_thread():
    """Background thread for capturing video frames"""
    global video_capture, current_video_frame, current_plate_text, video_streaming
    
    while video_streaming:
        if video_capture is not None and video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                # Save current frame for streaming
                with frame_lock:
                    current_video_frame = frame.copy()
                
                # Process frame with ANPR
                try:
                    # Save frame temporarily for ANPR
                    temp_path = 'static/uploads/temp_frame.jpg'
                    cv2.imwrite(temp_path, frame)
                    
                    # Run ANPR
                    plate = anpr.predict(temp_path)
                    current_plate_text = plate if plate else "No plate detected"
                    
                    # Draw result on frame
                    cv2.putText(frame, f"Plate: {current_plate_text}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    current_plate_text = f"Error: {str(e)}"
        
        time.sleep(0.03)  # ~30 FPS
    
    if video_capture is not None:
        video_capture.release()
        video_capture = None


def generate_frames():
    """Generator for video stream"""
    global current_video_frame
    
    while video_streaming:
        with frame_lock:
            if current_video_frame is not None:
                frame = current_video_frame.copy()
            else:
                continue
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video_page():
    """Video streaming page"""
    return render_template('video.html')


@app.route('/video/feed')
def video_feed():
    """Video streaming endpoint"""
    global video_streaming
    
    if not video_streaming:
        return "Camera not started", 400
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video/start', methods=['POST'])
def video_start():
    """Start video capture"""
    global video_streaming, video_capture, video_thread
    
    data = request.get_json() or {}
    camera_index = data.get('camera', 0)  # Default to webcam 0
    
    if video_streaming:
        return jsonify({'success': True, 'message': 'Already streaming'})
    
    # Try to open webcam (index or URL)
    try:
        video_capture = cv2.VideoCapture(camera_index)
    except Exception as e:
        return jsonify({'error': f'Could not open camera: {str(e)}'}), 400
    
    if not video_capture.isOpened():
        return jsonify({'error': f'Could not open webcam (index {camera_index}). Try different index (0-5) or use /video/start_url for network camera'}), 400
    
    video_streaming = True
    video_thread = threading.Thread(target=video_capture_thread)
    video_thread.daemon = True
    video_thread.start()
    
    return jsonify({'success': True, 'message': 'Video streaming started'})


@app.route('/video/start_url', methods=['POST'])
def video_start_url():
    """Start video capture from URL (e.g., DroidCam)"""
    global video_streaming, video_capture, video_thread
    
    data = request.get_json() or {}
    camera_url = data.get('url', 'http://192.168.1.100:4747/video')  # Default DroidCam URL
    
    if video_streaming:
        return jsonify({'success': True, 'message': 'Already streaming'})
    
    # Try to open camera URL
    video_capture = cv2.VideoCapture(camera_url)
    
    if not video_capture.isOpened():
        return jsonify({'error': f'Could not open camera URL: {camera_url}. Make sure DroidCam is running and accessible'}), 400
    
    video_streaming = True
    video_thread = threading.Thread(target=video_capture_thread)
    video_thread.daemon = True
    video_thread.start()
    
    return jsonify({'success': True, 'message': f'Video streaming started from {camera_url}'})


@app.route('/video/stop', methods=['POST'])
def video_stop():
    """Stop video capture"""
    global video_streaming, video_capture
    
    video_streaming = False
    
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    
    return jsonify({'success': True, 'message': 'Video streaming stopped'})


@app.route('/video/status')
def video_status():
    """Get current video status and plate"""
    global video_streaming, current_plate_text
    
    return jsonify({
        'streaming': video_streaming,
        'plate': current_plate_text
    })


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
    print("Features:")
    print("  - Image upload: http://localhost:5000/")
    print("  - Video stream: http://localhost:5000/video")
    app.run(host='0.0.0.0', port=5000, debug=True)
