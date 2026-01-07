from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import base64
from detect import WeedDetector
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Initialize detector
try:
    detector = WeedDetector('models/best.pt')
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    MODEL_LOADED = False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', model_loaded=MODEL_LOADED)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and detection"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Get confidence threshold from request
            conf_threshold = float(request.form.get('confidence', 0.25))
            
            # Detect weeds
            results = detector.detect_weeds(filepath, conf_threshold)
            
            # Save annotated image
            output_filename = f"annotated_{filename}"
            output_path = os.path.join('static/results', output_filename)
            cv2.imwrite(output_path, results['annotated_image'])
            
            # Convert images to base64 for display
            original_base64 = image_to_base64(filepath)
            annotated_base64 = image_to_base64(output_path)
            
            # Prepare response
            response = {
                'success': True,
                'original_image': original_base64,
                'annotated_image': annotated_base64,
                'detections': results['detections'],
                'total_weeds': results['total_weeds'],
                'image_size': results['image_size']
            }
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({'error': f'Detection failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download_coordinates', methods=['POST'])
def download_coordinates():
    """Generate downloadable coordinate file"""
    data = request.json
    detections = data.get('detections', [])
    
    # Format coordinates for laser system
    coordinate_data = {
        'total_targets': len(detections),
        'targets': []
    }
    
    for i, det in enumerate(detections, 1):
        target = {
            'target_id': i,
            'class': det['class'],
            'confidence': det['confidence'],
            'pixel_coordinates': det['center'],
            'laser_coordinates': det['laser_coordinates']
        }
        coordinate_data['targets'].append(target)
    
    return jsonify(coordinate_data)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': MODEL_LOADED
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Weed Detection & Laser Targeting System")
    print("="*50)
    print(f"Model Status: {'✓ Loaded' if MODEL_LOADED else '✗ Not Loaded'}")
    print("Server starting at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)