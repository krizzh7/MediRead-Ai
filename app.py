"""
MediRead AI - Backend Server
Flask application that handles prescription image analysis using Gemini Vision API
"""

import os
import json
import base64
import re
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24).hex())
CORS(app)

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise ValueError("GEMINI_API_KEY is required. Please add it to your .env file.")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use the latest Gemini Vision model
    MODEL_NAME = 'gemini-1.5-flash'
    model = genai.GenerativeModel(MODEL_NAME)
    logger.info(f"Successfully initialized Gemini model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    raise

# Constants
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compress_image(image_bytes, max_dimension=1024, quality=85):
    """Compress image to reduce size while maintaining quality"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        # Resize if too large
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            ratio = min(max_dimension/width, max_dimension/height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image from {width}x{height} to {new_size[0]}x{new_size[1]}")
        
        # Compress
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        compressed = buffer.getvalue()
        logger.info(f"Compressed image from {len(image_bytes)} to {len(compressed)} bytes")
        return compressed
        
    except Exception as e:
        logger.error(f"Image compression error: {str(e)}")
        return image_bytes

def get_prescription_prompt():
    """Generate the prompt for Gemini Vision API"""
    return """You are a medical prescription reader AI. Analyze the uploaded image and extract all medical information.

**IMPORTANT RULES:**
1. ONLY extract what is CLEARLY VISIBLE in the prescription
2. If handwriting is unclear, mark as "Unclear" or "Cannot determine"
3. NEVER guess or hallucinate missing information
4. If a field doesn't exist, leave it empty
5. Be extremely careful with medicine names - they must match what's written

**Required Output Format (JSON only):**
{
    "patient": {
        "name": "",
        "age": "",
        "gender": "",
        "date": ""
    },
    "doctor": {
        "name": "",
        "qualification": "",
        "registration": ""
    },
    "diagnosis": "",
    "medicines": [
        {
            "name": "",
            "purpose": "",
            "dosage": "",
            "frequency": "",
            "timing": "",
            "duration": "",
            "route": "",
            "notes": "",
            "confidence": ""
        }
    ],
    "warnings": "",
    "simple_explanation": "",
    "unclear_text": [],
    "confidence_score": 0
}

**Medicines:** List ALL medicines mentioned. If dosage is unclear, mark as "Unclear".

**Simple Explanation:** Write in patient-friendly language. Example: "Take one 500mg Paracetamol tablet whenever you have fever, up to 4 times a day."

**Confidence:** If handwriting is poor, reduce confidence score to below 50%.

Return ONLY valid JSON. Do not include any other text or markdown formatting."""

def analyze_prescription(image_data):
    """Send image to Gemini Vision API and get analysis"""
    try:
        logger.info("Starting prescription analysis with Gemini Vision API")
        
        # Prepare prompt and image
        prompt = get_prescription_prompt()
        
        # Create image part
        image_part = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_data).decode('utf-8')
        }
        
        # Generate response
        logger.info("Sending request to Gemini API...")
        response = model.generate_content([prompt, image_part])
        
        # Extract JSON from response
        response_text = response.text
        logger.info(f"Received response from Gemini: {len(response_text)} characters")
        
        # Clean response - remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = re.sub(r'^json\s*', '', response_text, flags=re.IGNORECASE)
        response_text = response_text.strip()
        
        # Parse JSON
        result = json.loads(response_text)
        
        # Ensure all required fields exist
        required_fields = ['patient', 'doctor', 'diagnosis', 'medicines', 'warnings', 
                          'simple_explanation', 'unclear_text', 'confidence_score']
        
        for field in required_fields:
            if field not in result:
                if field in ['patient', 'doctor']:
                    result[field] = {}
                elif field == 'medicines':
                    result[field] = []
                elif field == 'unclear_text':
                    result[field] = []
                else:
                    result[field] = ""
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['success'] = True
        result['model_used'] = MODEL_NAME
        
        logger.info(f"Analysis complete. Confidence: {result.get('confidence_score', 0)}%")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Response text: {response_text[:200]}...")
        return {
            'success': False,
            'error': 'Failed to parse AI response. The AI returned invalid JSON.',
            'raw_response': response_text[:500]  # Truncate for safety
        }
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return {
            'success': False,
            'error': f'AI analysis failed: {str(e)}'
        }

@app.route('/')
def serve_index():
    """Serve the main application"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main API endpoint for prescription analysis"""
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided. Please upload an image.'
            }), 400
        
        file = request.files['image']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected. Please choose an image to analyze.'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Unsupported file format. Allowed: {", ".join(ALLOWED_EXTENSIONS).upper()}'
            }), 400
        
        # Read and compress image
        image_bytes = file.read()
        
        if len(image_bytes) > MAX_IMAGE_SIZE:
            return jsonify({
                'success': False,
                'error': f'Image too large. Maximum size: {MAX_IMAGE_SIZE // (1024*1024)}MB'
            }), 400
        
        if len(image_bytes) == 0:
            return jsonify({
                'success': False,
                'error': 'Empty image file. Please upload a valid image.'
            }), 400
        
        # Compress image
        compressed_image = compress_image(image_bytes)
        
        # Analyze prescription
        result = analyze_prescription(compressed_image)
        
        if result.get('success', False):
            return jsonify(result)
        else:
            # Return error with proper status
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'gemini_configured': GEMINI_API_KEY is not None
    })

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again later.'
    }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting MediRead AI server on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Gemini API Key configured: {bool(GEMINI_API_KEY)}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)