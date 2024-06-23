from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
import os
import base64

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Get the API key from environment variable
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
# Initialize the InferenceHTTPClient with your API credentials
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# Dummy GET route
@app.route('/dummy', methods=['GET'])
def dummy_route():
    return jsonify({'message': 'Dummy route reached successfully'}), 200

# Endpoint to receive POST requests with image
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'}), 400

    image = request.files['image']

    # Make sure the image file has a valid filename
    if image.filename == '':
        return jsonify({'error': 'Image file must have a filename'}), 400

    try:
        # Read image data from request and encode as base64
        image_data = image.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Run inference using Roboflow API
        result = CLIENT.infer(image_base64, model_id="cat-breeds-obw8e/2")

        # Process the result as needed
        return jsonify(result), 200

    except Exception as e:
        print('Error processing request:', str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
