from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Initialize Flask App
app = Flask(__name__)

# Constants
IMG_SIZE = (224, 224)
THRESHOLD = 0.5

# Load Trained Model
model = load_model("model/best_model.keras")

print("Real trained model loaded successfully!")

# Home Route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    # Check image uploaded
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image uploaded'
        }), 400

    image = request.files['image']

    try:
        # Load image
        img = load_img(image, target_size=IMG_SIZE)

        # Convert image to array
        img_array = img_to_array(img)

        # Normalize image
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)

        # Convert prediction to probability
        probability = float(prediction[0][0])

        # Confidence percentage
        confidence = round(probability * 100, 2)

        # Risk classification
        if probability >= THRESHOLD:
            risk_level = "High Risk"
        else:
            risk_level = "Low Risk"

        # Return prediction result
        return jsonify({
            'risk_level': risk_level,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
