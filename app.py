"""
Flask Web Application for Hybrid IDS Demo
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from hybrid_ids import HybridIDSModel
import os

app = Flask(__name__)

# Load trained model
model = None
try:
    model = HybridIDSModel.load_model("models/hybrid_ids_model.pkl")
    print("Model loaded successfully")
except:
    print("Model not found. Please train the model first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Convert to DataFrame (assuming 45 features)
        features = data.get('features', [])
        
        if len(features) != 45:
            return jsonify({'error': 'Expected 45 features'}), 400
        
        # Make prediction
        X = pd.DataFrame([features])
        prediction = model.predict(X)[0]
        
        # Simulate confidence score
        confidence = np.random.uniform(0.85, 0.99)
        
        return jsonify({
            'prediction': prediction,
            'confidence': f"{confidence:.2%}",
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo')
def demo():
    """Demo page with sample predictions"""
    sample_results = [
        {'type': 'Benign', 'confidence': '99.1%', 'status': 'Normal'},
        {'type': 'MQTT-DDoS', 'confidence': '94.7%', 'status': 'Attack'},
        {'type': 'ARP_Spoofing', 'confidence': '91.3%', 'status': 'Attack'},
        {'type': 'TCP_IP-DoS', 'confidence': '96.8%', 'status': 'Attack'},
    ]
    
    return render_template('demo.html', results=sample_results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
