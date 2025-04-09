import pickle
import numpy as np
from flask import Flask, request, render_template

# Load model and scaler
with open('models/breast_cancer_svm.pkl', 'rb') as file:
    model, scaler = pickle.load(file)  # Unpack model and scaler

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values and convert to array
        data = [float(x) for x in request.form.values()]
        features = np.array([data])
        
        # Scale input data
        scaled_features = scaler.transform(features)  

        # Make prediction
        prediction = model.predict(scaled_features)[0]

        result = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-Cancerous)"
        
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        return str(e)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)


