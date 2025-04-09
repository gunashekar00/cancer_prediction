from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# Load the trained model and scaler
model_data = joblib.load('models/svm_breast_cancer_model.pkl')
scaler = model_data['scaler']
model = model_data['model']

# Feature names (must match form input names)
feature_names = [
    'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
    'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
    'radius_error', 'texture_error', 'perimeter_error', 'area_error', 'smoothness_error',
    'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
    'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
    'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
]

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form using the feature names
        input_data = [float(request.form[feature]) for feature in feature_names]
        input_array = np.array([input_data])  # Shape (1, 30)

        # Scale and predict
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)

        result = 'Malignant' if prediction[0] == 0 else 'Benign'
        return render_template('index.html', prediction_text=f'The tumor is likely: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # default to 10000
    app.run(debug=True, port=port)
