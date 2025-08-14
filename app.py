from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import pandas as pd

# Create the Flask application
app = Flask(__name__)

# Load the trained model and the pipeline
try:
    model = load('Dragon.joblib')
    pipeline = load('pipeline.joblib')
except FileNotFoundError:
    print("Error: 'Dragon.joblib' or 'pipeline.joblib' not found. Please train and save your model and pipeline first.")
    exit()
except Exception as e:
    print(f"Error loading model or pipeline: {e}")
    exit()

@app.route('/')
def home():
    """
    Renders the home page with the prediction form.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the form submission and returns a prediction.
    """
    try:
        # Get data from the request form
        data = {
            'CRIM': float(request.form['CRIM']),
            'ZN': float(request.form['ZN']),
            'INDUS': float(request.form['INDUS']),
            'CHAS': float(request.form['CHAS']),
            'NOX': float(request.form['NOX']),
            'RM': float(request.form['RM']),
            'AGE': float(request.form['AGE']),
            'DIS': float(request.form['DIS']),
            'RAD': float(request.form['RAD']),
            'TAX': float(request.form['TAX']),
            'PTRATIO': float(request.form['PTRATIO']),
            'B': float(request.form['B']),
            'LSTAT': float(request.form['LSTAT'])
        }

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([data])

        # Use the pipeline to transform the input data
        prepared_data = pipeline.transform(input_df)

        # Make a prediction using the loaded model
        prediction = model.predict(prepared_data)[0]

        # Return the prediction in a user-friendly format
        return render_template('result.html', prediction=f'{prediction:.2f}')

    except ValueError:
        return jsonify({'error': 'Invalid input. Please ensure all fields are numeric.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # This block will run the app directly when the script is executed.
    # It is a more robust way to run the app, especially with older Flask versions.
    # The debug=True flag enables development mode.
    app.run(debug=True)

