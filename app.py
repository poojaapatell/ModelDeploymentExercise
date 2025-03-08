from flask import Flask
from flask import request
from flask import jsonify
import logging
from preprocessing import preprocess_input
from model_utils import load_artifacts 
from model_utils import make_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model, encoder, and feature variables when the application starts
model, encoder, mod_vars = load_artifacts()

# Initialize Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])  # Define a route to handle POST requests for predictions
def predict():
    try:
        # Get the input data from the POST request's JSON body
        data = request.get_json()

        # If input data is a single dictionary (not a list), convert it into a list
        if isinstance(data, dict):
            data = [data]
        
        # Preprocess  input data to match the format the model expects
        input_data = preprocess_input(data, mod_vars, encoder)

        # Generate predictions using the preprocessed input data and the model
        response = make_predictions(input_data, data, model)

        # Return the predictions in JSON format
        return jsonify(response)

    except Exception as e:
        # If any error occurs, log it and return an error message
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)})

# Run the Flask application on all available IP addresses (0.0.0.0) at port 8080
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
