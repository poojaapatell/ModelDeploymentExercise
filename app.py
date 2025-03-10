from flask import Flask, request, jsonify
import logging
from preprocessing import preprocess_input
from model_utils import load_artifacts
from model_utils import make_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model, encoder, and feature variables when the application starts
model, encoder, mod_vars = load_artifacts()

# Initialize Flask application
app = Flask(__name__)

@app.route('/inference/', methods=['POST'])
def predict():
    try:
        # Check if the request content type is JSON
        if request.content_type != 'application/json':
            return jsonify({"error": "Invalid Content-Type, expected application/json"}), 400

        # Get the input data from the POST request's JSON body
        data = request.get_json()

        # Handle case where data is None (empty body or invalid JSON)
        if not data:
            return jsonify({"error": "No data provided or invalid JSON"}), 400

        logging.info(f"Parsed JSON: {data}")

        # If input data is a single dictionary (not a list), convert it into a list
        if isinstance(data, dict):
            data = [data]

        # Handle null values in the data (you can replace them with 0, or another value, as required)
        for record in data:
            for key, value in record.items():
                if value is None:
                    record[key] = 0  # Replace None with 0 (or another default value as needed)
                    logging.info(f"Null value found for key '{key}', replaced with 0")

        # Preprocess input data to match the format the model expects
        input_data = preprocess_input(data, mod_vars, encoder)

        # Generate predictions using the preprocessed input data and the model
        response = make_predictions(input_data, data, model)

        # Return the predictions in JSON format
        return jsonify(response)

    except Exception as e:
        # If any error occurs, log it and return an error message
        logging.error(f"Prediction error: {str(e)}")
        logging.error("Exception Traceback: ", exc_info=True)
        return jsonify({"error": str(e)}), 500


# Run the Flask application on all available IP addresses (0.0.0.0) at port 8080
if __name__ == '__main__':
    app.run(port=8080)
