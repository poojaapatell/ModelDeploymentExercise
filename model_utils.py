import joblib
import numpy as np
import logging


# load_artifacts: load the artifacts: model, encoders, and feature variables
import joblib

def load_artifacts():
    try:
        model = joblib.load("/Users/Pooja/Desktop/Data Projects/Model-Deployment-Exercise/ModelDeploymentExercise/model.pkl")
        encoder = joblib.load("/Users/Pooja/Desktop/Data Projects/Model-Deployment-Exercise/ModelDeploymentExercise/encoders.pkl")
        mod_vars = joblib.load("/Users/Pooja/Desktop/Data Projects/Model-Deployment-Exercise/ModelDeploymentExercise/mod_vars.pkl")  # Feature names used in training
        return model, encoder, mod_vars
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise FileNotFoundError("Model or encoding files are missing.")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

# make_predictions: generate model predictions and format json output
def make_predictions(input_data, original_data, model):
    try:
        # Check if the model is valid
        if model is None:
            raise ValueError("The model is not loaded or is invalid.")

        # Ensure input_data is in the correct format (a 2D array)
        if input_data.ndim != 2:
            raise ValueError("Input data should be a 2D array (n_samples, n_features).")

        # Ensure original_data is not empty and matches input_data length
        if not original_data:
            raise ValueError("Original data is empty. Cannot generate predictions without original data.")
        if len(original_data) != len(input_data):
            raise ValueError("Mismatch between input data and original data lengths.")

        # Get the positive class probability at index 1
        probabilities = model.predict_proba(input_data)[:, 1]

        # Apply threshold of 0.75 to determine predictions
        predictions = (probabilities >= 0.75).astype(int)

        # Prepare response structure
        response = [
            {
                "business_outcome": int(pred),
                "prediction": float(prob),
                "feature_inputs": row
            }
            for pred, prob, row in zip(predictions, probabilities, original_data)
        ]

        return response

    except ValueError as e:
        # Catch specific value errors (e.g., invalid model, data format issues)
        raise ValueError(f"Error in prediction generation: {str(e)}")

    except Exception as e:
        # Catch any other unexpected errors
        raise RuntimeError(f"Unexpected error during prediction: {str(e)}")




