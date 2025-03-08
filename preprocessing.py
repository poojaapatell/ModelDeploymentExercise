import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# preprocess_input: preprocess input data to match trained model data
def preprocess_input(data, mod_vars, encoder):
    try:
        # Convert the input data into a pandas DataFrame
        input_df = pd.DataFrame(data)

        # Ensure the necessary columns are present in the input data
        missing_columns = set(mod_vars) - set(input_df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in input data: {', '.join(missing_columns)}")

        # Select the final features of the model
        input_df = input_df[mod_vars]

        # Check if there are categorical columns to apply One-Hot Encoding
        categorical_columns = input_df.select_dtypes(include=['object']).columns
        if categorical_columns.empty:
            encoded_features = np.array([]).reshape(0, 0)  # Handle case where there are no categorical columns
        else:
            encoded_features = encoder.transform(input_df[categorical_columns])

        # Drop the original categorical columns
        input_df = input_df.drop(columns=categorical_columns)

        # Combine numerical features and encoded categorical features into 1 array
        preprocessed_input_df = np.hstack((input_df, encoded_features))

        return preprocessed_input_df
    
    except KeyError as e:
        # Catches KeyError if a column from mod_vars is not found in the input data
        raise ValueError(f"Column missing in input data: {str(e)}")
    
    except ValueError as e:
        # Catches issues such as missing columns, encoding issues, etc.
        raise ValueError(f"Error in data preprocessing: {str(e)}")
    
    except Exception as e:
        # Catches unexpected errors during preprocessing
        raise RuntimeError(f"Unexpected error during preprocessing: {str(e)}")
