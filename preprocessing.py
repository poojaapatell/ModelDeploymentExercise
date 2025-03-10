import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def handle_missing_columns(input_df, mod_vars):
    # Iterate over mod_vars and check if the columns are missing
    for column in mod_vars:
        if column not in input_df.columns:
            # Add the column with a default value
            input_df[column] = 0  # You can change this to "" for categorical columns
    return input_df

def preprocess_input(data, mod_vars, encoder):
    try:
        # Convert the input data into a pandas DataFrame
        input_df = pd.DataFrame(data)

        # Handle missing columns by adding them with default values
        input_df = handle_missing_columns(input_df, mod_vars)

        # Ensure the necessary columns are present in the input data
        missing_columns = set(mod_vars) - set(input_df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in input data: {', '.join(missing_columns)}")

        # Select the final features of the model
        input_df = input_df[mod_vars]

        for col in input_df.columns:
            if input_df[col].dtype == 'object':  # Check if the column is of string type
                if input_df[col].str.contains('\$').any():  # Check for currency format
                    input_df[col] = input_df[col].replace({'\$': '', ',': ''}, regex=True)  # Remove '$' and ','
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')  # Convert to numeric (NaN if invalid)

        # Handle any NaN values created by the currency conversion
        input_df = input_df.fillna(0) 

        # Check if there are categorical columns to apply One-Hot Encoding
        categorical_columns = input_df.select_dtypes(include=['object']).columns
        # Check if there are categorical columns to apply One-Hot Encoding
        categorical_columns = input_df.select_dtypes(include=['object']).columns

        if not categorical_columns.empty:
            # Apply OneHotEncoder to all categorical columns at once
            encoded_features = encoder.transform(input_df[categorical_columns])
            # Drop the original categorical columns after encoding
            input_df = input_df.drop(columns=categorical_columns)
        else:
            # If no categorical columns, encoded_features will be an empty array
            encoded_features = np.array([]).reshape(0, 0)

        # Combine numerical features and encoded categorical features into one array
        if encoded_features.size > 0:
            preprocessed_input_df = np.hstack((input_df.values, encoded_features))
        else:
            preprocessed_input_df = input_df.values  # Just use numerical features if no encoding

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
