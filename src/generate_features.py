"""
Aumatically generate features for model

Feature engineering:
- Numerical: log, sqrt, square, cube, interaction, ratio, binning, etc.
- Categorical: one-hot encoding, label encoding, target encoding, etc.

Notes
- Handle missing values
- Handle outliers
- Handle skewness

Data types
- Date
- Categorical
- Numerical

Typical dataset:
- Date
- Customer ID
- Miscellaneous Categorical and Numerical variables
- Target variable
"""

import pandas as pd
import numpy as np
import json
from itertools import permutations, product

# Sample data
df = pd.read_csv('data/sample_customer_data.csv')


def create_ratio_features(df):
    """
    Create ratio features based on the columns of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The DataFrame with ratio features added.
    """
    new_df = df.copy()
    
    for col1, col2 in permutations(df.columns, 2):
        # Create a new column name for the ratio
        new_col_name = f"{col1}_to_{col2}_ratio"

        # Compute ratio with conditions
        # Use np.where to handle the zero numerator and denominator cases
        new_df[new_col_name] = np.where(
            df[col1] == 0, 0,
            np.where(df[col2] == 0, 1, df[col1] / df[col2])
        )

    return new_df

def create_log_sqrt_features(df):
    """
    Create logarithm and square root features based on the columns of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The DataFrame with logarithm and square root features added.
    """
    new_df = df.copy()
    
    for col in df.columns:
        # Logarithm of the column (adding a small constant to avoid log(0))
        new_df[f'{col}_log'] = np.log(df[col] + 1e-5)

        # Square root of the column (only for non-negative values)
        new_df[f'{col}_sqrt'] = np.where(df[col] >= 0, np.sqrt(df[col]), np.nan)

    return new_df


def create_multiplication_features(df):
    """
    Create multiplication features based on the columns of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The DataFrame with multiplication features added.
    """
    new_df = df.copy()
    
    for col1, col2 in product(df.columns, repeat=2):
        # Create a new column name for the multiplication
        new_col_name = f"{col1}_times_{col2}"

        # Compute multiplication
        new_df[new_col_name] = df[col1] * df[col2]

    return new_df

def create_interaction_features(df):

    # Create the interaction terms
    df_ratios = create_ratio_features(df)
    df_log_sqrt = create_log_sqrt_features(df)
    df_multiplication = create_multiplication_features(df)
    df = pd.concat([df, df_ratios, df_log_sqrt, df_multiplication], axis=1)

    return df

def process_categorical_columns(df, threshold=0.10):
    """
    Process categorical columns based on frequency threshold. 
    Applies specified encoding and stores mappings for inference.

    Args:
    df (pandas.DataFrame): Input DataFrame with categorical columns.
    threshold (float): Frequency threshold for categorizing a value.

    Returns:
    pandas.DataFrame: Processed DataFrame.
    dict: Dictionary of mappings or encodings for each categorical column.
    """
    processed_df = df.copy()
    mappings = {}

    for col in processed_df.columns:
        # Convert column to uppercase
        processed_df[col] = processed_df[col].str.upper()

        # Calculate frequency of each value
        freq = processed_df[col].value_counts(normalize=True)

        if freq.max() >= threshold:
            # Keep values above threshold and replace others with 'OTHER'
            top_values = freq[freq >= threshold].index.tolist()
            processed_df[col] = processed_df[col].apply(lambda x: x if x in top_values else 'OTHER')
            mappings[col] = {'type': 'top_values', 'values': top_values}
        else:
            # Apply frequency encoding
            freq_map = freq.to_dict()
            processed_df[col] = processed_df[col].map(freq_map)
            mappings[col] = {'type': 'frequency', 'values': freq_map}

    return processed_df, mappings

def apply_mappings_inference(df, mappings):
    """
    Apply saved mappings to a new DataFrame during inference.

    Args:
    df (pandas.DataFrame): New DataFrame for inference.
    mappings (dict): Dictionary of saved mappings.

    Returns:
    pandas.DataFrame: DataFrame with applied mappings.
    """
    infer_df = df.copy()

    for col, mapping in mappings.items():
        if col in infer_df.columns:
            infer_df[col] = infer_df[col].str.upper()
            
            if mapping['type'] == 'top_values':
                top_values = mapping['values']
                infer_df[col] = infer_df[col].apply(lambda x: x if x in top_values else 'OTHER')
            elif mapping['type'] == 'frequency':
                freq_map = mapping['values']
                min_freq = min(freq_map.values())
                infer_df[col] = infer_df[col].map(freq_map).fillna(min_freq)

    return infer_df

# Example usage:
# Load the mappings
# with open('category_mappings.json', 'r') as f:
#     saved_mappings = json.load(f)

# Apply mappings to new data
# new_inference_df = pd.DataFrame({'Category': [...], 'AnotherCategory': [...]})
# infer_df = apply_mappings_inference(new_inference_df, saved_mappings)
