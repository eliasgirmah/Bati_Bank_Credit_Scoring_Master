# src/feature_engineering.py

import pandas as pd
from category_encoders.woe import WOEEncoder

def apply_woe_binning(df, target_col, categorical_cols):
    """
    Apply WoE binning to categorical columns using WOEEncoder from category_encoders.
    
    Parameters:
    - df: DataFrame containing the data
    - target_col: The name of the target column
    - categorical_cols: List of categorical column names to apply WoE on
    
    Returns:
    - df: DataFrame with WoE encoded columns
    """
    encoder = WOEEncoder(cols=categorical_cols)
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols], df[target_col])
    return df

def perform_feature_engineering(df, customer_id_col, transaction_amount_col, transaction_start_time_col, target_col):
    """
    Perform feature engineering on the given DataFrame.
    
    Parameters:
    - df: DataFrame containing the raw data
    - customer_id_col: Name of the customer ID column
    - transaction_amount_col: Name of the transaction amount column
    - transaction_start_time_col: Name of the transaction start time column
    - target_col: The name of the target column
    
    Returns:
    - df: DataFrame with engineered features
    """
    # Ensure that the DataFrame has the required columns
    required_cols = [customer_id_col, transaction_amount_col, transaction_start_time_col, target_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame is missing one of the required columns.")
    
    # Feature engineering steps
    df['Transaction_Hour'] = pd.to_datetime(df[transaction_start_time_col]).dt.hour
    df['Transaction_Day'] = pd.to_datetime(df[transaction_start_time_col]).dt.day
    df['Transaction_Weekday'] = pd.to_datetime(df[transaction_start_time_col]).dt.weekday
    df['Transaction_Month'] = pd.to_datetime(df[transaction_start_time_col]).dt.month

    # Apply WoE binning on specified categorical columns
    categorical_cols = ['ProductCategory_airtime', 'ProductCategory_data_bundles', 'ProductCategory_financial_services',
                        'ProductCategory_movies', 'ProductCategory_other', 'ProductCategory_ticket',
                        'ProductCategory_transport', 'ProductCategory_tv', 'ProductCategory_utility_bill']
    
    df = apply_woe_binning(df, target_col, categorical_cols)
    
    return df
