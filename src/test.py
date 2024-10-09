import pandas as pd
from xverse.transformer import WOE
from sklearn.preprocessing import OneHotEncoder

def perform_feature_engineering(df, customer_id_col, transaction_amount_col, transaction_start_time_col, target_col, categorical_cols):
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated parameter
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=categorical_cols, inplace=True)

    # Check the DataFrame columns after one-hot encoding
    print("Columns after one-hot encoding:", df.columns.tolist())

    feature_col = 'ProductCategory_airtime'  # Original column name
    if feature_col in df.columns:
        woe_transformer = WOE()  # Initialize WOE correctly, adjust params if necessary
        df['woe_transformed'] = woe_transformer.fit_transform(df[[feature_col]], df[target_col])  # Reshape into DataFrame
    else:
        # Check for new column names if original is not found
        print(f"Column '{feature_col}' not found in DataFrame.")
        new_feature_col = 'ProductCategory_airtime_True'  # Change this as needed
        if new_feature_col in df.columns:
            woe_transformer = WOE()  # Initialize WOE correctly, adjust params if necessary
            df['woe_transformed'] = woe_transformer.fit_transform(df[[new_feature_col]], df[target_col])  # Reshape into DataFrame
        else:
            print(f"Column '{new_feature_col}' also not found in DataFrame.")

    return df

if __name__ == "__main__":
    # Load your DataFrame
    df = pd.read_csv('./data/cleaned_data.csv')  # Ensure you provide the correct path
    print("Columns in DataFrame:", df.columns.tolist())  # Print columns

    customer_id_col = 'CustomerId'
    transaction_amount_col = 'Amount'
    transaction_start_time_col = 'TransactionStartTime'
    target_col = 'FraudResult'
    categorical_cols = ['ProductCategory_airtime', 'ProductCategory_data_bundles', 'ProductCategory_financial_services']

    result_df = perform_feature_engineering(df, customer_id_col, transaction_amount_col, transaction_start_time_col, target_col, categorical_cols)
    print(result_df.head())  # Display the result
# Assuming 'result_df' is your DataFrame
result_df.to_csv('./data/transformed_data.csv', index=False)
