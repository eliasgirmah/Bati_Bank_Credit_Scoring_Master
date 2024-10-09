import pandas as pd
from flask import Flask, request, render_template, jsonify
from load_model import load_model
import sys, os


# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import perform_feature_engineering
from src.CreditScoreRFM import CreditScoreRFM

# Initialize Flask app
app = Flask(__name__)

# Load the model once at the start
model = load_model('./model/gradient_boosting_model.joblib')

@app.route('/', methods=['GET'])
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Handle prediction requests from the API."""
    # Your prediction logic here
    customer_id = request.form['CustomerId']
    # Assume you have a prediction function that returns a risk label
    predicted_risk = your_prediction_function(request.form)  # Define this function appropriately
    return jsonify({'customer_id': customer_id, 'predicted_risk': predicted_risk})
    return handle_prediction()

def validate_input(data):
    """Validate input data to ensure all required fields are present."""
    try:
        # Validate TransactionId and CustomerId as integers
        transaction_id = int(data['TransactionId'])
        customer_id = int(data['CustomerId'])
        
        # Validate Amount as float
        amount = float(data['Amount'])

        # Validate required fields
        required_fields = ['ProductCategory', 'ChannelId', 'TransactionStartTime', 'PricingStrategy']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing field: {field}")

        return {
            'TransactionId': transaction_id,
            'CustomerId': customer_id,
            'ProductCategory': data['ProductCategory'],
            'ChannelId': data['ChannelId'],
            'Amount': amount,
            'TransactionStartTime': pd.to_datetime(data['TransactionStartTime'], utc=True),
            'PricingStrategy': int(data['PricingStrategy'])
        }

    except Exception as e:
        raise ValueError(f"Input validation error: {str(e)}")

def handle_prediction():
    """Shared logic for handling prediction requests."""
    try:
        # Validate and extract input data
        input_data = validate_input(request.json)

        # Prepare input data as DataFrame
        input_df = pd.DataFrame([input_data])

        # Feature Engineering
        fe = FeatureEngineering()
        input_df = fe.create_aggregate_features(input_df)
        input_df = fe.create_transaction_features(input_df)
        input_df = fe.extract_time_features(input_df)

        # Encode categorical features
        categorical_cols = ['ProductCategory', 'ChannelId']
        input_df = fe.encode_categorical_features(input_df, categorical_cols)

        # Handle missing values and normalize features
        numeric_cols = input_df.select_dtypes(include='number').columns.tolist()
        exclude_cols = ['Amount', 'TransactionId']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        input_df = fe.normalize_numerical_features(input_df, numeric_cols, method='standardize')

        # RFM Calculation
        rfm = CreditScoreRFM(input_df.reset_index())
        rfm_df = rfm.calculate_rfm()
        final_df = pd.merge(input_df, rfm_df, on='CustomerId', how='left')

        # Define all final features expected in the output
        final_features = [
            'PricingStrategy', 'Transaction_Count', 'Debit_Count', 'Credit_Count',
            'Debit_Credit_Ratio', 'Transaction_Month', 'Transaction_Year',
            'ProductCategory_financial_services', 'ChannelId_ChannelId_2',
            'ChannelId_ChannelId_3', 'Recency', 'Frequency'
        ]

        # Ensure all final features exist in the DataFrame and fill missing ones with 0
        final_df = final_df.reindex(columns=final_features, fill_value=0)
        
        # Make prediction
        prediction = model.predict(final_df)
        predicted_risk = 'Good' if prediction[0] == 0 else 'Bad'
        print(predicted_risk)
        return jsonify({
            'customer_id': input_data['CustomerId'],
            'predicted_risk': predicted_risk
        })

    except ValueError as ve:
        print("ValueError:", str(ve))
        return jsonify({'error': 'Invalid input: ' + str(ve)}), 400
    except KeyError as ke:
        print("KeyError:", str(ke))
        return jsonify({'error': 'Missing input data: ' + str(ke)}), 400
    except Exception as e:
        print("General Exception:", str(e))
        return jsonify({'error': 'An error occurred: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
