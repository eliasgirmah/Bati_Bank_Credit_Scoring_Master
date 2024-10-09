# tests/test_feature_engineering.py

import unittest
import pandas as pd
from src.feature_engineering import perform_feature_engineering

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.df = pd.DataFrame({
            'CustomerId': [1, 2, 1, 2],
            'TransactionAmount': [100, 200, 150, 300],
            'TransactionStartTime': ['2024-01-01 12:00:00', '2024-01-01 13:00:00', 
                                     '2024-01-02 12:30:00', '2024-01-02 14:30:00'],
            'FraudResult': [0, 1, 0, 1],
            'ProductCategory_airtime': [1, 0, 1, 0],
            'ProductCategory_data_bundles': [0, 1, 0, 1],
            'ProductCategory_financial_services': [0, 0, 1, 1],
            'ProductCategory_movies': [1, 1, 0, 0],
            'ProductCategory_other': [0, 0, 0, 0],
            'ProductCategory_ticket': [0, 0, 0, 0],
            'ProductCategory_transport': [0, 0, 0, 0],
            'ProductCategory_tv': [0, 0, 0, 0],
            'ProductCategory_utility_bill': [0, 0, 0, 0],
        })

    def test_feature_engineering(self):
        result_df = perform_feature_engineering(self.df, 'CustomerId', 'TransactionAmount', 
                                                'TransactionStartTime', 'FraudResult')
        
        # Assertions to validate the feature engineering
        self.assertIn('Transaction_Hour', result_df.columns)
        self.assertIn('Transaction_Day', result_df.columns)
        self.assertIn('Transaction_Weekday', result_df.columns)
        self.assertIn('Transaction_Month', result_df.columns)

        # Check if WoE encoding is applied (you can modify this based on actual values)
        self.assertTrue(all(result_df['ProductCategory_airtime'].notna()))
        self.assertTrue(all(result_df['ProductCategory_data_bundles'].notna()))

if __name__ == '__main__':
    unittest.main()
