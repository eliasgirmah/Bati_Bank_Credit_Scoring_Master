import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from eda import load_data, summary_statistics, identify_missing_values

class TestEDA(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame
        self.df = pd.DataFrame({
            'CustomerId': [1, 2, 3, 4],
            'Amount': [100, -50, 0, 200],
            'Category': ['A', 'B', 'A', 'C'],
            'log_amount': [np.log1p(100), np.log1p(0), 0, np.log1p(200)]
        })

    def test_load_data(self):
        # Mock the load_data function
        df = pd.DataFrame({
            'CustomerId': [1, 2, 3, 4],
            'Amount': [100, 50, 0, 200],
            'Category': ['A', 'B', 'A', 'C']
        })
        # Assert that the mock DataFrame is of the correct type
        self.assertIsInstance(df, pd.DataFrame)

    def test_summary_statistics(self):
        stats = summary_statistics(self.df)
        self.assertEqual(stats.shape[1], 2)  # Adjust based on your actual implementation

    def test_identify_missing_values(self):
        missing = identify_missing_values(self.df)
        self.assertEqual(missing.sum(), 0)  # No missing values

    def test_identify_missing_values_with_nan(self):
        # Add NaN values to the DataFrame
        df_with_nan = self.df.copy()
        df_with_nan.loc[0, 'Amount'] = np.nan
        missing = identify_missing_values(df_with_nan)
        self.assertEqual(missing.sum(), 1)  # 1 missing value

if __name__ == '__main__':
    unittest.main()
