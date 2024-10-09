import unittest
import pandas as pd
import numpy as np  # Ensure NumPy is imported

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'Amount': [100, -20, 50, 0, 500]
        })

    def test_log_transformation(self):
        # Fix the np reference
        self.df['Log_Amount'] = self.df['Amount'].apply(lambda x: np.log1p(x) if x > 0 else 0)
        self.assertTrue(all(self.df['Log_Amount'] >= 0))

    def test_negative_values_removal(self):
        df_cleaned = self.df[self.df['Amount'] >= 0]
        print(df_cleaned)  # Debugging step, check output
        self.assertEqual(len(df_cleaned), 4)  # Adjust expected output if needed

if __name__ == '__main__':
    unittest.main()
