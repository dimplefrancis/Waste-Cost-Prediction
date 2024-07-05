# test_waste_cost_prediction.py

import unittest
import pandas as pd
import numpy as np
from data_preparation import handle_missing_values, standardize_values, merge_datasets
from feature_engineering import create_new_features, apply_polynomial_features
from utils import remove_outliers
from model import create_stacking_regressor
import config

class TestDataPreparation(unittest.TestCase):
    def test_handle_missing_values(self):
        data = pd.DataFrame({'A': [1, np.nan, 3], 'B': ['x', 'y', np.nan]})
        result = handle_missing_values(data)
        self.assertFalse(result.isnull().any().any())

    def test_standardize_values(self):
        self.assertEqual(standardize_values('TEST'), 'Test')
        self.assertEqual(standardize_values('test '), 'Test')
        self.assertEqual(standardize_values(123), 123)

    def test_merge_datasets(self):
        batch_data = pd.DataFrame({'Material': ['A', 'B'], 'Batch': [1, 2], 'Value': [10, 20]})
        fail_data = pd.DataFrame({'Product/Material': ['A'], 'Batch Number': [1], 'Fail': [True]})
        result = merge_datasets(batch_data, fail_data)
        self.assertEqual(len(result), 2)
        self.assertTrue('Failed' in result.columns)

class TestFeatureEngineering(unittest.TestCase):
    def test_create_new_features(self):
        data = pd.DataFrame({
            'Theoretical Yield': [100, 200],
            'G.R.Qty': [90, 180],
            'Total Input in ML': [110, 220],
            'Waste in ML': [10, 20]
        })
        result = create_new_features(data)
        self.assertIn('Yield_Efficiency', result.columns)
        self.assertIn('Waste_Percentage', result.columns)

    def test_apply_polynomial_features(self):
        X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = apply_polynomial_features(X)
        self.assertEqual(result.shape[1], 5)  # 2 original features + 3 polynomial features

class TestModel(unittest.TestCase):
    def test_create_stacking_regressor(self):
        model = create_stacking_regressor()
        self.assertIsNotNone(model)
        self.assertEqual(len(model.estimators), 4)  # Assuming 4 base models

if __name__ == '__main__':
    unittest.main()