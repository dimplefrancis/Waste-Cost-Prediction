import numpy as np
from scipy import stats
import pandas as pd

def remove_outliers(X, y, z_threshold=3):
    # Implement outlier removal logic
    z_scores = np.abs(stats.zscore(y))
    mask = z_scores < z_threshold
    return X[mask], y[mask]

def inverse_transform_target(transformer, y):
    # Inverse transform the target variable
    return transformer.inverse_transform(y.reshape(-1, 1)).ravel()

def save_model(model, filename):
    # Save the model to a file
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    # Load the model from a file
    import joblib
    return joblib.load(filename)

def create_datetime_features(df, date_column):
    # Create datetime features from a date column
    df[date_column] = pd.to_datetime(df[date_column])
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['quarter'] = df[date_column].dt.quarter
    return df

def encode_categorical_variables(df, columns):
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

if __name__ == "__main__":
    # This block allows you to test the utility functions independently
    # You can add test code here if needed
    
    # Example usage:
    # X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [100, 200]])
    # y = np.array([1, 2, 3, 4, 1000])
    # X_clean, y_clean = remove_outliers(X, y)
    # print("Original shape:", X.shape)
    # print("Shape after removing outliers:", X_clean.shape)
    
    # from sklearn.preprocessing import PowerTransformer
    # pt = PowerTransformer()
    # y_transformed = pt.fit_transform(y.reshape(-1, 1)).ravel()
    # y_original = inverse_transform_target(pt, y_transformed)
    # print("Original y:", y)
    # print("Transformed y:", y_transformed)
    # print("Inverse transformed y:", y_original)
    
    pass