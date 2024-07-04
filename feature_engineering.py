from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

def create_new_features(data):
    # Implement feature creation logic
    data['Yield_Efficiency'] = data['G.R.Qty'] / data['Theoretical Yield']
    data['Waste_Percentage'] = data['Waste in ML'] / data['Total Input in ML'] * 100
    return data

def apply_polynomial_features(X, degree=2):
    # Implement polynomial feature transformation
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Get feature names (compatible with both older and newer scikit-learn versions)
    if hasattr(poly, 'get_feature_names_out'):
        feature_names = poly.get_feature_names_out(X.columns)
    else:
        feature_names = poly.get_feature_names(X.columns)
    
    return pd.DataFrame(X_poly, columns=feature_names, index=X.index)

def scale_features(X):
    # Implement feature scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

def select_features(X, y, n_features_to_select=20):
    # Implement feature selection using RFE
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features_to_select)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.support_]
    return pd.DataFrame(X_selected, columns=selected_features), selector

def engineer_features(X, y):
    # Orchestrate the feature engineering process
    X = create_new_features(X)
    X = apply_polynomial_features(X)
    X = scale_features(X)
    X_selected, selector = select_features(X, y)
    return X_selected, selector

if __name__ == "__main__":
    # This block allows you to test the feature engineering process independently
    # You'll need to load your data here
    # For example:
    # data = pd.read_csv('your_data.csv')
    # X = data.drop('target', axis=1)
    # y = data['target']
    
    # X_engineered, selector = engineer_features(X, y)
    # print("Feature engineering complete.")
    # print(f"Engineered features shape: {X_engineered.shape}")
    # print(f"Selected features: {X_engineered.columns.tolist()}")
    pass