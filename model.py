from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import numpy as np

def create_stacking_regressor():
    # Define and return the stacking regressor
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42))
    ]
    
    meta_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    
    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )
    
    return stacking_regressor

def train_model(model, X_train, y_train):
    # Train the model
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    # Make predictions
    return model.predict(X_test)

def evaluate_model(model, X, y, cv=5):
    # Evaluate the model using cross-validation
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return scores

if __name__ == "__main__":
    # This block allows you to test the model independently
    # You'll need to load your data here
    # For example:
    # X_train, X_test, y_train, y_test = load_your_data()
    
    # model = create_stacking_regressor()
    # trained_model = train_model(model, X_train, y_train)
    # predictions = predict(trained_model, X_test)
    # scores = evaluate_model(trained_model, X_train, y_train)
    # print(f"Cross-validation scores: {scores}")
    # print(f"Mean R2 score: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
    pass