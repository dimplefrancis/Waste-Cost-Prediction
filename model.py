from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import numpy as np
from error_handling import error_handler, ModelError, logger
import config

@error_handler
def create_stacking_regressor():
    try:
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
        logger.info("Stacking regressor created successfully")
        return stacking_regressor
    except Exception as e:
        logger.error(f"Error creating stacking regressor: {str(e)}")
        raise ModelError(f"Unable to create stacking regressor: {str(e)}")

@error_handler
def train_model(model, X_train, y_train):
    try:
        # Train the model
        model.fit(X_train, y_train)
        logger.info("Model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise ModelError(f"Error in model training: {str(e)}")

@error_handler
def predict(model, X_test):
    try:
        predictions = model.predict(X_test)
        logger.info(f"Predictions made for {len(X_test)} samples")
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise ModelError(f"Error in making predictions: {str(e)}")

@error_handler
def evaluate_model(model, X, y):
    try:
        scores = cross_val_score(model, X, y, cv=config.CV_FOLDS, scoring='r2')
        logger.info(f"Model evaluated. Mean R2 score: {np.mean(scores):.4f}")
        return scores
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise ModelError(f"Error in model evaluation: {str(e)}")

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