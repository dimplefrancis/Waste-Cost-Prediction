from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(y_true, y_pred):
    # Calculate and return evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return metrics

def cross_validate(model, X, y, cv=5, scoring='r2'):
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores)
    }

def plot_actual_vs_predicted(y_true, y_pred, title):
    # Create and show the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, title):
    # Create and show residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_importance, feature_names, title):
    # Create and show feature importance plot
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(10, 12))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # This block allows you to test the evaluation functions independently
    # You'll need to load your data and model here
    # For example:
    # model = load_your_model()
    # X, y = load_your_data()
    # y_pred = model.predict(X)
    
    # metrics = evaluate_model(y, y_pred)
    # print("Evaluation metrics:", metrics)
    
    # cv_results = cross_validate(model, X, y)
    # print("Cross-validation results:", cv_results)
    
    # plot_actual_vs_predicted(y, y_pred, "Actual vs Predicted")
    # plot_residuals(y, y_pred, "Residual Plot")
    
    # if hasattr(model, 'feature_importances_'):
    #     plot_feature_importance(model.feature_importances_, X.columns, "Feature Importance")
    pass