from data_preparation import load_data, handle_missing_values, merge_datasets, preprocess_data, split_data, standardize_values
from feature_engineering import create_new_features, apply_polynomial_features, scale_features, select_features
from model import create_stacking_regressor, train_model, predict
from evaluation import evaluate_model, cross_validate, plot_actual_vs_predicted
from utils import remove_outliers, inverse_transform_target
import pandas as pd
from error_handling import error_handler, logger, DataError, ModelError
import config

@error_handler
def main():
    try:
        # Load data
        batch_data, fail_data = load_data()
        
        # Handle missing values
        fail_data = fail_data.drop(columns=['Additional Information'])
        batch_data = handle_missing_values(batch_data)
        fail_data = handle_missing_values(fail_data)
        
        # Standardize values in specific columns
        columns_to_standardize = [
            'SAP Fail Category / Description',
            'Sub Category - Fails',
            'Manufacturing Location',
            'Vessel',
            'Material'
        ]
        
        for col in columns_to_standardize:
            if col in batch_data.columns:
                batch_data[col] = batch_data[col].apply(standardize_values)
            if col in fail_data.columns:
                fail_data[col] = fail_data[col].apply(standardize_values)
        
        # Specific standardization for 'SAP Fail Category / Description'
        if 'SAP Fail Category / Description' in fail_data.columns:
            fail_data['SAP Fail Category / Description'] = fail_data['SAP Fail Category / Description'].replace({
                'Equipment': 'Equipment',
                'EQUIPMENT': 'Equipment',
                'equipment': 'Equipment',
                'Appearance': 'Appearance',
                'APPEARANCE': 'Appearance',
                'Performance': 'Performance',
                'performance': 'Performance',
                'PERFORMANCE': 'Performance',
                'Contamination': 'Contamination',
                'Manufacturing': 'Manufacturing',
                'Operator': 'Operator',
                'operator': 'Operator',
                'OPERATOR': 'Operator',
                'Ph': 'PH',
                'Documentation': 'Documentation'
            })

        # Merge datasets
        merged_data = merge_datasets(batch_data, fail_data)
        
        # Preprocess data
        X, y, pt = preprocess_data(merged_data)
        
        # Feature engineering
        #X = create_new_features(X)
        X = apply_polynomial_features(X)
        X = scale_features(X)
        
        # Remove outliers
        X, y = remove_outliers(X, y)
        
        # Feature selection
        X, selector = select_features(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Create and train model
        model = create_stacking_regressor()
        trained_model = train_model(model, X_train, y_train)
        
        # Make predictions
        y_pred = predict(trained_model, X_test)
        
        # Evaluate model
        metrics = evaluate_model(y_test, y_pred)
        print("Model performance (transformed scale):", metrics)
        
        # Cross-validate
        cv_scores = cross_validate(trained_model, X, y)
        print("Cross-validation scores:", cv_scores)
        
        # Plot results in transformed scale
        plot_actual_vs_predicted(y_test, y_pred, "Actual vs Predicted (Transformed Scale)")
        
        # Transform back to original scale
        y_test_original = inverse_transform_target(pt, y_test)
        y_pred_original = inverse_transform_target(pt, y_pred)
        
        # Evaluate model in original scale
        metrics_original = evaluate_model(y_test_original, y_pred_original)
        print("Model performance (original scale):", metrics_original)
        
        # Plot results in original scale
        plot_actual_vs_predicted(y_test_original, y_pred_original, "Actual vs Predicted (Original Scale)")
        logger.info("Process completed successfully")
    except DataError as e:
        logger.error(f"Data processing error: {str(e)}")
    except ModelError as e:
        logger.error(f"Model error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
if __name__ == "__main__":
    main()