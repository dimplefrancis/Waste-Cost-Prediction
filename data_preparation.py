import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from error_handling import error_handler, DataError, logger
import config

@error_handler
def load_data():
    try:
        batch_data = pd.read_csv(config.DATA_PATH['batch_data'])
        fail_data = pd.read_csv(config.DATA_PATH['fail_data'])
        logger.info(f"Data loaded successfully. Batch data shape: {batch_data.shape}, Fail data shape: {fail_data.shape}")
        return batch_data, fail_data
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise DataError(f"Unable to load data: {str(e)}")
    except pd.errors.EmptyDataError:
        logger.error("The file is empty.")
        raise DataError("The data file is empty.")

@error_handler
def handle_missing_values(data):
    try:
        # Convert specified columns to numeric, coercing errors to NaN
        columns_to_convert = [
            'Order Quantity (Expected Yield)',
            'Total Quantity Produced (plate packs / bottles)',
            'Cost Variance (Expected Yield v Actual)',
            'Costing'
        ]
        
        for column in columns_to_convert:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors='coerce')

        # Re-select numeric and categorical columns after conversion
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(exclude=[np.number]).columns
        
        # Debugging statements
        #print(f"Numeric columns: {numeric_columns}")
        #print(f"Categorical columns: {categorical_columns}")
        
        # For numeric columns, impute with median
        numeric_imputer = SimpleImputer(strategy=config.MISSING_VALUE_STRATEGY['numeric'])
        data[numeric_columns] = pd.DataFrame(numeric_imputer.fit_transform(data[numeric_columns]), 
                                            columns=numeric_columns, index=data.index)
        
        # For categorical columns, impute with most frequent value
        categorical_imputer = SimpleImputer(strategy=config.MISSING_VALUE_STRATEGY['categorical'])
        data[categorical_columns] = pd.DataFrame(categorical_imputer.fit_transform(data[categorical_columns]), 
                                                columns=categorical_columns, index=data.index)
        logger.info(f"Missing values handled. Remaining null values: {data.isnull().sum().sum()}")
        return data
    except Exception as e:
        logger.error(f"Error handling missing values: {str(e)}")
        raise DataError(f"Error in handling missing values: {str(e)}")

def standardize_values(value):
    if isinstance(value, str):
        return value.lower().strip().capitalize()
    return value

@error_handler
def merge_datasets(batch_data, fail_data):
    # Implement dataset merging logic
    merged_data = pd.merge(batch_data, fail_data, 
                           left_on=['Material', 'Batch'], 
                           right_on=['Product/Material', 'Batch Number'], 
                           how='left')
    
    merged_data['Failed'] = merged_data['Product/Material'].notna().astype(int)
    merged_data.drop(['Product/Material', 'Batch Number'], axis=1, inplace=True)
    
    return merged_data

@error_handler
def preprocess_data(merged_data):
    # Create new features
    merged_data['Yield_Efficiency'] = merged_data['G.R.Qty'] / merged_data['Theoretical Yield']
    merged_data['Waste_Percentage'] = merged_data['Waste in ML'] / merged_data['Total Input in ML'] * 100

    # Select features and target
    features = ['Theoretical Yield', 'G.R.Qty', 'QC Qty (ML)', 'Waste in ML', '% Waste loss', 
                'Yield_Efficiency', 'Waste_Percentage', 'Failed']
    target = 'Waste Total Cost'
    
    X = merged_data[features]
    y = merged_data[target]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Transform target variable
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    y_transformed = pt.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    return X_scaled, y_transformed, pt

@error_handler
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

@error_handler
def prepare_data(batch_file, fail_file):
    batch_data, fail_data = load_data(batch_file, fail_file)
    batch_data = handle_missing_values(batch_data)
    fail_data = handle_missing_values(fail_data)
    merged_data = merge_datasets(batch_data, fail_data)
    X, y, pt = preprocess_data(merged_data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, pt

if __name__ == "__main__":
    batch_data = pd.read_csv(config.DATA_PATH['batch_data'])
    fail_data = pd.read_csv(config.DATA_PATH['fail_data'])
    X_train, X_test, y_train, y_test, pt = prepare_data(batch_data, fail_data)
    print("Data preparation complete.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
