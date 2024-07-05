# Data paths
DATA_PATH = {
    'batch_data': 'data/data_batch.csv',
    'fail_data': 'data/data_fail.csv'
}

# Data preparation
MISSING_VALUE_STRATEGY = {
    'numeric': 'median',
    'categorical': 'most_frequent'
}

COLUMNS_TO_STANDARDIZE = [
    'SAP Fail Category / Description',
    'Sub Category - Fails',
    'Manufacturing Location',
    'Vessel',
    'Material'
]

# Feature engineering
POLYNOMIAL_DEGREE = 2
N_FEATURES_TO_SELECT = 20

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Stacking Regressor parameters
BASE_MODELS = [
    ('rf', {'n_estimators': 100, 'random_state': RANDOM_STATE}),
    ('gb', {'n_estimators': 100, 'random_state': RANDOM_STATE}),
    ('xgb', {'n_estimators': 100, 'random_state': RANDOM_STATE}),
    ('lgb', {'n_estimators': 100, 'random_state': RANDOM_STATE})
]

META_MODEL = {
    'hidden_layer_sizes': (100, 50),
    'max_iter': 1000,
    'random_state': RANDOM_STATE
}

# Outlier removal
Z_THRESHOLD = 3

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
