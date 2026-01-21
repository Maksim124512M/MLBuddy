import pandas as pd

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression

from ml.src.data_preprocessing import split_dataset, build_column_transformer


def regression_training(df_path: str, target: str) -> list:
    """
    Train and evaluate regression models.

    Parameters:
        df_path (str): Path to the dataset CSV file.
        target (str): Target column name.

    Returns:
        list: List of dictionaries containing model results.
    """

    MODELS = {
        'LinReg': LinearRegression(),
        'Ridge': Ridge(solver='auto'),
        'TreeReg': DecisionTreeRegressor(),
        'RF': RandomForestRegressor(),
        'XGB': XGBRegressor(),
        'LGBM': LGBMRegressor(force_col_wise=True),
    }

    results = []

    grid_params = {
        'RF': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
        },
        'XGB': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 6],
            'model__learning_rate': [0.01, 0.1],
        },
        'Ridge': {
            'model__alpha': [0.01, 0.1, 1.0, 10.0],
        },
        'TreeReg': {
            'model__max_depth': [None, 4, 6, 8],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        },
        'LGBM': {
            'model__n_estimators': [100, 300, 500, 1000],
            'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
            'model__num_leaves': [20, 31, 50, 70, 100],
            'model__max_depth': [-1, 5, 10, 20, 30],
            'model__min_child_samples': [5, 10, 20, 50],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0],
            'model__reg_alpha': [0, 0.01, 0.1, 1],
            'model__reg_lambda': [0, 0.01, 0.1, 1]
        }
    }

    df = pd.read_csv(df_path)

    X_train, X_test, y_train, y_test = split_dataset(df, target)
    preprocessor = build_column_transformer(X_train)

    def _clean_params(params: dict) -> dict:
        return {k.replace('model__', ''): v for k, v in params.items()}

    for model_name, model_class in MODELS.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model_class),
        ])
            
        X_test = pd.DataFrame(X_test, columns=X_train.columns)

        if model_name == 'LinReg':
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)

            results.append({
                'model_name': model_name,
                'best_score': mean_absolute_error(y_test, y_pred),
                'predictions': y_pred.tolist(),
            })

            continue
        
        params = grid_params[model_name]

        grid = RandomizedSearchCV(estimator=pipeline, param_distributions=params, n_iter=8,
                                cv=5, scoring='neg_mean_absolute_error', random_state=42)

        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)

        results.append({
            'model_name': model_name,
            'best_score': float(-grid.best_score_),
            'predictions': y_pred.tolist(),
            'params': _clean_params(grid.best_params_),
        })
        
    return results


def classification_training(df_path: str, target: str) -> list:
    """
    Train and evaluate classification models.

    Parameters:
        df_path (str): Path to the dataset CSV file.
        target (str): Target column name.

    Returns:
        list: List of dictionaries containing model results.
    """

    MODELS = {
        'LogReg': LogisticRegression(),
        'TreeReg': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(),
        'XGB': XGBClassifier(),
        'LGBM': LGBMClassifier(force_col_wise=True),
    }

    results = []

    grid_params = {
        'LogReg': {
            'model__penalty': ['l1', 'l2'],
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__solver': ['liblinear', 'saga']
        },
        'RF': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
        },
        'XGB': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 6],
            'model__learning_rate': [0.01, 0.1],
        },
        'TreeReg': {
            'model__max_depth': [None, 4, 6, 8],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        },
        'LGBM': {
            'model__n_estimators': [100, 300, 500, 1000],
            'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
            'model__num_leaves': [20, 31, 50, 70, 100],
            'model__max_depth': [-1, 5, 10, 20, 30],
            'model__min_child_samples': [5, 10, 20, 50],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0],
            'model__reg_alpha': [0, 0.01, 0.1, 1],
            'model__reg_lambda': [0, 0.01, 0.1, 1]
        },
    }

    df = pd.read_csv(df_path)

    X_train, X_test, y_train, y_test = split_dataset(df, target)
    preprocessor = build_column_transformer(X_train)

    def _clean_params(params: dict) -> dict:
        return {k.replace('model__', ''): v for k, v in params.items()}

    for model_name, model_class in MODELS.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model_class),
        ])
        params = grid_params[model_name]

        grid = RandomizedSearchCV(estimator=pipeline, param_distributions=params, n_iter=10,
                                cv=5, scoring='f1', random_state=42)

        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)

        results.append({
            'model_name': model_name,
            'best_score': float(grid.best_score_),
            'predictions': y_pred.tolist(),
            'params': _clean_params(grid.best_params_),
        })

    return results