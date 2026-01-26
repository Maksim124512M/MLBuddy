import pandas as pd
import multiprocessing

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression

from ml.src.data_preprocessing import split_dataset, build_column_transformer

from concurrent.futures import ThreadPoolExecutor, as_completed


def train_single_model(
    model_name,
    model_class,
    params,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_class),
    ])

    X_tr = X_train
    y_tr = y_train

    X_test = pd.DataFrame(X_test, columns=X_train.columns)

    if len(X_train) > 50_000 and model_name in ('XGBoost', 'LightGBM'):
        idx = X_train.sample(50_000, random_state=42).index
        X_tr = X_train.loc[idx]
        y_tr = y_train.loc[idx]

    if model_name == 'LinearRegression':
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        return {
            'model_name': model_name,
            'best_score': mean_absolute_error(y_test, y_pred),
            'predictions': y_pred.tolist(),
        }

    grid = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=4,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=1,
    )

    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    return {
        'model_name': model_name,
        'best_score': float(-grid.best_score_),
        'predictions': y_pred.tolist(),
        'params': {k.replace('model__', ''): v for k, v in grid.best_params_.items()},
    }


def regression_training(task, df_path: str, target: str) -> list:
    """
    Train and evaluate regression models.

    Parameters:
        df_path (str): Path to the dataset CSV file.
        target (str): Target column name.

    Returns:
        list: List of dictionaries containing model results.
    """

    MODELS = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(solver='auto'),
        'DesicionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(),
        'XGBoost': XGBRegressor(n_jobs=1),
        'LightGBM': LGBMRegressor(n_jobs=1, force_col_wise=True),
    }

    grid_params = {
        'RandomForest': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
        },
        'XGBoost': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 6],
            'model__learning_rate': [0.01, 0.1],
        },
        'Ridge': {
            'model__alpha': [0.01, 0.1, 1.0, 10.0],
        },
        'DesicionTree': {
            'model__max_depth': [None, 4, 6, 8],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        },
        'LightGBM': {
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

    tasks = []
    results = []

    total = len(MODELS)
    max_workers = min(
        len(MODELS),
        max(1, multiprocessing.cpu_count() - 1)
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, (model_name, model_class) in enumerate(MODELS.items(), start=1):
            params = grid_params.get(model_name)

            task.update_state(
                state='PROGRESS',
                meta={
                    'current': i,
                    'total': total,
                    'model': model_name,
                    'step': 'training'
                }
            )

            tasks.append(
                executor.submit(
                    train_single_model,
                    model_name,
                    model_class,
                    params,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    preprocessor,
                )
            )

        for future in as_completed(tasks):
            results.append(future.result())

    return results