import pandas as pd

from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression

from src.data_preprocessing import split_dataset, build_column_transformer


def regression_training(df_path: str, target: str):
    MODELS = {
        'LinReg': LinearRegression(),
        'Ridge': Ridge(),
        'TreeReg': DecisionTreeRegressor(),
        'RF': RandomForestRegressor(),
        'XGB': XGBRegressor(),
        'CatBoost': CatBoostRegressor(),
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
        'CatBoost': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 6],
            'model__learning_rate': [0.01, 0.1],
            'model__verbose': [False]
        },
        'Ridge': {
            'model__alpha': [0.01, 0.1, 1.0, 10.0],
            'model__solver': ['auto', 'svd', 'cholesky', 'lsqr']
        },
        'TreeReg': {
            'model__max_depth': [None, 4, 6, 8],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    }

    df = pd.read_csv(df_path)

    X_train, X_test, y_train, y_test = split_dataset(df, target)
    preprocessor = build_column_transformer(df)

    for model_name, model_class in MODELS.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model_class),
        ])
        params = grid_params[model_name]

        grid = RandomizedSearchCV(estimator=pipeline, param_distributions=params, n_iter=10,
                                cv=5, scoring='mean_absolute_error', random_state=42)

        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)

        results.append({
            'model_name': model_name,
            'best_score': -grid.best_score_,
            'best_estimator': grid.best_estimator_,
            'y_test': y_test,
            'predictions': y_pred,
        })

    return results


def classification_training(df_path: str, target: str):
    MODELS = {
        'LinReg': LogisticRegression(),
        'TreeReg': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(),
        'XGB': XGBClassifier(),
        'CatBoost': CatBoostClassifier(),
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
        'CatBoost': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 6],
            'model__learning_rate': [0.01, 0.1],
            'model__verbose': [False]
        },
        'TreeReg': {
            'model__max_depth': [None, 4, 6, 8],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    }

    df = pd.read_csv(df_path)

    X_train, X_test, y_train, y_test = split_dataset(df, target)
    preprocessor = build_column_transformer(df)

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
            'best_score': grid.best_score_,
            'best_estimator': grid.best_estimator_,
            'y_test': y_test,
            'predictions': y_pred,
        })

    return results