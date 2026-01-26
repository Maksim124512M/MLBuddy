from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_dataset(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into train and test sets.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        target (str): Target column name.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """

    y = df[target]
    X = df.drop(columns=[target])

    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_column_transformer(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a column transformer for preprocessing.

    Parameters:
        df (pd.DataFrame): Input dataframe.

    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['string', 'category']).columns

    numeric_transformer = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, cat_cols),
        ]
    )

    return preprocessor
