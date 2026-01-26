import os

import pandas as pd
from sklearn.compose import ColumnTransformer

from ml.src.data_preprocessing import build_column_transformer


def test_build_column_transformer():
    file_path = os.path.join('tests', 'data', 'Titanic-Dataset.csv')

    df = pd.read_csv(file_path)

    preprocessor = build_column_transformer(df)

    assert isinstance(preprocessor, ColumnTransformer)
