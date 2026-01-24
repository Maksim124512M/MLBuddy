import os
import pandas as pd

from ml.src.data_preprocessing import split_dataset


def test_split_dataset():
    file_path = os.path.join('tests', 'data', 'Titanic-Dataset.csv')

    df = pd.read_csv(file_path)
    target = 'Survived'

    X_train, X_test, y_train, y_test = split_dataset(df, target)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
