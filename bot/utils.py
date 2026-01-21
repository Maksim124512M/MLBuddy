import pandas as pd

from pathlib import Path

DATASET_STORAGE_DIR = Path('storage/datasets')

def save_dataset_as_csv(df: pd.DataFrame, user_id: str, dataset_id: str) -> str:
    """
    Save the dataset as a CSV file.

    Parameters:
        df (pd.DataFrame): Dataframe to save.
        user_id (str): ID of the user.
        dataset_id (str): ID of the dataset.

    Returns:
        str: Path to the saved CSV file.
    """

    DATASET_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f'{user_id}_{dataset_id}.csv'
    file_path = DATASET_STORAGE_DIR / file_name
    df.to_csv(file_path, index=False)

    return str(file_path)