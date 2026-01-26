import httpx
import asyncio
import hashlib
import pandas as pd

from pathlib import Path

from aiogram.types import Message

from core.config import settings

from api.db.db_config import SessionLocal
from api.db.crud.predictions import create_new_prediction


DATASET_STORAGE_DIR = Path('storage/datasets')

def save_dataset_as_csv(df: pd.DataFrame, user_id: str, dataset_id: str) -> str:
    '''
    Save the dataset as a CSV file.

    Parameters:
        df (pd.DataFrame): Dataframe to save.
        user_id (str): ID of the user.
        dataset_id (str): ID of the dataset.

    Returns:
        str: Path to the saved CSV file.
    '''

    DATASET_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f'{user_id}_{dataset_id}.csv'
    file_path = DATASET_STORAGE_DIR / file_name
    df.to_csv(file_path, index=False)

    return str(file_path)


def generate_dataset_hash(csv_path: str) -> str:
    hasher = hashlib.sha256()
    with open(csv_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


async def poll_training_status(message: Message, task_id: str, task_type: str, target: str, dataset_hash: str):
    async with httpx.AsyncClient() as client:
        sent_progress = False

        metric = 'MAE' if task_type == 'Regression' else 'F1'

        if task_type == 'Regression':
            metric = 'MAE'
        else:
            metric = 'F1'

        while True:
            await asyncio.sleep(5)

            response = await client.get(
                f'{settings.API_URL}/v2/{task_type}/tasks/{task_id}'
            )

            data = response.json()

            if data['state'] == 'PROGRESS':
                if not sent_progress:
                    sent_progress = True
                    await message.answer('⏳ Training models...')

            elif data['state'] == 'SUCCESS':
                result = data['info']
                best = result['best_model']

                await message.answer(
                    f'✅ Training completed\n\n'
                    f'🏆 Model: {best['model_name']}\n'
                    f'📉 {metric}: {best['best_score']:.4f}'
                )

                with SessionLocal() as db:
                    create_new_prediction(
                        db=db, 
                        user_telegram_id=message.from_user.id,
                        task_type=task_type, 
                        best_model=best['model_name'], 
                        target=target,
                        metric=best['best_score'],
                        dataset_hash=dataset_hash,
                    )
                break   

            elif data['state'] == 'FAILURE':
                await message.answer('❌ Training failed')
                break