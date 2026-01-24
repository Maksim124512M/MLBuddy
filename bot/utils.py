import httpx
import asyncio
import pandas as pd
import bot.bot_messages

from pathlib import Path

from aiogram.types import Message

from core.config import settings


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


async def poll_training_status(message: Message, task_id: str, task_type: str):
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
                break

            elif data['state'] == 'FAILURE':
                await message.answer('❌ Training failed')
                break