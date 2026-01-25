from fastapi import APIRouter

from ml.src.tasks.classification import classification_task

from celery.result import AsyncResult
from core.celery_app import celery_app

from api.db.schemas import PredictionRequest


router = APIRouter()

@router.post('/train/')
async def classification_predict(data: PredictionRequest):
    task = classification_task.delay(data.df_path, data.target)

    return {
        'task_id': task.id,
        'status': 'started',
    }


@router.get('/tasks/{task_id}')
def get_task_status(task_id: str):
    task = AsyncResult(task_id, app=celery_app)

    return {
        'state': task.state,
        'info': task.info
    }