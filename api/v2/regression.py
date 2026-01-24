from fastapi import APIRouter
from pydantic import BaseModel

from ml.src.tasks.regression import regression_task

from celery.result import AsyncResult

from core.celery_app import celery_app


router = APIRouter()

class PredictionRequest(BaseModel):
    df_path: str
    target: str


@router.post('/train/')
async def regression_predict(data: PredictionRequest):
    task = regression_task.delay(data.df_path, data.target)

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