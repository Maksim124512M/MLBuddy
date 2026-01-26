from celery.result import AsyncResult
from fastapi import APIRouter

from api.db.schemas import PredictionRequest
from core.celery_app import celery_app
from ml.src.tasks.regression import regression_task

router = APIRouter()


@router.post('/train/')
async def regression_predict(data: PredictionRequest) -> dict:
    """
    Initiates a regression training task.
    Args:
        data (PredictionRequest): The request body containing the dataframe path and target column.
    Returns:
        dict: A dictionary containing the task ID and status.
    """

    task = regression_task.delay(data.df_path, data.target)

    return {
        'task_id': task.id,
        'status': 'started',
    }


@router.get('/tasks/{task_id}')
def get_task_status(task_id: str) -> dict:
    """
    Retrieves the status of a regression training task.
    Args:
        task_id (str): The ID of the task to check.
    Returns:
        dict: A dictionary containing the state and info of the task.
    """

    task = AsyncResult(task_id, app=celery_app)

    return {'state': task.state, 'info': task.info}
