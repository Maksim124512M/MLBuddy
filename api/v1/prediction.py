from fastapi import APIRouter
from pydantic import BaseModel

from ml.src.models_training import regression_training

router = APIRouter()

class PredictionRequest(BaseModel):
    df_path: str
    target: str

@router.post('/regression/')
async def regression_predict(data: PredictionRequest):
    results = regression_training(data.df_path, data.target)
    best_result = min(results, key=lambda x: x['best_score'])

    return {
        'model': best_result['model_name'].__class__.__name__,
        'mae': float(best_result['best_score']),
        'predictions_sample': best_result['predictions'][:5],
    }