from fastapi import APIRouter
from pydantic import BaseModel

from ml.src.regression import regression_training

router = APIRouter()


class PredictionRequest(BaseModel):
    df_path: str
    target: str


@router.post('/regression/')
async def regression_predict(data: PredictionRequest):
    results = regression_training(data.df_path, data.target)
    best_result = min(results, key=lambda x: x['best_score'])

    print(results)
    print(best_result)

    return {
        'model_name': best_result['model_name'],
        'best_score': float(best_result['best_score']),
        'predictions': best_result['predictions'][:5],
        'params': best_result.get('params', {}),
    }
