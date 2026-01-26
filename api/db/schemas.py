from enum import Enum

from pydantic import BaseModel


class TaskType(str, Enum):
    classification = 'сlassification'
    regression = 'regression'

    
class PredictionRequest(BaseModel):
    df_path: str
    target: str