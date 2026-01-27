from enum import Enum

from pydantic import BaseModel


class TaskType(str, Enum):
    classification = 'Classification'
    regression = 'Regression'


class PredictionRequest(BaseModel):
    df_path: str
    target: str
