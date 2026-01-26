from uuid import UUID

from enum import Enum

from pydantic import BaseModel

from datetime import datetime


class TaskType(str, Enum):
    classification = 'сlassification'
    regression = 'regression'


class Prediction(BaseModel):
    id: str
    user_id: str
    task_type: TaskType
    best_model: str
    target: str
    metric: float
    dataset_hash: str
    created_at: datetime

    class Config:
        from_attributes = True

    
class PredictionRequest(BaseModel):
    df_path: str
    target: str


class UserOut(BaseModel):
    id: UUID
    username: str
    created_at: datetime

    class Config:
        from_attributes = True