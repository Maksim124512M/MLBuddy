from pydantic import BaseModel

from datetime import datetime


class PredictionRequest(BaseModel):
    df_path: str
    target: str


class UserOut(BaseModel):
    id: str
    username: str
    created_at: datetime

    class Config:
        from_attributes = True