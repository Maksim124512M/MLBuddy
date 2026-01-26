from sqlalchemy.orm import Session

from api.db.models import Prediction
from api.db.schemas import TaskType


def create_new_prediction(
    db: Session,
    user_telegram_id: str,
    task_type: TaskType,
    best_model: str,
    target: str,
    metric: float,
    dataset_hash: str,
):
    prediction = Prediction(
        user_telegram_id=user_telegram_id,
        task_type=task_type,
        best_model=best_model,
        target=target,
    )

    try:
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
    except Exception:
        db.rollback()
        raise

    return prediction
