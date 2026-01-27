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
) -> Prediction:
    """
    Create a new prediction record in the database.
    Args:
        db (Session): Database session.
        user_telegram_id (str): Telegram ID of the user.
        task_type (TaskType): Type of the task.
        best_model (str): Best model used for the prediction.
        target (str): Target variable for the prediction.
        metric (float): Metric value of the prediction.
        dataset_hash (str): Hash of the dataset used.
    Returns:
        Prediction: The created prediction record.
    """
    prediction = Prediction(
        user_telegram_id=user_telegram_id,
        task_type=task_type,
        best_model=best_model,
        target=target,
        metric=metric,
        dataset_hash=dataset_hash,
    )

    try:
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
    except Exception:
        db.rollback()
        raise

    return prediction


def get_user_prediction(db: Session, user_telegram_id: str) -> list[Prediction]:
    """
    Retrieve the latest predictions for a specific user.
    Args:
        db (Session): Database session.
        user_telegram_id (str): Telegram ID of the user.
    Returns:
        List[Prediction]: List of the latest prediction records for the user.
    """
    predictions = db.query(Prediction).filter(Prediction.user_telegram_id == user_telegram_id).order_by(Prediction.created_at.desc()).limit(10).all()

    return predictions