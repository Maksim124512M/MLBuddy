from sqlalchemy.orm import Session

from api.db.models import User


def get_or_create_user(db: Session, tg_id: str, username: str):
    user = db.query(User).filter(User.telegram_id==tg_id, User.username==username).first()

    if user:
        return user

    user = User(
        telegram_id=tg_id,
        username=username,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def get_user_profile(db: Session, tg_id: str):
    user = db.query(User).filter(User.telegram_id==tg_id).first()

    return user