from api.db.crud.users import get_or_create_user
from api.db.models import User


def test_create_user(db):
    telegram_id = '1234'
    username = 'User'

    user = get_or_create_user(db, tg_id=telegram_id, username=username)

    db_user = db.query(User).filter(User.telegram_id == user.telegram_id).first()

    assert user.id is not None
    assert user.username == 'User'
    assert db_user is not None
