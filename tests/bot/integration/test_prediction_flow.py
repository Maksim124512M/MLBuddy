import pytest
from unittest.mock import AsyncMock

from aiogram import Bot, Dispatcher
from aiogram.types import Update, Message, Document

from bot.handlers import router
from core.config import settings


@pytest.mark.asyncio
async def test_prediction_flow():
    """
    Test the end-to-end prediction flow of the bot.
    """

    bot = Bot(token=settings.BOT_TOKEN)
    dp = Dispatcher(bot=bot)
    dp.include_router(router)

    csv_bytes = b'col1,col2,target\n1,2,0\n3,4,1'

    # Mock bot methods
    bot.session.make_request = AsyncMock(return_value={
        "ok": True,
        "result": {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "date": 1670000000,
            "text": "ok"
        }
    })

    bot.get_file = AsyncMock(return_value=type('File', (), {'file_path': 'fake_path'}))
    bot.download_file = AsyncMock(return_value=csv_bytes)

    # Create fake document
    fake_document = Document(
        file_id='FAKE_FILE_ID',
        file_unique_id='FAKE_UNIQUE_ID',
        file_name='test.csv',
        mime_type='text/csv',
        file_size=123
    )

    # Simulate /start command
    update1 = Update(
        update_id=1,
        message=Message(
            message_id=1,
            date=1670000000,
            chat={'id': 123, 'type': 'private'},
            text='/start',
            from_user={'id': 123, 'is_bot': False, 'first_name': 'Max'}
        )
    )

    await dp.feed_update(bot=bot, update=update1)

    # Simulate making a new prediction
    update2 = Update(
        update_id=1,
        message=Message(
            message_id=1,
            date=1670000000,
            chat={'id': 123, 'type': 'private'},
            text='Making new prediction',
            from_user={'id': 123, 'is_bot': False, 'first_name': 'Max'},
            document=fake_document,
        )
    )

    await dp.feed_update(bot=bot, update=update2)

    # Simulate uploading CSV file
    update_csv = Update(
        update_id=3,
        message=Message(
            message_id=3,
            date=1670000000,
            chat={'id': 123, 'type': 'private'},
            from_user={'id': 123, 'is_bot': False, 'first_name': 'Max'},
            document=fake_document
        )
    )
    await dp.feed_update(bot=bot, update=update_csv)

    # Simulate selecting task type and target
    update_task = Update(
        update_id=4,
        message=Message(
            message_id=4,
            date=1670000000,
            chat={'id': 123, 'type': 'private'},
            from_user={'id': 123, 'is_bot': False, 'first_name': 'Max'},
            text='regression'
        )
    )
    await dp.feed_update(bot=bot, update=update_task)

    # Simulate selecting target column
    update_target = Update(
        update_id=5,
        message=Message(
            message_id=5,
            date=1670000000,
            chat={'id': 123, 'type': 'private'},
            from_user={'id': 123, 'is_bot': False, 'first_name': 'Max'},
            text='target'
        )
    )
    await dp.feed_update(bot=bot, update=update_target)

    assert bot.session.make_request.called
