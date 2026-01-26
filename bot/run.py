import asyncio

from aiogram import Bot, Dispatcher

from bot.handlers import router
from core.config import settings

bot = Bot(token=settings.BOT_TOKEN)
dp = Dispatcher()


async def main():
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Bot disabled')
