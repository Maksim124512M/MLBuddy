from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

main_menu = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text='Make new prediction')],
    [KeyboardButton(text='View my stats')],
], resize_keyboard=True)

task_types_markup = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text='Regression')],
    [KeyboardButton(text='Classification')],
], resize_keyboard=True)