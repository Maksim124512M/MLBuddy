from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

# Define the main menu keyboard
main_menu = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text='Make new prediction')],
    [KeyboardButton(text='View my stats')],
], resize_keyboard=True)

# Define the task types keyboard
task_types_markup = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text='Regression')],
    [KeyboardButton(text='Classification')],
], resize_keyboard=True)