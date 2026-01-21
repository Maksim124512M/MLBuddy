import httpx
import pandas as pd
import bot.bot_messages
import bot.keyboards

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext

from core.config import settings

from bot.utils import save_dataset_as_csv

router = Router()

class MakingPrediction(StatesGroup):
    """
    States for making a prediction.
    """

    csv_dataset_id = State()
    task_type = State()
    target = State()
    dataset_path = State()


@router.message(Command('start'))
async def start(message: Message) -> None:
    """
    Handle the /start command.

    Parameters:
        message (Message): Incoming message.

    Returns:
        None
    """

    await message.answer(bot.bot_messages.WELCOME_MESSAGE, reply_markup=bot.keyboards.main_menu)

@router.message(Command('prediction'))
@router.message(F.text == 'Make new prediction')
async def make_prediction_first_stage(message: Message, state: FSMContext) -> None:
    """
    Start the prediction process.

    Parameters:
        message (Message): Incoming message.
        state (FSMContext): FSM context.

    Returns:
        None
    """

    await state.set_state(MakingPrediction.csv_dataset_id)
    await message.answer(bot.bot_messages.DATASET_UPLOADING_MESSAGE)


@router.message(MakingPrediction.csv_dataset_id)
async def dataset_uploading(message: Message, state: FSMContext) -> None:
    """
    Handle dataset uploading.

    Parameters:
        message (Message): Incoming message.
        state (FSMContext): FSM context.

    Returns:
        None
    """

    # Check if the uploaded file is a CSV
    if message.document.mime_type == 'text/csv':
        file_id = message.document.file_id
        file = await message.bot.get_file(file_id)
        file_content = await message.bot.download_file(file.file_path)

        df = pd.read_csv(file_content)

        rows_count, columns_count = df.shape

        await message.answer(bot.bot_messages.TASK_TYPE_SETTINGS.format(rows=rows_count, columns=columns_count),
                            reply_markup=bot.keyboards.task_types_markup)

        await state.update_data(csv_dataset_id=file_id)
        await state.set_state(MakingPrediction.task_type)
    else:
        await message.answer('You dataset is not CSV.')


@router.message(MakingPrediction.task_type)
async def setting_task_type(message: Message, state: FSMContext) -> None:
    """
    Set the task type for prediction.

    Parameters:
        message (Message): Incoming message.
        state (FSMContext): FSM context.

    Returns:
        None
    """

    await state.update_data(task_type=message.text)
    await state.set_state(MakingPrediction.target)
    await message.answer(bot.bot_messages.TARGET_SETTING)


@router.message(MakingPrediction.target)
async def target_setting(message: Message, state: FSMContext) -> None:
    """
    Set the target column and start training.

    Parameters:
        message (Message): Incoming message.
        state (FSMContext): FSM context.

    Returns:
        None
    """

    data = await state.get_data()
    file_id = data['csv_dataset_id']
    file = await message.bot.get_file(file_id)
    file_content = await message.bot.download_file(file.file_path)

    df = pd.read_csv(file_content)
    columns = df.columns.to_list()

    # Check if target column exists
    if message.text in columns:
        await state.update_data(target=message.text)
        await state.set_state(MakingPrediction.dataset_path)

        file_path = save_dataset_as_csv(df, message.from_user.id, file_id)
        
        await message.answer(bot.bot_messages.TRAINING_STARTED)
        await state.update_data(dataset_path=file_path)

        # Call the regression endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{settings.API_URL}/v1/predictions/regression',
                json={'df_path': file_path, 'target': state.data['target']}
            )

        results = response.json()

        await message.answer(bot.bot_messages.TRAINING_COMPLETED.format(model_name=results['model'],
                            mae=results['mae'], predictions=results['predictions_sample']))

        await state.clear()
    else:
        await message.answer(bot.bot_messages.TARGET_NOT_FOUND)
