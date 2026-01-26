WELCOME_MESSAGE = """👋 Welcome to MLBuddy!

I help you build machine learning models automatically.
No code. No setup. Just upload your dataset.

What I can do:
• Classification & Regression
• Automatic model comparison
• Best model selection
• Metrics & predictions"""

DATASET_UPLOADING_MESSAGE = """📂 Please upload your CSV dataset.

Requirements:
• The dataset must include a target column
• No empty headers

Waiting for your file ⏳"""

TASK_TYPE_SETTINGS = """✅ Dataset uploaded successfully!

Rows: {rows}
Columns: {columns}

Now choose the task type 👇"""

TARGET_SETTING = """🎯 Please enter the name of the target column.

Example:
price
is_fraud
churn"""

TARGET_NOT_FOUND = """❌ Column not found.

Please make sure the name is correct and try again.
"""

TRAINING_STARTED = """⚙️ Training models...

This may take a few minutes depending on the dataset size.
Please wait ⏳
"""

TRAINING_COMPLETED = """
✅ Prediction Result

Best model: {model_name}
{metric}: {best_score}

📌 Example predictions:
- Predicted: {predictions}
- Params: {params}

"""

USER_PROFILE = """
👤 Your profile

🆔 Telegram ID: {telegram_id}
👤 Username: {username}
📅 Joined: {created_at}
"""