from core.celery_app import celery_app
from ml.src.classification import classification_training


@celery_app.task(bind=True)
def classification_task(self, df_path: str, target: str):
    self.update_state(state='PROGRESS', meta={'step': 'loading data'})

    results = classification_training(self, df_path, target)

    best = min(results, key=lambda x: x['best_score'])

    return {
        'status': 'done',
        'best_model': best,
        'all_results': results,
    }
