from core.celery_app import celery_app
from ml.src.regression import regression_training


@celery_app.task(bind=True)
def regression_task(self, df_path: str, target: str):
    self.update_state(state='PROGRESS', meta={'step': 'loading data'})

    results = regression_training(self, df_path, target)

    best = min(results, key=lambda x: x['best_score'])

    return {
        'status': 'done',
        'best_model': best,
        'all_results': results,
    }
