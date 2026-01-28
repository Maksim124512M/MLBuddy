from celery import Celery

celery_app = Celery(
    'automl',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/1',
    include=['ml.src.tasks.regression', 'ml.src.tasks.classification'],
)

celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,
    result_backend='redis://redis:6379/1',
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
)
