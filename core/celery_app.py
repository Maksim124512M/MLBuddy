import os

from celery import Celery

REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')

celery_app = Celery(
    'automl',
    broker=REDIS_URL,
    backend=f'{REDIS_URL}/1',
    include=[
        'ml.src.tasks.regression',
        'ml.src.tasks.classification',
    ],
)

celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,
    result_backend='redis://redis:6379/1',
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
)
