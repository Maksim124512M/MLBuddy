from fastapi import APIRouter

from api.v1.prediction import router as prediction_router
from api.v1.users import router as users_router

router = APIRouter()

router.include_router(prediction_router, prefix='/predictions', tags=['prediction'])
router.include_router(users_router, prefix='/users', tags=['users'])