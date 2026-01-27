from fastapi import APIRouter

from api.v2.classification import router as classification_router
from api.v2.regression import router as regression_router

router = APIRouter()

router.include_router(regression_router, prefix='/regression', tags=['regression'])
router.include_router(
    classification_router, prefix='/classification', tags=['classification']
)