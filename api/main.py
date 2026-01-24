from fastapi import FastAPI

from api.v1.router import router as v1_router
from api.v2.router import router as v2_router

app = FastAPI()

app.include_router(v1_router, prefix='/v1')
app.include_router(v2_router, prefix='/v2')