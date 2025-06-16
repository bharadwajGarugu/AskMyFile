from fastapi import FastAPI
from .api.routes import router

app = FastAPI(title="FaQBot API")

app.include_router(router, prefix="/api")

# ... your routes ...