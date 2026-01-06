# Entrypoint of the web api.
from fastapi import FastAPI
from src.api.predict_controller import predict_router

app = FastAPI()
app.include_router(predict_router)