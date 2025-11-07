from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
#from app.config.logging_config import setup_logging

# Set up logging first
#logger = setup_logging()
#logger.info("Starting HenryAPI Backend Application")

from app.api.v1.endpoints.AskHenry import router as ask_router

app = FastAPI()

# Allow CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8081", "exp://192.168.0.1:19000"],  # Add your Expo/React Native URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# v1 API
app.include_router(ask_router, prefix="/api/v1/ask", tags=["ask"])

@app.get("/")
async def home():
    return {"message": "The API is running"}