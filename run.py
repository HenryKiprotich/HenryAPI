import uvicorn
import logging
from main import app

if __name__ == "__main__":
    # Disable uvicorn's default logger to avoid duplicate logs
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_config=None  # Disable uvicorn's logging config to use ours
    )
