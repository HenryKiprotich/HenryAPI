"""import logging
import logging.handlers
import os
from pathlib import Path

def setup_logging():
   
    #Set up logging configuration for the application.
    #This should be called only once at application startup.
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler with less verbose output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler for debug logs with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/app.log', 
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        'logs/error.log', 
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Set specific module loggers if needed
    sqlalchemy_logger = logging.getLogger('sqlalchemy')
    sqlalchemy_logger.setLevel(logging.WARNING)  # Reduce SQL noise
    
    fastapi_logger = logging.getLogger('fastapi')
    fastapi_logger.setLevel(logging.INFO)
    
    uvicorn_logger = logging.getLogger('uvicorn')
    uvicorn_logger.handlers = []  # Remove default handlers to avoid duplication
    uvicorn_logger.propagate = True  # Let root logger handle it
    
    # Return configured logger
    return logging.getLogger('app')"""