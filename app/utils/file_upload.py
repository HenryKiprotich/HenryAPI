import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

async def upload_file_to_storage(
    file: UploadFile, 
    directory: str, 
    filename: Optional[str] = None
) -> str:
    """
    Upload a file to local storage or cloud storage service.
    
    Args:
        file: The file to upload
        directory: Subdirectory to store the file in
        filename: Optional custom filename, if not provided will use UUID
        
    Returns:
        URL or path to the uploaded file
    """
    # Create directory if it doesn't exist
    upload_dir = Path(f"uploads/{directory}")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a unique filename if not provided
    if not filename:
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ""
        filename = f"{uuid.uuid4()}{file_extension}"
    
    # Full path to save the file
    file_path = upload_dir / filename
    
    # Save the file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Return the relative path/URL to the file
    return f"/uploads/{directory}/{filename}"