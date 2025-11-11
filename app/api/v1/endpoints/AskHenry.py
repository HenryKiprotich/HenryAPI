import logging
import shutil
from pathlib import Path
import uuid
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, UploadFile, File, Request, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.engine.llm import LLMManager
from app.database.database import get_db
from app.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def diagnose_root():
    return {"status": "ok", "message": "Henry API is running"}


@router.get("/llms")
async def list_llms():
    try:
        manager = LLMManager()
        llms = manager.get_available_llms()
        return {"llms": llms}
    except Exception as e:
        logger.exception(e)
        return JSONResponse(status_code=500, content={"error": "Failed to list LLMs"})


@router.post("/ask/")
async def ask_llm(
    request: Request,
    files: Optional[List[UploadFile]] = None,  # <- changed: remove File(...) so JSON requests work reliably
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question to a configured LLM. Supports JSON or Form data.
    Optional file uploads will be saved and passed to the LLM.
    Returns the LLM answer, conversation_id, and preserves contact/context info.
    """
    upload_paths: List[str] = []
    json_data: Dict[str, Any] = {}

    # Try JSON first, fall back to form parsing
    try:
        json_data = await request.json()
    except Exception:
        try:
            form = await request.form()
            # form can contain UploadFile objects too; convert to dict
            json_data = {k: v for k, v in form.items()}
        except Exception:
            json_data = {}

    # Extract relevant fields
    question = (json_data.get("message") or json_data.get("question")) if isinstance(json_data, dict) else None
    llm_name = json_data.get("llm_name") if isinstance(json_data, dict) else None
    conversation_id = (json_data.get("conversation_id") or str(uuid.uuid4())) if isinstance(json_data, dict) else str(uuid.uuid4())
    contact = json_data.get("contact") if isinstance(json_data, dict) else None
    context_message_id = json_data.get("context_message_id") if isinstance(json_data, dict) else None

    if not question:
        return JSONResponse(status_code=422, content={"error": "Missing 'message' or 'question' field"})

    # Save uploaded files (if any)
    try:
        if files:
            base_dir = Path(getattr(settings, "UPLOAD_DIR", "temp_uploads")) / conversation_id
            base_dir.mkdir(parents=True, exist_ok=True)
            for upload in files:
                if getattr(upload, "filename", None):
                    dest = base_dir / (upload.filename or f"file-{uuid.uuid4().hex}")
                    with dest.open("wb") as out_file:
                        shutil.copyfileobj(upload.file, out_file)
                        upload_paths.append(str(dest))
                    try:
                        upload.file.close()
                    except Exception:
                        pass
    except Exception:
        logger.exception("Failed saving uploaded files")
        return JSONResponse(status_code=500, content={"error": "Failed to save uploaded files"})

    # Ask the LLM using the public manager.query (handles prompt formatting & storage)
    try:
        manager = LLMManager()

        # Set default to Gemini 2.5 Flash Preview
        default_llm = "gemini-2.5-flash-preview"
        chosen_llm_name = llm_name if llm_name else default_llm

        # Run with a timeout to avoid indefinitely hanging requests
        import asyncio
        try:
            answer = await asyncio.wait_for(
                manager.query(db=db, question=question, llm_name=chosen_llm_name, session_id=conversation_id),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.exception("LLM request timed out")
            return JSONResponse(status_code=504, content={"error": "LLM request timed out"})

        return {
            "conversation_id": conversation_id,
            "llm": chosen_llm_name,
            "answer": answer,
            "contact": contact,
            "context_message_id": context_message_id,
            "uploaded_files": upload_paths
        }

    except Exception:
        logger.exception("LLM ask failed")
        return JSONResponse(status_code=500, content={"error": "Failed to get answer from LLM"})
