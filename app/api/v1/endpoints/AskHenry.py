import logging
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from app.engine.llm import LLMManager
from app.database.schema_loader import get_schema
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text

from app.models.ai_model import AIConversation
from app.schemas.ai_schema import AIConversation as AIConversationSchema
from app.schemas.ai_schema import FeedbackUpdate, ConversationHistory
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

@router.post("/ask")
async def ask_llm(
    question: str = Form(...),
    llm_name: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question to a configured LLM. Optional file uploads will be saved and passed to the LLM.
    Returns the LLM answer and a conversation_id (generated if not provided).
    """
    upload_paths: List[str] = []
    conv_id = conversation_id or str(uuid.uuid4())

    # save uploaded files to a temp directory
    try:
        if files:
            base_dir = Path(getattr(settings, "UPLOAD_DIR", "temp_uploads")) / conv_id
            base_dir.mkdir(parents=True, exist_ok=True)
            for upload in files:
                dest = base_dir / (upload.filename or f"file-{uuid.uuid4().hex}")
                with dest.open("wb") as out_file:
                    # UploadFile.file is a file-like object (sync), use shutil.copyfileobj
                    shutil.copyfileobj(upload.file, out_file)
                    upload_paths.append(str(dest))
                try:
                    upload.file.close()
                except Exception:
                    pass
    except Exception as e:
        logger.exception("Failed saving uploaded files")
        return JSONResponse(status_code=500, content={"error": "Failed to save uploaded files"})

    # ask the LLM via LLMManager
    try:
        manager = LLMManager()
        # prefer explicit llm_name if given, otherwise allow manager to pick default
        llm_callable = None
        if llm_name:
            # manager is expected to expose a way to get a specific LLM or to ask directly by name
            # try common variants and fall back to manager.ask
            if hasattr(manager, "get_llm"):
                llm_instance = manager.get_llm(llm_name)
                # assume instance exposes an 'ask' method
                llm_callable = getattr(llm_instance, "ask", None) or llm_instance
            elif hasattr(manager, "ask"):
                llm_callable = lambda **kwargs: manager.ask(llm_name=llm_name, **kwargs)
            else:
                raise AttributeError("LLMManager has no known method to select an LLM by name")
        else:
            # no llm_name provided: try manager.ask or default LLM
            if hasattr(manager, "ask"):
                llm_callable = lambda **kwargs: manager.ask(**kwargs)
            elif hasattr(manager, "get_default_llm"):
                default = manager.get_default_llm()
                llm_callable = getattr(default, "ask", None) or default
            else:
                raise AttributeError("LLMManager has no usable ask method")

        # prepare kwargs commonly used by LLM implementations
        ask_kwargs: Dict[str, Any] = {
            "prompt": question,
            "files": upload_paths,
            "conversation_id": conv_id,
        }

        result = llm_callable(**ask_kwargs) if callable(llm_callable) else llm_callable
        # await if coroutine
        if inspect.isawaitable(result):
            result = await result

        # normalize result if object with 'text' or 'answer' fields
        if isinstance(result, dict):
            answer = result.get("answer") or result.get("text") or result.get("response") or result
        else:
            answer = result

        return {"conversation_id": conv_id, "llm": llm_name, "answer": answer}
    except Exception as e:
        logger.exception("LLM ask failed")
        return JSONResponse(status_code=500, content={"error": "Failed to get answer from LLM"})
