from fastapi import Depends, HTTPException, status, Cookie
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import logging
from app.database.database import get_db
from app.services.auth_service import decode_access_token
from app.models.user_model import User

# Get a module-specific logger
logger = logging.getLogger(__name__)

async def get_current_user_from_cookie(token: str = Cookie(None)):
    if token is None:
        logger.warning("Authentication failed: No token cookie provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: token cookie missing"
        )
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if user_id is None:
            logger.warning("Authentication failed: Token missing user_id")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: user_id missing"
            )
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return {"user_id": int(user_id)}

async def get_current_user(
    db: AsyncSession = Depends(get_db), 
    current_user_dict: dict = Depends(get_current_user_from_cookie)
):
    """Get current user object from the database using the user ID from the token"""
    user_id = current_user_dict["user_id"]
    try:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalars().first()
        
        if user is None:
            logger.error(f"User not found in database: ID {user_id}")
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    except Exception as e:
        logger.exception(f"Error retrieving user {user_id} from database")
        raise HTTPException(status_code=500, detail="Internal server error while retrieving user data")