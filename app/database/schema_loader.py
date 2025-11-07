# app/database/schema_loader.py
# This lets you pass real schema into your LLM prompt (nl_to_sql).

from typing import Dict, Any
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine
# changed import to import the settings instance (not the package/module)
from app.config.settings import settings

async def get_schema() -> Dict[str, Any]:
    """
    Use an AsyncEngine and conn.run_sync to perform SQLAlchemy inspection on a sync connection.
    This avoids the MissingGreenlet error when the project uses the asyncpg driver.
    """
    engine = create_async_engine(settings.SQLALCHEMY_DATABASE_URL, echo=False)
    async with engine.connect() as conn:
        def _inspect(sync_conn):
            inspector = inspect(sync_conn)
            schema = {}
            for table_name in inspector.get_table_names():
                cols = inspector.get_columns(table_name)
                schema[table_name] = [
                    {"name": c["name"], "type": str(c["type"])} for c in cols
                ]
            return schema

        schema = await conn.run_sync(_inspect)

    await engine.dispose()
    return schema
