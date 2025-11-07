"""Note: this function no longer enforces "must be SELECT" — SQLExecutor
is responsible for validating/ rejecting non-SELECT queries."""

import asyncpg
import logging
import traceback
from app.config.settings import settings

logging.basicConfig(level=logging.DEBUG)

class SQLExecutor:
    def __init__(self, db_url=None):
        try:
            self.db_url = db_url or settings.DATABASE_URL
            logging.debug(f"Initialized SQLExecutor with db_url: {self.db_url}")
        except Exception as e:
            logging.error(f"Error initializing SQLExecutor: {e}\n{traceback.format_exc()}")
            raise

    async def query(self, sql: str):
        """Execute a read-only SQL query and return the results asynchronously."""
        try:
            if not sql.strip().lower().startswith("select"):
                raise ValueError("Only SELECT queries are allowed.")
            conn = await asyncpg.connect(self.db_url)
            try:
                stmt = await conn.prepare(sql)
                rows = await stmt.fetch()
                columns = [a.name for a in stmt.get_attributes()]
                rows_list = [tuple(row) for row in rows]
            finally:
                await conn.close()
            logging.debug(f"Query executed successfully: {sql}")
            return {"columns": columns, "rows": rows_list}
        except Exception as e:
            logging.error(f"Error executing query: {sql}\n{e}\n{traceback.format_exc()}")
            raise

    async def execute_sql(self, sql: str):
        """
        Receives an already converted SQL SELECT query and executes it asynchronously.
        """
        try:
            return await self.query(sql)
        except Exception as e:
            logging.error(f"Error in execute_sql with sql: {sql}\n{e}\n{traceback.format_exc()}")
            raise