import logging
from langchain.prompts import PromptTemplate
try:
    from langchain_huggingface import HuggingFaceEndpoint
except ImportError:
    from langchain_community.llms import HuggingFaceHub as HuggingFaceEndpoint
from app.config.settings import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import HTTPException
from typing import Optional, Dict, Any
import os
import httpx
import asyncio
import uuid
from datetime import datetime

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from app.engine.sql_executor import SQLExecutor
from app.models.ai_model import AIConversation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_prompt_template(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

class DeepSeekLLM:
    """Minimal async DeepSeek API wrapper for chat/completions."""
    def __init__(self, api_key: str, api_base: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model

    async def ainvoke(self, prompt: str):
        url = f"{self.api_base}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return type("Obj", (), {"content": data["choices"][0]["message"]["content"]})

class LLMManager:
    def __init__(self):
        try:
            self.llms = {}
            
            # Google Gemini models (working and free)
            if settings.google_api_key:
                try:
                    # Primary model - fast and reliable
                    self.llms["gemini-2.5-flash"] = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        google_api_key=settings.google_api_key,
                        temperature=0.7,
                        max_output_tokens=2048
                    )
                    logger.info("SUCCESS: Successfully initialized Gemini 2.5 Flash")
                except Exception as e:
                    logger.warning(f"FAILED: Failed to initialize Gemini 2.5 Flash: {e}")
                
                try:
                    # Preview model with more features
                    self.llms["gemini-2.5-flash-preview"] = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash-preview-05-20",
                        google_api_key=settings.google_api_key,
                        temperature=0.7,
                        max_output_tokens=2048
                    )
                    logger.info("SUCCESS: Successfully initialized Gemini 2.5 Flash Preview")
                except Exception as e:
                    logger.warning(f"FAILED: Failed to initialize Gemini 2.5 Flash Preview: {e}")
                
                try:
                    # Lite model for faster responses
                    self.llms["gemini-2.5-flash-lite"] = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash-lite-preview-06-17",
                        google_api_key=settings.google_api_key,
                        temperature=0.7,
                        max_output_tokens=2048
                    )
                    logger.info("SUCCESS: Successfully initialized Gemini 2.5 Flash Lite")
                except Exception as e:
                    logger.warning(f"FAILED: Failed to initialize Gemini 2.5 Flash Lite: {e}")
            
            # HuggingFace 
            if settings.huggingfacehub_api_token:
                try:
                    self.llms["Llama-3.1-8B-Instruct"] = HuggingFaceEndpoint(
                        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                        huggingfacehub_api_token=settings.huggingfacehub_api_token,
                        temperature=0.7,
                        max_new_tokens=512
                    )
                    logger.info("SUCCESS: Successfully initialized Llama 3.1 8B")
                except Exception as e:
                    logger.warning(f"FAILED: Failed to initialize Llama: {e}")
            
            # DeepSeek 
            if (settings.deepseek_api_key and 
                hasattr(settings, 'deepseek_enabled') and 
                getattr(settings, 'deepseek_enabled', False)):
                try:
                    self.llms["DeepSeek-R1-0528"] = DeepSeekLLM(
                        api_key=settings.deepseek_api_key,
                        api_base=settings.deepseek_api_base,
                        model="deepseek-chat"
                    )
                    logger.info("SUCCESS: Successfully initialized DeepSeek")
                except Exception as e:
                    logger.warning(f"FAILED: Failed to initialize DeepSeek: {e}")
            
            # OpenAI 
            if (ChatOpenAI and settings.openai_api_key and 
                len(settings.openai_api_key) > 20 and 
                hasattr(settings, 'openai_enabled') and 
                getattr(settings, 'openai_enabled', False)):
                try:
                    self.llms["GPT-3.5-Turbo"] = ChatOpenAI(
                        api_key=settings.openai_api_key,
                        model="gpt-3.5-turbo",
                        temperature=0.7,
                        max_tokens=1024
                    )
                    logger.info("SUCCESS: Successfully initialized OpenAI GPT-3.5")
                except Exception as e:
                    logger.warning(f"FAILED: Failed to initialize OpenAI: {e}")
            
            # Ensure there is at least one working LLM
            if not self.llms:
                logger.error("ERROR: No LLMs could be initialized!")
                raise HTTPException(status_code=500, detail="No LLMs available")
            else:
                logger.info(f"SUCCESS: Successfully initialized {len(self.llms)} LLM(s): {list(self.llms.keys())}")
        except Exception as e:
            logger.error(f"Failed to initialize LLMs: {str(e)}")
            raise

        # Load prompt templates
        self.ask_response_template = load_prompt_template("ask_response.txt")
        self.nl_to_sql_template = load_prompt_template("nl_to_sql.txt")
        self.rag_prompt_template = load_prompt_template("rag_prompt.txt")
        
    def get_available_llms(self):
        return list(self.llms.keys())

    def get_llm(self, llm_name: str):
        llm = self.llms.get(llm_name)
        if not llm:
            raise HTTPException(status_code=400, detail=f"LLM '{llm_name}' not available")
        return llm

    def _choose_default_llm_name(self, llm_name: Optional[str]) -> str:
        # If specific LLM name provided, use that
        if llm_name and llm_name in self.llms:
            return llm_name
        # Otherwise prefer working Gemini models, then fallback to others
        preference_order = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-preview", 
            "gemini-2.5-flash-lite",
            "Llama-3.1-8B-Instruct",
            "GPT-3.5-Turbo",
            "DeepSeek-R1-0528"
        ]
        
        for preferred in preference_order:
            if preferred in self.llms:
                return preferred
                
        # If none of the preferred models available, use first available
        keys = list(self.llms.keys())
        if not keys:
            raise HTTPException(status_code=500, detail="No LLMs configured")
        return keys[0]

    async def _invoke_llm(self, llm, prompt: str):
        """
        Normalize invocation across different LLM wrappers:
         - async .ainvoke(prompt) returning object with .content or .text
         - async .agenerate / .generate patterns (LangChain-style)
         - callable sync objects run in threadpool
        """
        # prefer async ainvoke
        if hasattr(llm, "ainvoke"):
            try:
                return await llm.ainvoke(prompt)
            except TypeError:
                # some wrappers expect different signature
                return await llm.ainvoke({"input": prompt})
        # langchain-style async generate/agenerate
        if hasattr(llm, "agenerate"):
            return await llm.agenerate([prompt])
        if hasattr(llm, "generate"):
            # some langchain models expose async generate
            gen = llm.generate([prompt]) if not asyncio.iscoroutinefunction(llm.generate) else await llm.generate([prompt])
            return gen
        # sync callable: run in threadpool
        if callable(llm):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, llm, prompt)
        raise RuntimeError("Unsupported LLM interface")

    async def _extract_content(self, resp) -> str:
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp
        # object with content or text
        if hasattr(resp, "content"):
            c = getattr(resp, "content")
            return c.strip() if isinstance(c, str) else str(c)
        if hasattr(resp, "text"):
            t = getattr(resp, "text")
            return t.strip() if isinstance(t, str) else str(t)
        # LangChain generate output: try to dig into generations
        if hasattr(resp, "generations"):
            try:
                gens = resp.generations
                # gens may be List[List[Generation]]
                first = gens[0][0]
                if hasattr(first, "text"):
                    return first.text.strip()
            except Exception:
                pass
        # dict-like responses
        if isinstance(resp, dict):
            if "content" in resp:
                return str(resp["content"]).strip()
            if "text" in resp:
                return str(resp["text"]).strip()
        # fallback to str
        return str(resp).strip()
    
    async def _store_conversation(self, db, user_id: Optional[uuid.UUID], prompt: str, response: str, 
                               model: str, metadata: Optional[Dict[str, Any]] = None):
        """Store conversation in AIConversations table using SQLAlchemy session"""
        try:
            # Generate a session ID if not already in metadata
            if not metadata:
                metadata = {}
            
            if 'session_id' not in metadata:
                metadata['session_id'] = str(uuid.uuid4())
            
            # Estimate tokens 
            # use tokenizer from the specific model in future for better accuracy
            prompt_tokens = len(prompt.split())
            response_tokens = len(response.split())
            
            # Get temperature from metadata or use default
            temperature = metadata.get('temperature', 0.7)
            
            session_id = uuid.UUID(metadata.get('session_id')) if isinstance(metadata.get('session_id'), str) else metadata.get('session_id')
            
            # Create new AIConversation object
            conversation = AIConversation(
                user_id=user_id,
                session_id=session_id,
                prompt=prompt,
                response=response,
                model=model,
                temperature=temperature,
                tokens_prompt=prompt_tokens,
                tokens_response=response_tokens,
                metadata=metadata
            )
            
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            
            logger.info(f"Stored conversation with ID: {conversation.id}, model: {model}")
            return conversation
            
        except Exception as e:
            logger.exception(f"Failed to store conversation: {e}")
            # Don't fail the main flow if storing conversations fails
            # so just log the error but don't re-raise
            return None

    async def query(self, db, question: str, llm_name: str = None, 
                   user_id: Optional[uuid.UUID] = None, session_id: Optional[uuid.UUID] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        llm_name = self._choose_default_llm_name(llm_name)
        prompt = self.ask_response_template.format(question=question)
        llm = self.get_llm(llm_name)
        
        # Ensure metadata exists and include session_id if provided
        if not metadata:
            metadata = {}
        if session_id:
            metadata['session_id'] = str(session_id)
            
        try:
            resp = await self._invoke_llm(llm, prompt)
            response_content = await self._extract_content(resp)
            
            # Store conversation
            await self._store_conversation(
                db=db,
                user_id=user_id,
                prompt=question,
                response=response_content,
                model=llm_name,
                metadata=metadata
            )
            
            return response_content
        except Exception as e:
            logger.exception(f"Error in query: {e}")
            raise    

    async def nl_to_sql(self, db, schema: str, question: str, llm_name: str = None,
                      user_id: Optional[uuid.UUID] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        llm_name = self._choose_default_llm_name(llm_name)
        prompt = self.nl_to_sql_template.format(schema=schema, question=question)
        llm = self.get_llm(llm_name)
        try:
            resp = await self._invoke_llm(llm, prompt)
            response_content = await self._extract_content(resp)
            
            # Store conversation with type in metadata
            if metadata is None:
                metadata = {}
            metadata['conversation_type'] = 'nl_to_sql'
            metadata['question'] = question
            
            await self._store_conversation(
                db=db,
                user_id=user_id,
                prompt=prompt,
                response=response_content,
                model=llm_name,
                metadata=metadata
            )
            
            return response_content
        except Exception as e:
            logger.exception(f"Error in nl_to_sql: {e}")
            raise

    async def rag_prompt(self, db, context: str, question: str, llm_name: str = None,
                       user_id: Optional[uuid.UUID] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        llm_name = self._choose_default_llm_name(llm_name)
        prompt = self.rag_prompt_template.format(context=context, question=question)
        llm = self.get_llm(llm_name)
        try:
            resp = await self._invoke_llm(llm, prompt)
            response_content = await self._extract_content(resp)
            
            # Store conversation with type in metadata
            if metadata is None:
                metadata = {}
            metadata['conversation_type'] = 'rag'
            metadata['question'] = question
            
            await self._store_conversation(
                db=db,
                user_id=user_id,
                prompt=prompt,
                response=response_content,
                model=llm_name,
                metadata=metadata
            )
            
            return response_content
        except Exception as e:
            logger.exception(f"Error in rag_prompt: {e}")
            raise

    async def sql_response(self, db, question: str, table: str, llm_name: str = None,
                         user_id: Optional[uuid.UUID] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        llm_name = self._choose_default_llm_name(llm_name)
        prompt = self.sql_response_template.format(question=question, table=table)
        llm = self.get_llm(llm_name)
        try:
            resp = await self._invoke_llm(llm, prompt)
            response_content = await self._extract_content(resp)
            
            # Store conversation with type in metadata
            if metadata is None:
                metadata = {}
            metadata['conversation_type'] = 'sql_response'
            metadata['question'] = question
            
            await self._store_conversation(
                db=db,
                user_id=user_id,
                prompt=prompt,
                response=response_content,
                model=llm_name,
                metadata=metadata
            )
            
            return response_content
        except Exception as e:
            logger.exception(f"Error in sql_response: {e}")
            raise
            
    
   
    async def nl_to_sql_and_execute(self, db, schema: str, question: str, llm_name: str = None,
                                  user_id: Optional[uuid.UUID] = None, session_id: Optional[uuid.UUID] = None):
        """
        Convenience orchestration: generate SQL via LLM, execute via SQLExecutor,
        and return a formatted LLM response.        
        """
        llm_name = self._choose_default_llm_name(llm_name)
        
        # Create metadata with session ID
        metadata = {
            "session_id": str(session_id) if session_id else str(uuid.uuid4()),
            "conversation_type": "nl_to_sql_and_execute",
            "question": question
        }
        
        # Generate SQL
        sql = await self.nl_to_sql(
            db=db,
            schema=schema, 
            question=question, 
            llm_name=llm_name,
            user_id=user_id,
            metadata=metadata
        )

        # Log the generated SQL so WE can inspect what the LLM produced
        logger.info(f"Generated SQL from LLM ({llm_name}):\n{sql}")
        
        # Add SQL to metadata for the final response
        metadata["generated_sql"] = sql

        executor = SQLExecutor()
        try:
            raw = await executor.execute_sql(sql)
        except ValueError as e:
            # SQLExecutor uses ValueError for validation failures (e.g. non-SELECT)
            logger.warning(f"SQL validation failed: {e}\nGenerated SQL:\n{sql}")
            metadata["error"] = str(e)
            
            # Even though it failed, store the error response
            error_convo = await self._store_conversation(
                db=db,
                user_id=user_id,
                prompt=question,
                response=f"Error: {str(e)}",
                model=llm_name,
                metadata=metadata
            )
            
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception(f"SQL execution failed: {e}\nGenerated SQL:\n{sql}")
            metadata["error"] = str(e)
            
            # Store the error
            error_convo = await self._store_conversation(
                db=db,
                user_id=user_id,
                prompt=question,
                response=f"Error: {str(e)}",
                model=llm_name,
                metadata=metadata
            )
            
            raise HTTPException(status_code=500, detail="Failed to execute generated SQL.")

        # format table text for the sql_response prompt (simple string representation)
        table_text = {"columns": raw.get("columns"), "rows": raw.get("rows")}
        
        # Add raw results to metadata
        metadata["raw_result"] = {
            "columns": raw.get("columns"),
            "row_count": len(raw.get("rows", []))
        }
        
        # Get formatted response
        formatted = await self.sql_response(
            db=db,
            question=question, 
            table=str(table_text), 
            llm_name=llm_name,
            user_id=user_id,
            metadata=metadata
        )
        
        # Store the final answer with all context
        final_convo = await self._store_conversation(
            db=db,
            user_id=user_id,
            prompt=question,
            response=formatted,
            model=llm_name,
            metadata=metadata
        )
        
        result = {"sql": sql, "raw": raw, "formatted": formatted}
        if final_convo:
            result["conversation_id"] = final_convo.id
            
        return result
        

            
   