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
            
            # HuggingFace (only if working)
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
            
            # DeepSeek (only if enabled and has credits)
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
            
            # OpenAI (only if enabled and has valid key)
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
            
            # Ensure we have at least one working LLM
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
        self.explain_score_template = load_prompt_template("explain_score_with_llm.txt")
        self.nl_to_sql_template = load_prompt_template("nl_to_sql.txt")
        self.rag_prompt_template = load_prompt_template("rag_prompt.txt")
        self.sql_response_template = load_prompt_template("sql_response.txt")
        self.health_history_template = load_prompt_template("health_history_prompt.txt")
        self.diagnose_sql_template = load_prompt_template("diagnose_with_sql.txt")

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
            
            # Estimate tokens (this is a simple approximation)
            # In a production system, you'd use the tokenizer from the specific model
            prompt_tokens = len(prompt.split())
            response_tokens = len(response.split())
            
            # Get temperature from metadata or use default
            temperature = metadata.get('temperature', 0.7)
            
            session_id = uuid.UUID(metadata.get('session_id')) if isinstance(metadata.get('session_id'), str) else metadata.get('session_id')
            
            # Create new AIConversation object
            conversation = AIConversation(
                UserID=user_id,
                SessionID=session_id,
                Prompt=prompt,
                Response=response,
                Model=model,
                Temperature=temperature,
                TokensPrompt=prompt_tokens,
                TokensResponse=response_tokens,
                Metadata=metadata
            )
            
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            
            logger.info(f"Stored conversation with ID: {conversation.ID}, model: {model}")
            return conversation
            
        except Exception as e:
            logger.exception(f"Failed to store conversation: {e}")
            # We don't want to fail the main flow if storing conversations fails
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

    async def explain_score(self, db, question: str, score: str, llm_name: str = None,
                         user_id: Optional[uuid.UUID] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        llm_name = self._choose_default_llm_name(llm_name)
        prompt = self.explain_score_template.format(question=question, score=score)
        llm = self.get_llm(llm_name)
        try:
            resp = await self._invoke_llm(llm, prompt)
            response_content = await self._extract_content(resp)
            
            # Store conversation with context in metadata
            if metadata is None:
                metadata = {}
            metadata['context'] = f"Explaining score: {score}"
            
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
            logger.exception(f"Error in explain_score: {e}")
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
            
    async def chat_with_health_history(self, db, message: str, animal_id: Optional[int] = None, 
                                    llm_name: str = None, user_id: Optional[uuid.UUID] = None, 
                                    session_id: Optional[uuid.UUID] = None) -> Dict[str, Any]:
        """
        Enhanced chatbot function that retrieves animal health history when an animal ID is provided
        to give more context to the LLM response.
        
        Args:
            db: SQLAlchemy database session
            message: User's message/question
            animal_id: Optional animal ID to retrieve health history for
            llm_name: LLM to use
            user_id: User ID for tracking
            session_id: Session ID for conversation tracking
        
        Returns:
            Dictionary with response and conversation tracking info
        """
        from sqlalchemy import select, and_, desc
        from sqlalchemy.orm import selectinload
        from app.models.health_history_model import HealthHistory
        from app.models.animal_model import Animal
        
        llm_name = self._choose_default_llm_name(llm_name)
        session_uuid = session_id or uuid.uuid4()
        
        # Create metadata for tracking
        metadata = {
            "session_id": str(session_uuid),
            "conversation_type": "chatbot",
            "user_message": message
        }
        
        if animal_id:
            metadata["animal_id"] = animal_id
        
        # Get conversation history (last 5 messages)
        history = ""
        if session_uuid:
            from app.models.ai_model import AIConversation
            
            history_query = select(AIConversation).filter(
                AIConversation.SessionID == session_uuid
            ).order_by(AIConversation.CreatedAt.desc()).limit(5)
            
            history_result = await db.execute(history_query)
            conversations = history_result.scalars().all()
            
            if conversations:
                history = "\n".join([
                    f"User: {conv.Prompt}\nAssistant: {conv.Response}"
                    for conv in reversed(list(conversations))
                ])
        
        # If an animal ID is provided, get health history
        health_history = ""
        animal_name = "the animal"
        
        if animal_id:
            # Get animal details
            animal_query = (
                select(Animal)
                .options(
                    selectinload(Animal.animal_type),
                    selectinload(Animal.breed)
                )
                .where(Animal.ID == animal_id)
            )
            animal_result = await db.execute(animal_query)
            animal = animal_result.scalars().first()
            
            if animal:
                animal_name = animal.AnimalName
                
                # Get health history
                history_query = (
                    select(HealthHistory)
                    .options(
                        selectinload(HealthHistory.disease),
                        selectinload(HealthHistory.symptom),
                        selectinload(HealthHistory.drug),
                        selectinload(HealthHistory.drug_combination)
                    )
                    .where(HealthHistory.AnimalID == animal_id)
                    .order_by(desc(HealthHistory.Date))
                    .limit(5)
                )
                
                history_result = await db.execute(history_query)
                records = history_result.scalars().all()
                
                if records:
                    health_history = "Health History:\n"
                    for idx, record in enumerate(records):
                        health_history += f"Record {idx+1} (Date: {record.Date}):\n"
                        
                        if record.disease:
                            health_history += f"- Disease: {record.disease.Name}\n"
                            
                        if record.symptom:
                            health_history += f"- Symptom: {record.symptom.Name}\n"
                            
                        if record.drug:
                            health_history += f"- Drug: {record.drug.Name}\n"
                            
                        if record.drug_combination:
                            health_history += f"- Drug Combination: {record.drug_combination.Name}\n"
                            
                        if record.DiagnosisNotes:
                            health_history += f"- Notes: {record.DiagnosisNotes}\n"
                            
                        health_history += "\n"
                else:
                    health_history = f"No health history found for {animal_name} (ID: {animal_id})."
        
        # Build prompt with appropriate context
        if animal_id:
            prompt = self.health_history_template.format(
                health_history=health_history,
                conversation_history=history,
                animal_name=animal_name,
                animal_id=animal_id,
                question=message
            )
        else:
            # Use standard prompt if no animal ID
            prompt = self.ask_response_template.format(
                question=message
            )
        
        # Get response from LLM
        llm = self.get_llm(llm_name)
        try:
            resp = await self._invoke_llm(llm, prompt)
            response = await self._extract_content(resp)
            
            # Store conversation
            conversation = await self._store_conversation(
                db=db,
                user_id=user_id,
                prompt=message,  # Store original user message, not the full prompt
                response=response,
                model=llm_name,
                metadata=metadata
            )
            
            result = {
                "response": response,
                "session_id": str(session_uuid),
                "model_used": llm_name
            }
            
            if conversation:
                result["conversation_id"] = conversation.ID
                
            if animal_id:
                result["animal_id"] = animal_id
                result["animal_name"] = animal_name
                result["has_health_history"] = health_history != ""
                
            return result
        except Exception as e:
            logger.exception(f"Error in chat_with_health_history: {e}")
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

        # Log the generated SQL so you can inspect what the LLM produced
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
            result["conversation_id"] = final_convo.ID
            
        return result
        
    async def diagnose_with_health_history(self, db, animal_id: int, symptoms: str, question: str, llm_name: str = None,
                                        user_id: Optional[uuid.UUID] = None, session_id: Optional[uuid.UUID] = None):
        """
        Specialized function for animal diagnosis that includes health history data
        from the database to provide context for the LLM.
        
        Args:
            db: SQLAlchemy database session
            animal_id: ID of the animal being diagnosed
            symptoms: Symptoms described by the user
            question: The diagnosis question
            llm_name: Which LLM to use
            user_id: User ID for tracking
            session_id: Session ID for tracking
            
        Returns:
            dict: Contains diagnosis, related health history, and conversation tracking
        """
        from sqlalchemy import select, and_, desc
        from sqlalchemy.orm import selectinload
        from app.models.health_history_model import HealthHistory
        from app.models.animal_model import Animal
        from app.models.disease_model import Disease, Symptom
        from app.models.drug_model import Drug
        
        llm_name = self._choose_default_llm_name(llm_name)
        
        # Create metadata with session ID and animal ID
        metadata = {
            "session_id": str(session_id) if session_id else str(uuid.uuid4()),
            "conversation_type": "animal_diagnosis",
            "animal_id": animal_id,
            "symptoms": symptoms,
            "question": question
        }
        
        # Get animal details
        animal_query = (
            select(Animal)
            .options(
                selectinload(Animal.animal_type),
                selectinload(Animal.breed)
            )
            .where(Animal.ID == animal_id)
        )
        animal_result = await db.execute(animal_query)
        animal = animal_result.scalars().first()
        
        if not animal:
            error_msg = f"Animal with ID {animal_id} not found"
            await self._store_conversation(
                db=db,
                user_id=user_id,
                prompt=question,
                response=error_msg,
                model=llm_name,
                metadata=metadata
            )
            raise HTTPException(status_code=404, detail=error_msg)
            
        # Get health history for this animal
        history_query = (
            select(HealthHistory)
            .options(
                selectinload(HealthHistory.disease),
                selectinload(HealthHistory.symptom),
                selectinload(HealthHistory.drug),
                selectinload(HealthHistory.drug_combination)
            )
            .where(HealthHistory.AnimalID == animal_id)
            .order_by(desc(HealthHistory.Date))
            .limit(5)  # Get the 5 most recent health records
        )
        
        history_result = await db.execute(history_query)
        health_history = history_result.scalars().all()
        
        # Format health history as context for the LLM
        health_context = ""
        if health_history:
            health_context = "\n\nHealth History:\n"
            for idx, record in enumerate(health_history):
                health_context += f"Record {idx+1} (Date: {record.Date}):\n"
                
                if record.disease:
                    health_context += f"- Disease: {record.disease.Name}\n"
                    
                if record.symptom:
                    health_context += f"- Symptom: {record.symptom.Name}\n"
                    
                if record.drug:
                    health_context += f"- Drug: {record.drug.Name}\n"
                    
                if record.drug_combination:
                    health_context += f"- Drug Combination: {record.drug_combination.Name}\n"
                    
                if record.DiagnosisNotes:
                    health_context += f"- Notes: {record.DiagnosisNotes}\n"
                    
                health_context += "\n"
        
        # Build full prompt with animal details and health history
        animal_type = animal.animal_type.Type if animal.animal_type else "Unknown"
        animal_breed = animal.breed.Breed if animal.breed else "Unknown"
        
        prompt = f"""You are a veterinary AI assistant. Please provide a potential diagnosis for this animal based on the symptoms and health history.

Animal Details:
- Name: {animal.AnimalName}
- Type: {animal_type}
- Breed: {animal_breed}
- Sex: {animal.Sex}
- Age: {animal.DOB}

Current Symptoms:
{symptoms}

{health_context}

Question: {question}

Please analyze this case and provide:
1. Possible diagnoses (most likely first)
2. Reasoning for each potential diagnosis
3. Recommended next steps or tests
4. Treatment considerations based on health history
"""

        # Get diagnosis from LLM
        llm = self.get_llm(llm_name)
        try:
            resp = await self._invoke_llm(llm, prompt)
            diagnosis = await self._extract_content(resp)
            
            # Store conversation with all context
            conversation = await self._store_conversation(
                db=db,
                user_id=user_id,
                prompt=prompt,
                response=diagnosis,
                model=llm_name,
                metadata=metadata
            )
            
            result = {
                "diagnosis": diagnosis,
                "animal_details": {
                    "name": animal.AnimalName,
                    "type": animal_type,
                    "breed": animal_breed,
                    "sex": animal.Sex,
                    "age": str(animal.DOB)
                },
                "has_health_history": len(health_history) > 0
            }
            
            if conversation:
                result["conversation_id"] = conversation.ID
                result["session_id"] = metadata["session_id"]
                
            return result
        except Exception as e:
            logger.exception(f"Error in diagnose_with_health_history: {e}")
            # Store the error conversation
            await self._store_conversation(
                db=db,
                user_id=user_id,
                prompt=prompt,
                response=f"Error: {str(e)}",
                model=llm_name,
                metadata=metadata
            )
            raise
            
    async def diagnose_with_sql_query(self, db, animal_id: int, symptoms: str, llm_name: str = None,
                                   user_id: Optional[uuid.UUID] = None, session_id: Optional[uuid.UUID] = None):
        """
        Enhanced diagnosis function that uses direct SQL queries to gather comprehensive health history
        and uses it for diagnosis.
        
        Args:
            db: SQLAlchemy database session
            animal_id: ID of the animal to diagnose
            symptoms: Current symptoms
            llm_name: Which LLM to use
            user_id: User ID for tracking
            session_id: Session ID for conversation tracking
        
        Returns:
            dict: Diagnosis results with SQL context
        """
        from app.database.schema_loader import get_schema
        from app.engine.sql_executor import SQLExecutor
        
        llm_name = self._choose_default_llm_name(llm_name)
        session_uuid = session_id or uuid.uuid4()
        
        # Create metadata
        metadata = {
            "session_id": str(session_uuid),
            "conversation_type": "sql_diagnosis",
            "animal_id": animal_id,
            "symptoms": symptoms
        }
        
        # First, get animal details via SQL
        animal_sql = f"""
        SELECT a."ID", a."AnimalName", a."DOB", a."Sex", a."Color", a."Status",
               at."Type" as "AnimalType", b."Breed" as "BreedName"
        FROM "Animals" a
        LEFT JOIN "AnimalTypes" at ON a."AnimalTypeID" = at."ID"
        LEFT JOIN "Breeds" b ON a."BreedID" = b."ID"
        WHERE a."ID" = {animal_id}
        """
        
        try:
            executor = SQLExecutor()
            animal_result = await executor.execute_sql(animal_sql)
            
            if not animal_result["rows"]:
                error_msg = f"Animal with ID {animal_id} not found"
                await self._store_conversation(
                    db=db,
                    user_id=user_id,
                    prompt=f"Diagnose animal {animal_id}",
                    response=error_msg,
                    model=llm_name,
                    metadata=metadata
                )
                raise HTTPException(status_code=404, detail=error_msg)
            
            # Extract animal details from SQL result
            animal_row = animal_result["rows"][0]
            animal_data = dict(zip(animal_result["columns"], animal_row))
            
            # Now get health history with SQL
            history_sql = f"""
            SELECT hh."Date", hh."DiagnosisNotes",
                   d."Name" as "DiseaseName", d."Description" as "DiseaseDescription",
                   s."SymptomName" as "SymptomName", s."Description" as "SymptomDescription",
                   dr."DrugName" as "DrugName", dr."Dosage" as "DrugDosage",
                   dc."ComboName" as "CombinationName"
            FROM "HealthHistory" hh
            LEFT JOIN "Diseases" d ON hh."DiseaseID" = d."ID"
            LEFT JOIN "Symptoms" s ON hh."SymptomID" = s."ID"
            LEFT JOIN "Drugs" dr ON hh."DrugID" = dr."ID"
            LEFT JOIN "DrugCombination" dc ON hh."ComboID" = dc."ID"
            WHERE hh."AnimalID" = {animal_id}
            ORDER BY hh."Date" DESC
            LIMIT 10
            """
            
            history_result = await executor.execute_sql(history_sql)
            
            # Format SQL results for LLM
            sql_results = "Animal Information:\n"
            sql_results += f"ID: {animal_data.get('ID')}\n"
            sql_results += f"Name: {animal_data.get('AnimalName')}\n"
            sql_results += f"Type: {animal_data.get('AnimalType')}\n"
            sql_results += f"Breed: {animal_data.get('BreedName')}\n"
            sql_results += f"Sex: {animal_data.get('Sex')}\n"
            sql_results += f"DOB: {animal_data.get('DOB')}\n"
            sql_results += f"Status: {animal_data.get('Status')}\n\n"
            
            sql_results += "Health History:\n"
            if history_result["rows"]:
                for i, row in enumerate(history_result["rows"]):
                    history_data = dict(zip(history_result["columns"], row))
                    sql_results += f"Record {i+1} (Date: {history_data.get('Date')}):\n"
                    
                    if history_data.get('DiseaseName'):
                        sql_results += f"- Disease: {history_data.get('DiseaseName')}\n"
                        if history_data.get('DiseaseDescription'):
                            sql_results += f"  Description: {history_data.get('DiseaseDescription')}\n"
                    
                    if history_data.get('SymptomName'):
                        sql_results += f"- Symptom: {history_data.get('SymptomName')}\n"
                        if history_data.get('SymptomDescription'):
                            sql_results += f"  Description: {history_data.get('SymptomDescription')}\n"
                    
                    if history_data.get('DrugName'):
                        sql_results += f"- Drug: {history_data.get('DrugName')}\n"
                        if history_data.get('DrugDosage'):
                            sql_results += f"  Dosage: {history_data.get('DrugDosage')}\n"
                    
                    if history_data.get('CombinationName'):
                        sql_results += f"- Drug Combination: {history_data.get('CombinationName')}\n"
                    
                    if history_data.get('DiagnosisNotes'):
                        sql_results += f"- Notes: {history_data.get('DiagnosisNotes')}\n"
                    
                    sql_results += "\n"
            else:
                sql_results += "No health history records found for this animal.\n"
            
            # Use diagnose_sql_template to format prompt
            prompt = self.diagnose_sql_template.format(
                sql_results=sql_results,
                symptoms=symptoms,
                animal_name=animal_data.get('AnimalName'),
                animal_type=animal_data.get('AnimalType', 'Unknown'),
                animal_breed=animal_data.get('BreedName', 'Unknown'),
                animal_sex=animal_data.get('Sex', 'Unknown'),
                animal_age=animal_data.get('DOB', 'Unknown')
            )
            
            # Get diagnosis from LLM
            llm = self.get_llm(llm_name)
            resp = await self._invoke_llm(llm, prompt)
            diagnosis = await self._extract_content(resp)
            
            # Store conversation
            conversation = await self._store_conversation(
                db=db,
                user_id=user_id,
                prompt=prompt,
                response=diagnosis,
                model=llm_name,
                metadata=metadata
            )
            
            result = {
                "diagnosis": diagnosis,
                "animal_details": {
                    "name": animal_data.get('AnimalName'),
                    "type": animal_data.get('AnimalType'),
                    "breed": animal_data.get('BreedName'),
                    "sex": animal_data.get('Sex'),
                    "age": str(animal_data.get('DOB'))
                },
                "has_health_history": bool(history_result["rows"]),
                "sql_query_successful": True
            }
            
            if conversation:
                result["conversation_id"] = conversation.ID
                result["session_id"] = str(session_uuid)
                
            return result
            
        except HTTPException:
            # Re-raise HTTP exceptions (like 404 not found)
            raise
        except Exception as e:
            logger.exception(f"Error in diagnose_with_sql_query: {e}")
            
            # Try fallback to the standard diagnose_with_health_history
            logger.info("Falling back to standard diagnose_with_health_history function")
            try:
                return await self.diagnose_with_health_history(
                    db=db,
                    animal_id=animal_id,
                    symptoms=symptoms,
                    question="Please provide a diagnosis based on the symptoms and health history",
                    llm_name=llm_name,
                    user_id=user_id,
                    session_id=session_uuid
                )
            except Exception as fallback_error:
                logger.exception(f"Fallback also failed: {fallback_error}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate diagnosis using both SQL and ORM methods."
                )