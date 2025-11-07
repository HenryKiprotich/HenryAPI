# LangChain RAG pipeline
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from app.engine.llm import LLMManager
from fastapi import HTTPException
from app.engine.web_search import search_financial_trends

class RAGPipeline:
    def __init__(self):
        self.loader_map = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader
        }
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.llm_manager = LLMManager()
        self.prompt_template = PromptTemplate.from_file("app/prompts/rag_prompt.txt")

    def load_documents(self, file_path: str):
        file_extension = file_path.split(".")[-1]
        loader_class = self.loader_map.get(f".{file_extension}")
        if not loader_class:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        loader = loader_class(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def initialize_vector_store(self, documents):
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    async def query(self, question: str) -> str:
        if not self.vector_store:
            raise HTTPException(status_code=400, detail="Vector store not initialized")
        llm = self.llm_manager.llms[self.llm_manager.select_llm()]
        # Retrieve from vector store
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(),
            prompt=self.prompt_template
        )
        rag_response = await chain.ainvoke({"question": question, "chat_history": []})
        # Retrieve from web
        web_results = search_financial_trends(question)
        web_context = "\n\n".join(web_results)
        # Compose final answer using LLM with both contexts
        final_prompt = (
            f"Question: {question}\n\n"
            f"Context from documents:\n{rag_response['answer']}\n\n"
            f"Context from latest financial websites:\n{web_context}\n\n"
            "Based on the above, provide a concise, up-to-date answer."
        )
        final_answer = await llm.ainvoke(final_prompt)
        return final_answer["text"] if isinstance(final_answer, dict) and "text" in final_answer else str(final_answer)