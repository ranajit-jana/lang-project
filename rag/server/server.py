from dotenv import load_dotenv
import os


load_dotenv()

from fastapi import FastAPI
from langserve import add_routes
from .prompt import chain


app = FastAPI(
    title="RAG Chatbot API",
    description="RAG API built with FastAPI and LangServe",
    version="1.0.0",
)


add_routes(app, chain , path="/rag")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    



