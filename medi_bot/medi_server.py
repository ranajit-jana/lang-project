from dotenv import load_dotenv
import os


load_dotenv()

from fastapi import FastAPI
from langserve import add_routes
from .medi_langchain import chain


app = FastAPI(
    title="Medical Chatbot API",
    description="mediacal chatbot API built with FastAPI and LangServe",
    version="1.0.0",
)


add_routes(app, chain , path="/medi_chat")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    



