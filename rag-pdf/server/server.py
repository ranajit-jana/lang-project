from fastapi import FastAPI
from .prompt import get_rag_prompt
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
import os

app = FastAPI()

@app.post("/rag/invoke")
def rag_invoke(request: dict):
    user_input = request.get("input")

    print("USER INPUT:", user_input)  

    """Initialize and return the vector store (Chroma database)"""
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    vectorstore = Chroma(persist_directory=os.path.abspath("./rag-pdf/chroma_db"), embedding_function=embeddings)
    print("Current working directory:", os.getcwd())
    print("DB absolute path:", os.path.abspath("./rag-pdf/chroma_db"))
    print("Collection count:", vectorstore._collection.count())
    docs = vectorstore.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    print("CONTEXT:", context)

    # Step 3: Prepare prompt
    llm_prompt = get_rag_prompt()
    llm = OllamaLLM(model="llama3", temperature=0.4)
    chain = llm_prompt | llm

    # Step 4: Call LLM
    response = chain.invoke({
        "user_input": user_input,
        "context": context
    })

    return {"output": response}