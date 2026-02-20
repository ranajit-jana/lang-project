import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


def get_vector_store():
    """Initialize and return the vector store (Chroma database)"""
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    vectorstore = Chroma(persist_directory=os.path.abspath("./rag-pdf/chroma_db"), embedding_function=embeddings)
    print("Current working directory:", os.getcwd())
    print("DB absolute path:", os.path.abspath("./rag-pdf/chroma_db"))
    print("Collection count:", vectorstore._collection.count())
    docs = vectorstore.similarity_search("What is nutrition?", k=3)
    for doc in docs:
        print(doc.page_content[:200])
        print("-------")
    return vectorstore