import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(project_root, "data", "nutrition.pdf")
chroma_db_path = os.path.join(project_root, "chroma_db")

loader = PyPDFLoader(pdf_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

texts = text_splitter.split_documents(documents)
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")
embeddings = OllamaEmbeddings(model=MODEL_NAME)

db = Chroma.from_documents(texts, embeddings, persist_directory=chroma_db_path)
db.persist()

print("Document loaded and vector store created successfully.")


"""
[0 ---------------------- 1000]  (Chunk 1)

With chunk_overlap=200, the next chunk will look like:

[800 ---------------------- 1800]  (Chunk 2)

So characters 800–1000 are repeated in both chunks.

This overlap helps maintain context between chunks, 
which can be especially beneficial for tasks like question-answering or summarization, 
where understanding the relationship between different parts of the text is important.

"""