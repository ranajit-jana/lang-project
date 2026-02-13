from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os



text_loader = TextLoader("../data/sample.txt")
documents = text_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
MODEL_NAME= os.getenv("MODEL_NAME", "llama3")
embeddings = OllamaEmbeddings(model=MODEL_NAME)
db = Chroma.from_documents(texts, embeddings, collection_name="entertainment_docs", persist_directory="../chroma_db")
db.persist()
print("Documents loaded and indexed successfully.")
c