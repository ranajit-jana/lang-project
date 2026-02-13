from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

embedding = OllamaEmbeddings(model="llama3")
vectorstore = Chroma(collection_name="entertainment_docs", persist_directory="rag/chroma_db", embedding_function=embedding)

retriever = vectorstore.as_retriever()

llm_prompt = """
using the context and question give a answer
Context: {context}
Question: {question}
Answer:
"""


prompt = ChatPromptTemplate.from_messages([("system", llm_prompt), ("human", "{context} \n  {question}")])

llm = OllamaLLM(model="llama3", temperature=0.4)
import os

print("Current working directory:", os.getcwd())
print("DB absolute path:", os.path.abspath("./rag/chroma_db"))

print("Collection count:", vectorstore._collection.count())
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


