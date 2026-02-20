from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

def get_rag_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful AI assistant.

Use the provided context to answer the user's question.
If the answer is not in the context, say "I don't know".

Context:
{context}
"""),
        
        ("human", 
         """User Question:
{user_input}
""")
    ])

    return prompt