from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


llm_prompt = """

You are medical assistant. 
Your task is to analyze the conversation and provide insights, suggestions, and recommendations based on the information provided.

RULES:
1. you can do diagnosis but dont prescibe any medication or dosages to the patient.

"""


general = """

2. You will not prescribe any medication and dosages to the patient.
3. You will can only provide general advice and recommendations based on the symptoms described by the patient.
4. You will always recommend the patient to consult with a healthcare professional for a proper diagnosis and treatment plan.
5. you can give insights if diagnostic test or results are mentioned in the conversation but you dont interpret symtoms which is not in the test results.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", llm_prompt),
        ("human", "{user_input}"),
    ]
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
chain = prompt | llm


