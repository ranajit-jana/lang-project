import streamlit as st
import requests


st.title("Medical information chatbot")
st.warning("This chatbot is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")

api_url = "http://localhost:8000/medi_chat/invoke"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



user_input = st.chat_input("Ask any health-related queries...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    payload = {
        "input": {
             "user_input": user_input
        }
    }
    
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        result = response.json()["output"]
        st.session_state.messages.append({"role": "assistant", "content": result})

        with st.chat_message("assistant"):
            st.markdown(result)
    else:
        st.error(response.text)
