import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load .env
load_dotenv()

# App layout & title
st.set_page_config(page_title="Chatbot with Memory", page_icon="🤖")
st.title("🤖 Chatbot with Memory")

# Initialize model once
@st.cache_resource
def load_chat_model():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation",
        max_new_tokens=500,
        provider="auto"
    )
    return ChatHuggingFace(llm=llm)

model = load_chat_model()

# Initialize session state memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Handle user input
if prompt := st.chat_input("Type your message here..."):
    # Display user query
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to history
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Invoke model with full history
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.messages)
            st.markdown(response.content)

    # Add AI response to history
    st.session_state.messages.append(AIMessage(content=response.content))
