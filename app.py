import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Page Configuration
st.set_page_config(page_title="Local AI Chatbot", page_icon="💬", layout="centered")
st.title("💬 Local LangChain Chatbot")
st.caption("Powered by Llama 3.2:3b running locally via Ollama")

# 2. Initialize the Model (Cached so it doesn't reload on every click)
@st.cache_resource
def load_model():
    return ChatOllama(model="llama3.2:3b", temperature=0.7)

model = load_model()

# 3. Initialize Conversation History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful, conversational AI assistant running locally.")
    ]

# 4. Display Past Messages from History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# 5. Handle New User Input
if user_input := st.chat_input("Type your message here..."):
    # Display user message immediately
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Generate AI Response with Streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty() # Placeholder for streaming text
        full_response = ""
        
        try:
            # Stream the response chunk by chunk
            for chunk in model.stream(st.session_state.messages):
                full_response += chunk.content
                # Update the UI in real-time with a blinking cursor effect
                response_placeholder.markdown(full_response + "▌")
            
            # Final clean update without the cursor
            response_placeholder.markdown(full_response)
            
            # Save the AI's response to the history
            st.session_state.messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            st.error(f"❌ Error connecting to Ollama: {e}")
            st.info("Make sure 'ollama run llama3.2:3b' is active in your terminal.")