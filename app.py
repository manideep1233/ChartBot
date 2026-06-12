import streamlit as st
import json
import os
import re
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. UI Configuration
st.set_page_config(page_title="AI Page-Targeted Coach", page_icon="📚", layout="centered")
st.title("📚 Local AI Page-Targeted Coach")
st.caption("Tell the bot a specific page number, and it will pull data directly from that page.")

# 2. Initialize Models & Connect to FAISS
FAISS_DIR = "faiss_db"

@st.cache_resource
def load_app_resources():
    llm = ChatOllama(model="llama3.2:3b", temperature=0.3)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    if os.path.exists(FAISS_DIR):
        vector_db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_db = None
        
    return llm, vector_db

model, db_retriever = load_app_resources()

# 3. History Persistence
HISTORY_FILE = "study_history.json"
SYSTEM_PROMPT = (
    "You are an elite academic tutor preparing the user for an intensive corporate/technical exam. "
    "Your primary job is to explain the textbook content provided to you with deep reasoning. "
    "Always reference the specific page number you are pulling facts from. "
    "Log progress at the bottom under '--- DAILY STUDY LOG ---' when instructed."
)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                data = json.load(f)
                return [SystemMessage(content=SYSTEM_PROMPT)] + [
                    HumanMessage(content=m["content"]) if m["type"] == "human" else AIMessage(content=m["content"])
                    for m in data[1:]
                ]
            except: return None
    return None

def save_history(messages):
    data = [{"type": "system", "content": SYSTEM_PROMPT}]
    for msg in messages[1:]:
        data.append({"type": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content})
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=4)

if "messages" not in st.session_state:
    saved = load_history()
    st.session_state.messages = saved if saved else [SystemMessage(content=SYSTEM_PROMPT)]

# 4. Sidebar Controls
st.sidebar.header("📋 Study Dashboard")
if db_retriever:
    st.sidebar.success("✅ Textbook Loaded & Ready")
else:
    st.sidebar.warning("⚠️ Run 'python ingest.py' first.")

if st.sidebar.button("🗑️ Reset Progress"):
    if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
    st.rerun()

# 5. Render Chat Screen
for msg in st.session_state.messages[1:]:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

# 6. Chat Input & Page Retrieval Filtering Logic
if user_input := st.chat_input("e.g., Explain the core algorithm on page 45..."):
    st.chat_message("user").write(user_input)
    
    context_str = ""
    
    if db_retriever:
        # Regex to capture patterns like "page 45", "p 45", "page: 45"
        match = re.search(r'\b(?:page|p\.?|pg\.?)\s*[:\s]?\s*(\d+)\b', user_input.lower())
        
        if match:
            target_page = int(match.group(1))
            # PyPDF is 0-indexed internally (Page 1 in a PDF reader is index 0 in Python metadata)
            internal_page_index = target_page - 1
            
            st.toast(f"Searching specifically for Page {target_page}...")
            
            # Extract chunks directly matching this exact page number from the FAISS database index
            all_docs = db_retriever.docstore._dict.values()
            matched_chunks = [
                doc for doc in all_docs 
                if doc.metadata.get('page') == internal_page_index
            ]
            
            if matched_chunks:
                context_str = f"\n\n[CRITICAL TEXTBOOK CONTENT FROM PAGE {target_page}]:\n" + "\n".join([c.page_content for c in matched_chunks])
            else:
                context_str = f"\n\n[System Alert: The user asked for page {target_page}, but it exceeded the document index length or wasn't compiled.]"
                st.sidebar.error(f"Could not extract Page {target_page} text directly.")
        
        # Fallback to standard similarity search if no specific page number was mentioned
        if not context_str:
            relevant_docs = db_retriever.similarity_search(user_input, k=3)
            context_str = "\n\nRelevant Textbook Context:\n" + "\n".join([doc.page_content for doc in relevant_docs])
    
    # Append input with context to pass to Llama, but preserve clean display
    st.session_state.messages.append(HumanMessage(content=user_input + context_str))
    save_history(st.session_state.messages)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            for chunk in model.stream(st.session_state.messages):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            
            st.session_state.messages[-1] = HumanMessage(content=user_input)
            st.session_state.messages.append(AIMessage(content=full_response))
            save_history(st.session_state.messages)
        except Exception as e:
            st.error(f"Error: {e}")