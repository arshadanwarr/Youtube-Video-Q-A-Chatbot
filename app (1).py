import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
import tempfile
import json

# ------------------- CSS Styling -------------------
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        color: #000000;
    }
    .stApp {
        background-color: #ffffff;
    }
    .stSidebar {
        background-color: #f8f9fa;
    }
    .chat-bubble {
        padding: 12px;
        margin: 8px 0;
        border-radius: 10px;
        max-width: 80%;
    }
    .human {
        background-color: #007bff;
        color: #ffffff;
        align-self: flex-end;
    }
    .ai {
        background-color: #e9ecef;
        color: #000000;
        align-self: flex-start;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Sidebar Inputs -------------------
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
youtube_url = st.sidebar.text_input("YouTube Video URL")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["llama3-8b-8192", "llama3-70b-8192"]  # Recommended new models
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------- Session State -------------------
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------- Helper Functions -------------------
def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return None

def get_transcript(video_id):
    transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([t['text'] for t in transcript_data])

def create_vectorstore(text):
    embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=" "
    )
    vectorstore = FAISS.from_texts([text], embedding=embedding)
    return vectorstore

def get_llm_response(query, vectorstore):
    docs = vectorstore.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in docs])
    template = """You are a helpful assistant. Use the following context:
    {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(template)
    final_prompt = prompt.format(context=context, question=query)

    llm = ChatGroq(api_key=api_key, model_name=model_choice, temperature=temperature)
    response = llm.invoke(final_prompt)
    return response.content

# ------------------- Main UI -------------------
st.title("YouTube Q&A Chatbot")
st.write("Enter a YouTube video link in the sidebar and start chatting!")

if st.sidebar.button("Extract Transcript"):
    if api_key and youtube_url:
        try:
            video_id = extract_video_id(youtube_url)
            with st.spinner("Extracting transcript..."):
                transcript = get_transcript(video_id)
                st.session_state.transcript = transcript
                st.session_state.vectorstore = create_vectorstore(transcript)
            st.success("Transcript extracted and vectorstore created!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please provide both API key and YouTube URL.")

# Display transcript if available
if st.session_state.transcript:
    with st.expander("View Transcript"):
        st.write(st.session_state.transcript)

# ------------------- Chat Section -------------------
st.subheader("Ask Questions About the Video")
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query and st.session_state.vectorstore:
        with st.spinner("Generating answer..."):
            answer = get_llm_response(query, st.session_state.vectorstore)
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        st.warning("Please extract the transcript first.")

# Display Chat History
if st.session_state.chat_history:
    st.subheader("📜 Chat History")
    for chat in st.session_state.chat_history:
        css_class = "human" if chat["role"] == "user" else "ai"
        st.markdown(f"<div class='chat-bubble {css_class}'>{chat['content']}</div>", unsafe_allow_html=True)

# ------------------- Download Chat History -------------------
if st.session_state.chat_history:
    chat_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history])
    st.download_button("📥 Download Chat History", chat_text, file_name="chat_history.txt")
