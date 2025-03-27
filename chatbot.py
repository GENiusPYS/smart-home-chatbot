import os
import streamlit as st
import pandas as pd
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer

# Load Together AI API Key
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
TOGETHER_AI_ENDPOINT = "https://api.together.xyz/v1/chat/completions"

#  Connect to Qdrant
qdrant = QdrantClient("localhost", port=6333)

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create Collection in Qdrant
collection_name = "smart_home_logs"
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

#  Load Sensor & Activity Logs
sensor_df = pd.read_csv("kasteren_sensor_dataset.csv")
activity_df = pd.read_csv("kasteren_activity_dataset.csv")

#  Insert Data into Qdrant
def insert_into_qdrant(df, label):
    points = []
    for index, row in df.iterrows():
        text = f"{label}: {row['Activity Name' if 'Activity Name' in row else 'Sensor Name']} from {row['Start Time']} to {row['End Time']}"
        embedding = embedding_model.encode(text).tolist()
        points.append(PointStruct(id=index, vector=embedding, payload={"text": text}))
    qdrant.upsert(collection_name=collection_name, points=points)

insert_into_qdrant(sensor_df, "Sensor Triggered")
insert_into_qdrant(activity_df, "Activity Detected")

# Search Function in Qdrant
def search_qdrant(query, top_k=3):
    query_vector = embedding_model.encode(query).tolist()
    search_result = qdrant.search(collection_name=collection_name, query_vector=query_vector, limit=top_k)
    return [hit.payload["text"] for hit in search_result]

#  Call Together AI for Response
def query_together_ai(query):
    retrieved_context = search_qdrant(query)
    context_str = "\n".join(retrieved_context)
    prompt = f"Context:\n{context_str}\nUser: {query}\nAI:"

    headers = {"Authorization": f"Bearer {TOGETHER_AI_KEY}", "Content-Type": "application/json"}
    data = {"model": "mistralai/Mistral-7B-Instruct-v0.2", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
    
    response = requests.post(TOGETHER_AI_ENDPOINT, json=data, headers=headers)
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Error: No response")




import streamlit as st

# 🌐 Streamlit UI Setup
st.set_page_config(page_title="Smart Home Chatbot", page_icon="🏠", layout="wide")

# 💬 Title & Description
st.title("🏠 Smart Home Chatbot")
st.markdown("💡 *Ask about smart home activities and appliances!*")

# 🌟 Sidebar - Chatbot Settings
with st.sidebar:
    st.header("⚙️ Chatbot Settings")
    temperature = st.slider("🔧 Creativity Level (Temperature)", 0.0, 1.0, 0.7)
    max_tokens = st.slider("📝 Max Response Length", 50, 500, 200)
    st.markdown("---")
    st.markdown("### ℹ️ About the Chatbot")
    st.text("This chatbot retrieves answers from the Kasteren smart home sensor and activity data.")
    st.text("Made Using ** QdrantDB + Together AI **.")

st.markdown("""
    <style>
    .chat-container {
        max-height: 400px; 
        overflow-y: auto; 
        padding: 10px; 
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #f9f9f9;
    }
    .user-message {
        background-color: #DCF8C6; 
        color: black; 
        padding: 10px; 
        border-radius: 10px; 
        margin-bottom: 5px;
        width: fit-content;
    }
    .chatbot-message {
        background-color: #E1F5FE; 
        color: black; 
        padding: 10px; 
        border-radius: 10px; 
        margin-bottom: 5px;
        width: fit-content;
    }
    </style>
""", unsafe_allow_html=True)

# 🗂 Chat History (Stored in Session State)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📢 User Input Section
st.subheader("💬 Chat Here:")
user_query = st.text_input("Ask a question about smart home activities:", placeholder="E.g., When was the fridge last used?")

if st.button("🚀 Submit"):
    if user_query:
        # Simulate chatbot response time
        with st.spinner("🤖 Thinking..."):
            response = query_together_ai(user_query)  

        # 📝 Store in Chat History
        st.session_state.chat_history.append(("🧑‍💻 You", user_query))
        st.session_state.chat_history.append(("🤖 Chatbot", response))

# 🗂 Display Chat History
st.subheader("📝 Chat History:")
chat_container = st.container()
with chat_container:
    for sender, message in st.session_state.chat_history:
        if sender == "🧑‍💻 You":
            st.markdown(f'<div class="user-message"><b>{sender}:</b> {message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chatbot-message"><b>{sender}:</b> {message}</div>', unsafe_allow_html=True)






