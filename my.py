import streamlit as st
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pinecone import Pinecone
import os
import torch
from InstructorEmbedding import INSTRUCTOR
HF_TOKEN = "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"
os.environ['PINECONE_API_KEY'] = '73e13e27-461e-4b8d-973c-44c08e37ec2d'
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism warnings
KEY_NEW=st.secrets['KEY_NEW']
# Streamlit UI configuration
st.set_page_config(page_title="Mangesh+Aastha", page_icon="👨‍❤️‍👩")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index_name = "projectkshitij"

# Load the embedding model only once
@st.cache_resource
def load_embeddings():
    return HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl",
        query_instruction="Represent the chat for retrieving similar chat conversations:",
    )

# Load the language model only once
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        temperature=0.7,
        model_name='gpt-3.5-turbo',
        openai_api_key= KEY_NEW
    )

# Initialize models
embeddings = load_embeddings()
llm = load_llm()

# Set up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Set up prompt template
system_template = """You are a loving and caring boyfriend designed to talk like {name}, the boyfriend of the user. User's name is Aastha.
Use the following pieces of context to answer Aastha's question if it's helpful. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Always maintain a loving, caring, and slightly playful tone in your responses. Use pet names and endearing terms occasionally.
Remember specific details about your relationship and reference them when appropriate.

{chat_history}
"""

human_template = "{question}"

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
]

prompt = ChatPromptTemplate.from_messages(messages)

def query_pinecone(query, top_k=2):
    index = pc.Index(index_name)
    results = index.query(vector=embeddings.embed_query(query), top_k=top_k)
    return results['matches']  # Assuming matches contain the document details

# Streamlit UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');

.big-font {
    font-family: 'Pacifico', cursive;
    font-size: 48px !important;
    font-weight: bold;
    color: #FF69B4;
    text-align: center;
    text-shadow: 2px 2px 4px #000000;
}
.stApp {
    background-image: linear-gradient(to right top, #ff9a9e, #fad0c4);
}
.stTextInput>div>div>input {
    border-radius: 20px;
}
.stButton>button {
    border-radius: 20px;
    background-color: #FF69B4;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Mini Kshitij: Always Here for You 💖</p>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a form for user input
with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_input("What's on your mind, baby? Kshitij is not listening as usual? Give me a try", key="user_input")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    # Query Pinecone for related context
    results = query_pinecone(user_input)

    # Format the retrieved context
    context = "\n\n".join([match['metadata']['text'] for match in results if 'metadata' in match])

    # Retrieve memory (conversation history)
    chat_history = memory.load_memory_variables({})

    # Create prompt with context and memory
    final_prompt = prompt.format_prompt(
        name="Kshitij",
        context=context,
        question=user_input,
        chat_history=chat_history['chat_history'] if 'chat_history' in chat_history else ""
    )

    # Generate response using the correct message formatting
    response = llm(final_prompt.to_messages())  # Call the LLM

    # Save response to memory
    assistant_response = response.content  # Access content directly
    memory.save_context({"input": user_input}, {"output": assistant_response})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Sidebar content
st.sidebar.markdown("## 💕 Our Special Moments")
st.sidebar.image("hello.png", caption="All of me loves all of you")
st.sidebar.markdown("## 🎵 Our Song")
st.sidebar.audio("James_Arthur_feat._Anne-Marie_Matthew_Brind_-_Rewrite_The_Stars_(mp3.pm).mp3")
st.sidebar.markdown("## 💌 Note")
st.sidebar.info("Hey baby, I created this chatbot to keep you company when I'm not around. It's like having a piece of me with you always. I love you! 😘")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Aastha's only Aastha's Kshitij")
