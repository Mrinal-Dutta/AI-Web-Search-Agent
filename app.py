
import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool

# Page settings
st.set_page_config(
    page_title="AI Web Search Agent",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 AI Web Search Agent")
st.caption("Ask anything. I can search the web and help you find answers.")

# API Keys
groq_key = st.secrets["GROQ_API_KEY"]
serper_key = st.secrets["SERPER_API_KEY"]

# Google Search Setup
google_search = GoogleSerperAPIWrapper(
    serper_api_key=serper_key,
    k=3
)

# Search Tool
@tool
def web_search(query: str) -> str:
    """Search Google for current information and answer factual questions."""
    return google_search.run(query)

# AI Model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_key,
    temperature=0
)

# AI Agent
agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt="""
- Use web_search for current information, latest news, facts, and questions that need verification.
- For simple follow-up questions about previous answers, use conversation history when sufficient.
- Avoid unnecessary searches.
- Never guess facts that require verification.
"""
)

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# New Chat
if st.button("🗑️ New Chat"):
    st.session_state.messages = []
    st.rerun()

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
question = st.chat_input("Ask me anything...")

if question:

    # Save user question
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Send previous chat history to agent
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):

            res = agent.invoke({
                "messages": st.session_state.messages
            })

            answer = res["messages"][-1].content

            st.markdown(answer)

    # Save AI answer
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
