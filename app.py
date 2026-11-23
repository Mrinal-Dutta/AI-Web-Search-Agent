
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

# App design
st.title("🔎 AI Web Search Agent")
st.caption("Ask anything. I can search the web and help you find answers.")

# API keys
groq_key = st.secrets["GROQ_API_KEY"]
serper_key = st.secrets["SERPER_API_KEY"]

# Google search setup
google_search = GoogleSerperAPIWrapper(
    serper_api_key=serper_key
)

# Search tool
@tool
def web_search(query: str) -> str:
    """Search Google for latest information about a topic."""
    return google_search.run(query)

# AI model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0
)

# AI Agent
agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt="""
You are an AI web search assistant.

Rules:
- Always use web_search for current information, news, sports results and factual questions.
- Search the web before answering. Never guess.
- Carefully read and verify the search results before giving an answer.
- Use only information supported by search results.
- If the information is unavailable or not confirmed, clearly say so.
- Never invent dates, winners, scores, events or facts.
- Give direct, simple and concise answers.
- Answer in 1-3 sentences unless the user asks for details.
- If the user asks for a short answer, keep it very short.
"""
)

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# New chat button
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

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking and searching..."):

            response = agent.invoke(
                {"messages": st.session_state.messages}
            )

            answer = response["messages"][-1].content
            st.markdown(answer)

    # Save AI response in memory
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
