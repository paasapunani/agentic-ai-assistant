import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Agentic AI Assistant", page_icon="🤖")
st.title("🤖 Agentic AI Assistant")
st.caption("LangGraph agent with tool calling · FastAPI backend")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.write(f"Thread: `{st.session_state.thread_id[:8]}`")
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.markdown("**Try:**\n- What's the weather in Mumbai and Delhi?\n"
                "- Who is Ada Lovelace? Then compute 2024 - 1815\n"
                "- What time is it in Tokyo?")


def show_trace(plan, steps):
    with st.expander("Agent execution"):
        st.markdown("**Plan**")
        st.text(plan)
        for s in steps:
            if s["type"] == "tool_call":
                st.markdown(f"🔧 **Call** `{s['name']}`")
                st.json(s["args"])
            else:
                st.markdown(f"📥 **Result** from `{s['name']}`")
                st.code(s["content"])


for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item["role"] == "assistant":
            show_trace(item["plan"], item["steps"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is working..."):
            try:
                r = requests.post(
                    f"{API_URL}/chat",
                    json={"message": prompt, "thread_id": st.session_state.thread_id},
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()
        st.markdown(data["reply"])
        show_trace(data["plan"], data["steps"])

    st.session_state.history.append({
        "role": "assistant", "content": data["reply"],
        "plan": data["plan"], "steps": data["steps"],
    })
