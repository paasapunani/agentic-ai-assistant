# Agentic AI Assistant with Tool Calling

Python · LangGraph · LangChain · LLM APIs · FastAPI · Streamlit

- **LangGraph** graph: `planner -> agent <-> tools`, with per-thread memory (MemorySaver).
- **LangChain tools**: calculator, current time, weather, Wikipedia.
- **FastAPI**: `POST /chat`, `GET /health`.
- **Streamlit**: chat UI showing the plan and every tool call/result.

## Run
    python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env                               # add your API key
    uvicorn app.main:app --reload --port 8000          # terminal 1
    streamlit run streamlit_app.py                     # terminal 2
