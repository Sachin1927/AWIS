AWIS – Adaptive Workforce Intelligence System

AWIS is a modular AI system combining LLMs, LangChain, RAG pipelines, classical ML models, and a full FastAPI backend + Streamlit frontend.
It supports HR analytics, skill forecasting, mobility analysis, and knowledge retrieval from internal documents.

Designed for production use: configurable, containerized, and test-driven.

📌 Key Features
✔ LLM + LangChain

Central AWISAgent for reasoning, tool use, and chat.

LangChain tools for attrition, forecasting, mobility, and HR policy retrieval.

Extensible chain architecture under src/chains/.

✔ Full RAG Pipeline

Located in src/rag_index/:

Preprocessing & chunking

Embeddings (HuggingFaceEmbeddings)

FAISS/Chroma vector store

Retriever + context injection

Integrated into agent workflow

✔ ML Models

Used in:

Attrition prediction

Skill demand forecasting

Employee mobility recommendations

✔ FastAPI Backend

REST API with structured routers:

/auth

/attrition

/forecast

/mobility

/rag

/chat

✔ Streamlit Frontend

Interactive interface for employees, HR teams, and dashboards.

✔ Docker + CI/CD

Dockerized microservices + automated testing via GitHub Actions.

🧩 Architecture Diagram
flowchart LR
    subgraph Client
        UI[Streamlit UI]
        External[External Frontend]
    end

    subgraph API[FastAPI Backend]
        Auth[/Auth Router/]
        Attrition[/Attrition Router/]
        Forecast[/Forecast Router/]
        Mobility[/Mobility Router/]
        RAG[/RAG Router/]
        Chat[/Chat Router/]
    end

    subgraph Logic[AI Logic Layer]
        subgraph ML[ML Models]
            M1[(Attrition Model)]
            M2[(Forecast Model)]
            M3[(Mobility Model)]
        end

        subgraph RAGLayer[RAG + LangChain]
            Ret[[Retriever]]
            VS[(Vector Store)]
            Emb[(HuggingFace Embeddings)]
            Agent[[AWISAgent]]
            Tools[[LangChain Tools]]
        end

        Utils[[Config / Logging]]
    end

    subgraph Storage[Storage]
        Data[(./data)]
        Models[(./models)]
        Index[(./rag_index/vectorstore)]
    end

    UI --> API
    External --> API

    API --> Auth
    API --> Attrition
    API --> Forecast
    API --> Mobility
    API --> RAG
    API --> Chat

    Attrition --> M1
    Forecast --> M2
    Mobility --> M3

    RAG --> Ret
    Ret --> VS
    Ret --> Emb
    RAG --> Agent
    Agent --> Tools
    Tools --> Ret

    Logic --> Utils
    ML --> Models
    RAGLayer --> Index
    Data --> ML
    Data --> RAGLayer

🔍 RAG Sequence Diagram
sequenceDiagram
    participant U as User
    participant C as Client (Streamlit / Frontend)
    participant API as FastAPI /rag
    participant Ret as RAGRetriever
    participant VS as Vector Store
    participant AG as AWISAgent
    participant LLM as LLM (LangChain)

    U->>C: Ask question
    C->>API: POST /rag/query
    API->>Ret: retrieve(query, k)
    Ret->>VS: similarity_search
    VS-->>Ret: top-k documents
    Ret-->>API: docs + metadata
    API->>AG: send query + context
    AG->>LLM: LangChain prompt + tools
    LLM-->>AG: answer + reasoning
    AG-->>API: formatted response
    API-->>C: JSON answer with citations
    C-->>U: Show final answer

📚 API Documentation Summary

A full API doc file is included in:

👉 docs/API.md


## Main Endpoints

| Router    | Path        | Purpose                                      |
|-----------|-------------|-----------------------------------------------|
| **Auth**      | `/auth/*`     | Login, tokens, user info                     |
| **Attrition** | `/attrition/*` | Employee attrition prediction + dashboard stats |
| **Forecast**  | `/forecast/*`  | Skill demand forecasting                     |
| **Mobility**  | `/mobility/*`  | Internal career mobility                      |
| **RAG**       | `/rag/*`       | Document search, HR policy RAG               |
| **Chat**      | `/chat/*`      | General LLM agent chat                        |


All endpoints are automatically documented in FastAPI Swagger:

/docs (Swagger UI)
/redoc (ReDoc)

⚙️ Installation
git clone https://github.com/<your-username>/AWIS.git
cd AWIS
python -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate
pip install -r requirements.txt

🚀 Running the System
Start the FastAPI backend
uvicorn src.api.main:app --reload

Start the Streamlit app
streamlit run app/streamlit_app.py

Run everything
python run_all.py

📦 Docker Deployment
docker-compose up --build


Services:

FastAPI backend

Streamlit UI

Vector store persistent volume

Stop:

docker-compose down

🧪 Testing
pytest -v

🔧 CI/CD – GitHub Actions

Add this file:

.github/workflows/ci.yml
name: AWIS CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: pytest -v
