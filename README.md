# Voice-agent — Nano Voice Agents 🎙️📱

Real-time voice-based customer support agent for **Nano Voice Agents** — a fictional Indian telecom company. Customers speak to an AI assistant over a live audio stream that can look up account balances, answer plan questions using RAG over PDF docs, check network coverage, recommend plans, and escalate to human support.

Built with **LiveKit Agents**, **LangChain + Mistral**, **Docling**, and **Qdrant**.

---

## Features

- Live voice call via LiveKit (STT: Deepgram Nova-3, TTS: InWorld TTS-2)
- Text chat + voice upload via FastAPI (`/chat`, `/voice`)
- RAG over telecom PDFs (`docs/*.pdf`) with Qdrant
- Tools: `search_plans`, `check_account_balance`, `check_network_status`, `get_plan_recommendation`, `get_plan_details`, `escalate_to_support` (ticket `NANO-XXXXX`)
- Streamlit console: upload PDFs, inspect chunks, chat by text/audio
- LiveKit web token endpoint (`/livekit/token`, default room `nano-support`)

## Getting Started

### Prerequisites

- Python `>= 3.12`
- [LiveKit Cloud](https://livekit.io/) credentials (URL, API Key, API Secret)
- [Mistral](https://console.mistral.ai/api-keys) API Key

### Installation

```powershell
# Install dependencies (Windows PowerShell — run from project root)
uv sync

# Configure environment
Copy-Item .env.example .env
# Edit .env with your LiveKit and Mistral credentials

# Activate virtual env
.\.venv\Scripts\Activate.ps1

# Ingest PDF documents into vector DB
.\.venv\Scripts\python.exe ingest.py

# Start the voice agent
.\.venv\Scripts\python.exe -m voice_agent.main console
```

### Web API + UI

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

---

## Example Prompts

| Try saying… | What happens |
|---|---|
| *"What plans do you offer?"* | Searches plan docs via RAG |
| *"What's my balance? My number is 9876543210"* | Account lookup |
| *"What's the 5G coverage in Bangalore?"* | Returns network status by region |
| *"Recommend a plan for heavy usage"* | Plan recommendation |
| *"I want to speak to a human"* | Creates a support ticket and escalates |

---
Copyright©️ Codebasics Inc. All rights reserved.
