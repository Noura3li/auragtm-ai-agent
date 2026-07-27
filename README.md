# AuraGTM — AI Go-To-Market Strategy Agent

**Group:** Group03
**Live deployment:** https://auragtmai.com
**Project Files & Demo (Drive):** https://drive.google.com/drive/folders/1g8ABW5rjP9wrdwEVPX0kWaU7a0yZzSSs
# AuraGTM — AI Go-To-Market Strategy Agent

AuraGTM is a retrieval-augmented generation (RAG) application that generates
complete, evidence-grounded go-to-market strategies. It combines a hybrid
retrieval pipeline (vector search + keyword search), GPT-4o generation, a
multi-agent review workflow, real user authentication, and a project history
system, all served through a FastAPI backend.

---

## Folder Structure

```
AI_Go-To-Market_Strategy_Agent_Group03_Code_v1/
├── 01_data/              # Knowledge base documents (see README_DATA.txt inside)
├── 02_src/               # All application code
│   ├── templates/        # HTML pages (Jinja2)
│   ├── static/           # Static assets (logo, etc. — see README_STATIC.txt inside)
│   └── *.py              # Backend application code
├── 03_assets/            # Architecture diagram and workflow documentation images
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Prerequisites

- Python 3.10 or higher
- PostgreSQL (running locally, or a connection string to a cloud instance)
- An OpenAI API key
- (Optional, for the contact form) A Resend API key

---

## Setup Instructions

### 1. Move the data into place

The `01_data/` folder contains your knowledge base documents. Copy its contents
into `02_src/` so the application can find them at the path it expects:

```
02_src/knowledge_base/global/...
02_src/08_Clients_Data/<ClientName>/...
```

See `01_data/README_DATA.txt` for the exact expected structure.

### 2. Create a virtual environment

```
cd 02_src
python -m venv venv
venv\Scripts\activate        (Windows)
source venv/bin/activate     (Mac/Linux)
```

### 3. Install dependencies

```
pip install -r ../requirements.txt
```

### 4. Create a `.env` file inside `02_src/`

```
OPENAI_API_KEY=your_openai_key_here
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/auragtm_db
CONTACT_EMAIL=your_contact_email@example.com
RESEND_API_KEY=your_resend_api_key_here
```

`RESEND_API_KEY` is only required for the contact form to send emails; the
rest of the application works without it.

### 5. Set up the database

Make sure PostgreSQL is running, then create the tables:

```
python create_db.py
```

You should see `Tables created successfully!`.

### 6. Build the knowledge base vector database

```
python rag_pipeline.py
```

This reads everything in `knowledge_base/global/`, splits it into chunks,
embeds it using OpenAI's embeddings API, and stores it in a local ChromaDB
vector database (`vector_db/`). This step must be run once before the first
use, and again any time the knowledge base documents change.

### 7. Run the application

```
uvicorn app:app --reload
```

### 8. Open it in your browser

```
http://127.0.0.1:8000
```

Sign up for a new account, then explore the tool from the workspace.

---

## Key Features

- **GTM Strategy Generator** — full, structured go-to-market strategy grounded
  in real source documents, with sources listed separately.
- **Find Competitive Gaps**, **Compare 3 Directions**, **What-If Scenario
  Simulator**, **30/60/90 Day Launch Plan**, **Brand Match Scoring** — all
  available from a single tool selector in the workspace.
- **Strategist + Critic Agent Workflow** — a second AI agent independently
  reviews and refines the first agent's draft, with long-term project memory.
- **Ask AuraGTM** — a conversational assistant grounded in the knowledge base.
- **Client Mode** — upload client-specific documents to ground strategies in
  a particular client's own materials.
- Real authentication, PostgreSQL-backed project history with automatic
  versioning, and a public marketing site with a working contact form.

See `03_assets/` for a visual diagram of the system architecture and the
end-to-end AI/ML workflow.

---

## Notes on Deployment

The live version of this project is deployed on Render at
**https://auragtmai.com**, using a separate cloud-hosted PostgreSQL database
and OpenAI's embeddings API (rather than a local model) for compatibility
with the hosting platform's memory limits. The contact form uses the Resend
API rather than traditional SMTP, since the hosting provider's free tier
blocks outbound SMTP connections.
