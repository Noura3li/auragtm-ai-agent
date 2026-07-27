"""
AuraGTM — FastAPI Web App
=========================
Public marketing pages:  /  (landing)  and  /services
Gated tool (login):      /app          (the strategy generator + chatbot)
"""

from auth_service import login_user, create_user
from memory_service import save_strategy, get_user_history, get_site_stats

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

import uvicorn
import os
import shutil
import traceback
import requests

from auragtm_engine import get_engine
from rag_pipeline import ingest
from agent_workflow import run_two_agent_workflow
from project_memory import get_memory


app = FastAPI(title="AuraGTM", version="6.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

jinja_env = Environment(
    loader=FileSystemLoader(os.path.join(BASE_DIR, "templates")),
    cache_size=0
)

static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ---------------------------------------------------------------------------
# Custom branded error pages (404 / 500)
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    template = jinja_env.get_template("error.html")
    return HTMLResponse(
        content=template.render(
            code="404",
            title="Page Not Found",
            message="The page you're looking for doesn't exist or may have moved."
        ),
        status_code=404
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    traceback.print_exc()
    template = jinja_env.get_template("error.html")
    return HTMLResponse(
        content=template.render(
            code="500",
            title="Something Went Wrong",
            message="An unexpected error occurred on our end. Please try again in a moment."
        ),
        status_code=500
    )


# ---------------------------------------------------------------------------
# Real user authentication
# ---------------------------------------------------------------------------

COOKIE_NAME = "aura_auth"
COOKIE_MAX_AGE = 60 * 60 * 8
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"


def is_authed(request: Request) -> bool:
    return request.cookies.get(COOKIE_NAME) is not None


def get_current_user_id(request: Request):
    user_id = request.cookies.get(COOKIE_NAME)

    if not user_id:
        return None

    try:
        return int(user_id)
    except ValueError:
        return None


def auth_required_response():
    return JSONResponse(
        {"success": False, "error": "Please sign in first."},
        status_code=401
    )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class GTMRequest(BaseModel):
    product_name:        str
    product_description: str
    industry:            str
    region:              str
    business_goal:       str
    brand_tone:          str = "Professional & Authoritative"
    mode:                str = "General Mode"
    client_name:         str = ""
    refinement:          str = ""


class AskRequest(BaseModel):
    query:       str
    client_name: str = ""


class BrandScoreRequest(BaseModel):
    strategy:   str
    brand_tone: str = "Professional & Authoritative"


class WhatIfRequest(BaseModel):
    product_name:        str
    product_description: str
    industry:            str
    region:              str
    business_goal:       str
    brand_tone:          str = "Professional & Authoritative"
    mode:                str = "General Mode"
    client_name:         str = ""
    current_strategy:    str = ""
    scenario:            str


class ConfidenceScoreRequest(BaseModel):
    product_name:        str
    product_description: str
    industry:            str
    region:              str
    business_goal:       str
    brand_tone:          str = "Professional & Authoritative"
    mode:                str = "General Mode"
    client_name:         str = ""
    strategy:            str


class PipelineRequest(BaseModel):
    product_name:        str
    product_description: str
    industry:            str
    region:              str
    business_goal:       str
    brand_tone:          str = "Professional & Authoritative"
    mode:                str = "General Mode"
    client_name:         str = ""


class LaunchTimelineRequest(BaseModel):
    product_name:        str
    product_description: str
    industry:            str
    region:              str
    business_goal:       str
    brand_tone:          str = "Professional & Authoritative"
    mode:                str = "General Mode"
    client_name:         str = ""


class ContactRequest(BaseModel):
    name:    str
    email:   str
    message: str


# ---------------------------------------------------------------------------
# Public pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    template = jinja_env.get_template("landing.html")
    return HTMLResponse(content=template.render())


@app.get("/services", response_class=HTMLResponse)
async def services(request: Request):
    template = jinja_env.get_template("services.html")
    return HTMLResponse(content=template.render())


@app.get("/site-stats")
async def site_stats():
    """Public, anonymous, aggregate-only stats for the landing page. No per-user data."""
    try:
        stats = get_site_stats()
        return {"success": True, **stats}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/contact")
async def contact(data: ContactRequest):
    try:
        name    = data.name.strip()
        email   = data.email.strip()
        message = data.message.strip()

        if not name or not email or not message:
            return JSONResponse({"success": False, "error": "Please fill in all fields."})

        contact_email = os.getenv("CONTACT_EMAIL")
        resend_api_key = os.getenv("RESEND_API_KEY")

        if not contact_email or not resend_api_key:
            return JSONResponse({"success": False, "error": "Contact form is not configured yet."})

        body = (
            f"New inquiry from the AuraGTM website\n\n"
            f"Name: {name}\n"
            f"Email: {email}\n\n"
            f"Message:\n{message}\n"
        )

        res = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {resend_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": "AuraGTM <hello@auragtmai.com>",
                "to": [contact_email],
                "reply_to": email,
                "subject": f"AuraGTM Inquiry from {name}",
                "text": body,
            },
            timeout=10,
        )

        if res.status_code >= 400:
            raise Exception(f"Resend API error {res.status_code}: {res.text}")

        return JSONResponse({"success": True, "message": "Thanks! Your message has been sent."})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": "Could not send your message. Please try again later."})


# ---------------------------------------------------------------------------
# Register / Login / Logout
# ---------------------------------------------------------------------------

@app.post("/register")
async def register(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    success, message = create_user(
        username=username,
        email=email,
        password=password
    )

    if not success:
        return RedirectResponse(
            url="/?register_error=1#login",
            status_code=303
        )

    user = login_user(email, password)

    resp = RedirectResponse(
        url="/app",
        status_code=303
    )

    resp.set_cookie(
        COOKIE_NAME,
        str(user.id),
        httponly=True,
        secure=COOKIE_SECURE,
        max_age=COOKIE_MAX_AGE,
        samesite="lax"
    )

    return resp


@app.post("/login")
async def login(
    email: str = Form(...),
    password: str = Form(...)
):
    user = login_user(email, password)

    if not user:
        return RedirectResponse(
            url="/?error=1#login",
            status_code=303
        )

    resp = RedirectResponse(
        url="/app",
        status_code=303
    )

    resp.set_cookie(
        COOKIE_NAME,
        str(user.id),
        httponly=True,
        secure=COOKIE_SECURE,
        max_age=COOKIE_MAX_AGE,
        samesite="lax"
    )

    return resp


@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    return resp


# ---------------------------------------------------------------------------
# Gated tool page
# ---------------------------------------------------------------------------

@app.get("/app", response_class=HTMLResponse)
async def tool(request: Request):

    if not is_authed(request):
        return RedirectResponse(url="/#login", status_code=303)

    engine = get_engine()
    clients = [str(c) for c in engine.available_clients] if engine.available_clients else []

    template = jinja_env.get_template("index.html")
    return HTMLResponse(content=template.render(clients=clients))


# ---------------------------------------------------------------------------
# Project history
# ---------------------------------------------------------------------------

@app.get("/history")
async def history(request: Request):

    if not is_authed(request):
        return auth_required_response()

    user_id = get_current_user_id(request)
    records = get_user_history(user_id)

    return {
        "success": True,
        "history": [
            {
                "id": item.id,
                "project_name": item.project_name,
                "client_name": item.client_name,
                "product_name": item.product_name,
                "industry": item.industry,
                "region": item.region,
                "business_goal": item.business_goal,
                "brand_tone": item.brand_tone,
                "strategy_version": item.strategy_version,
                "recommended_strategy": item.recommended_strategy,
                "created_at": str(item.created_at)
            }
            for item in records
        ]
    }


# ---------------------------------------------------------------------------
# Main GTM generation
# ---------------------------------------------------------------------------

@app.post("/generate")
async def generate(data: GTMRequest, request: Request):

    if not is_authed(request):
        return auth_required_response()

    try:
        engine = get_engine()

        result = engine.generate_gtm_strategy({
            "product_name"       : data.product_name,
            "product_description": data.product_description,
            "industry"           : data.industry,
            "region"             : data.region,
            "business_goal"      : data.business_goal,
            "brand_tone"         : data.brand_tone,
            "mode"               : data.mode,
            "client_name"        : data.client_name,
            "refinement"         : data.refinement,
        })

        save_strategy(
            {
                "product_name": data.product_name,
                "product_description": data.product_description,
                "industry": data.industry,
                "region": data.region,
                "business_goal": data.business_goal,
                "brand_tone": data.brand_tone,
                "client_name": data.client_name,
            },
            result["strategy"],
            user_id=get_current_user_id(request)
        )

        return {"success": True, **result}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/generate-options")
async def generate_options(data: GTMRequest, request: Request):

    if not is_authed(request):
        return auth_required_response()

    try:
        engine = get_engine()

        result = engine.generate_gtm_options({
            "product_name"       : data.product_name,
            "product_description": data.product_description,
            "industry"           : data.industry,
            "region"             : data.region,
            "business_goal"      : data.business_goal,
            "brand_tone"         : data.brand_tone,
            "mode"               : data.mode,
            "client_name"        : data.client_name,
        })

        save_strategy(
            {
                "product_name": data.product_name,
                "product_description": data.product_description,
                "industry": data.industry,
                "region": data.region,
                "business_goal": data.business_goal,
                "brand_tone": data.brand_tone,
                "client_name": data.client_name,
            },
            result["options"],
            user_id=get_current_user_id(request)
        )

        return {"success": True, **result}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Existing AI features
# ---------------------------------------------------------------------------

@app.post("/brand-score")
async def brand_score(data: BrandScoreRequest, request: Request):

    if not is_authed(request):
        return auth_required_response()

    try:
        engine = get_engine()
        result = engine.score_brand_match(data.strategy, data.brand_tone)
        return {"success": True, **result}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/find-gaps")
async def find_gaps(data: GTMRequest, request: Request):

    if not is_authed(request):
        return auth_required_response()

    try:
        engine = get_engine()

        result = engine.find_competitive_gaps({
            "product_name"       : data.product_name,
            "product_description": data.product_description,
            "industry"           : data.industry,
            "region"             : data.region,
            "business_goal"      : data.business_goal,
            "brand_tone"         : data.brand_tone,
            "mode"               : data.mode,
            "client_name"        : data.client_name,
        })

        return {"success": True, **result}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/what-if")
async def what_if(data: WhatIfRequest, request: Request):

    if not is_authed(request):
        return auth_required_response()

    try:
        engine = get_engine()

        result = engine.simulate_what_if({
            "product_name": data.product_name,
            "product_description": data.product_description,
            "industry": data.industry,
            "region": data.region,
            "business_goal": data.business_goal,
            "brand_tone": data.brand_tone,
            "mode": data.mode,
            "client_name": data.client_name,
            "current_strategy": data.current_strategy,
            "scenario": data.scenario,
        })

        return {"success": True, **result}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/confidence-score")
async def confidence_score(data: ConfidenceScoreRequest, request: Request):

    if not is_authed(request):
        return auth_required_response()

    try:
        engine = get_engine()

        result = engine.calculate_confidence_score({
            "product_name": data.product_name,
            "product_description": data.product_description,
            "industry": data.industry,
            "region": data.region,
            "business_goal": data.business_goal,
            "brand_tone": data.brand_tone,
            "mode": data.mode,
            "client_name": data.client_name,
            "strategy": data.strategy,
        })

        return {"success": True, **result}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/pipeline-status")
async def pipeline_status(data: PipelineRequest, request: Request):

    if not is_authed(request):
        return auth_required_response()

    required_inputs = [
        data.product_name,
        data.product_description,
        data.industry,
        data.region,
        data.business_goal,
        data.brand_tone
    ]

    input_ready = all(value.strip() for value in required_inputs)

    client_selected = (
        data.mode == "Client Mode"
        and data.client_name.strip() != ""
    )

    client_db_path = os.path.join(
        BASE_DIR,
        "vector_db",
        "clients",
        f"{data.client_name}_db"
    )

    client_kb_ready = (
        client_selected
        and os.path.exists(client_db_path)
    )

    stages = [
        {
            "step": 1,
            "name": "Input Collection",
            "status": "completed" if input_ready else "pending",
            "details": "Product, industry, region, goal, and brand tone were provided."
            if input_ready else
            "Some required product or GTM inputs are still missing."
        },
        {
            "step": 2,
            "name": "Knowledge Base Selection",
            "status": "completed",
            "details": "General GTM Knowledge Base selected."
            if not client_selected else
            f"General KB + Client KB selected for {data.client_name}."
        },
        {
            "step": 3,
            "name": "Client Knowledge Check",
            "status": "completed" if client_kb_ready else "skipped",
            "details": f"Client vector database found for {data.client_name}."
            if client_kb_ready else
            "Client KB not selected or not available. System will rely on General KB."
        },
        {
            "step": 4,
            "name": "Retrieval",
            "status": "ready" if input_ready else "pending",
            "details": "AuraGTM is ready to retrieve relevant GTM, market, persona, localization, and competitor context."
            if input_ready else
            "Retrieval will start after all required inputs are provided."
        },
        {
            "step": 5,
            "name": "Strategy Generation",
            "status": "ready" if input_ready else "pending",
            "details": "Main strategy, A/B/C options, what-if simulation, and confidence scoring are ready to run."
            if input_ready else
            "Strategy generation is waiting for complete inputs."
        },
        {
            "step": 6,
            "name": "Validation & History",
            "status": "ready" if input_ready else "pending",
            "details": "Outputs can be saved to PostgreSQL project history with user ownership and versioning."
            if input_ready else
            "History saving will be available after generation."
        }
    ]

    return {
        "success": True,
        "type": "living_pipeline",
        "pipeline": stages
    }


@app.post("/launch-timeline")
async def launch_timeline(data: LaunchTimelineRequest, request: Request):

    if not is_authed(request):
        return auth_required_response()

    try:
        engine = get_engine()

        result = engine.generate_launch_timeline({
            "product_name": data.product_name,
            "product_description": data.product_description,
            "industry": data.industry,
            "region": data.region,
            "business_goal": data.business_goal,
            "brand_tone": data.brand_tone,
            "mode": data.mode,
            "client_name": data.client_name,
        })

        return {"success": True, **result}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/ask")
async def ask(data: AskRequest, request: Request):
    """Ask AuraGTM a strategic GTM question."""

    if not is_authed(request):
        return auth_required_response()

    try:
        question = data.query.strip()

        if not question:
            return {"success": False, "error": "Please enter a question."}

        engine = get_engine()

        result = engine.ask_aura(
            query=question,
            client_name=(data.client_name.strip() or None)
        )

        return {"success": True, **result}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Client upload and client status
# ---------------------------------------------------------------------------

@app.post("/upload-client")
async def upload_client(
    request: Request,
    client_name: str = Form(...),
    files: list[UploadFile] = File(...)
):

    if not is_authed(request):
        return auth_required_response()

    try:
        client_name = client_name.strip()

        if not client_name:
            return JSONResponse(
                {"success": False, "error": "Client name is required."},
                status_code=400
            )

        allowed = {".pdf", ".docx", ".doc", ".txt"}
        saved_files = []

        client_folder = os.path.join(BASE_DIR, "08_Clients_Data", client_name)
        os.makedirs(client_folder, exist_ok=True)

        for file in files:
            ext = os.path.splitext(file.filename)[1].lower()

            if ext not in allowed:
                continue

            dest = os.path.join(client_folder, file.filename)

            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)

            saved_files.append(file.filename)

        if not saved_files:
            return JSONResponse(
                {
                    "success": False,
                    "error": "No supported files found. Use PDF, DOCX, or TXT."
                },
                status_code=400
            )

        clients_vector_root = os.path.join(BASE_DIR, "vector_db", "clients")
        db_dir = os.path.join(clients_vector_root, f"{client_name}_db")
        hash_reg = os.path.join(clients_vector_root, f"{client_name}_ingested.json")

        os.makedirs(db_dir, exist_ok=True)

        ingest(
            source_folders=[client_folder],
            db_directory=db_dir,
            hash_registry=hash_reg,
            rebuild=False
        )

        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        _emb = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        _db = Chroma(
            persist_directory=db_dir,
            embedding_function=_emb
        )

        chunk_count = _db._collection.count()

        if chunk_count == 0:
            return JSONResponse(
                {
                    "success": False,
                    "error": (
                        "Files were saved but no text could be extracted. "
                        "The documents may be scanned images or password-protected."
                    )
                },
                status_code=400
            )

        engine = get_engine()
        engine.reload_client(client_name)

        return JSONResponse({
            "success": True,
            "client_name": client_name,
            "files_saved": saved_files,
            "chunks_indexed": chunk_count,
            "message": (
                f"{len(saved_files)} file(s) uploaded and indexed for [{client_name}] "
                f"- {chunk_count} chunks stored in vector DB."
            )
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


@app.get("/clients")
async def clients(request: Request):

    if not is_authed(request):
        return auth_required_response()

    engine = get_engine()
    return [str(c) for c in engine.available_clients]


@app.get("/clients/{client_name}/status")
async def client_status(client_name: str, request: Request):

    if not is_authed(request):
        return auth_required_response()

    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        db_dir = os.path.join(
            BASE_DIR,
            "vector_db",
            "clients",
            f"{client_name}_db"
        )

        if not os.path.exists(db_dir):
            return JSONResponse({
                "exists": False,
                "chunks": 0,
                "client": client_name
            })

        _emb = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        _db = Chroma(
            persist_directory=db_dir,
            embedding_function=_emb
        )

        count = _db._collection.count()

        return JSONResponse({
            "exists": True,
            "chunks": count,
            "client": client_name
        })

    except Exception as e:
        return JSONResponse({
            "exists": False,
            "chunks": 0,
            "error": str(e)
        })


# ---------------------------------------------------------------------------
# Task 5 — 2-Agent Workflow  (Agent 1: Strategist -> Agent 2: Critic/Reviewer)
# ---------------------------------------------------------------------------

@app.post("/agent-workflow/generate")
async def agent_workflow_generate(data: GTMRequest, request: Request):
    """
    Runs Agent 1 (Strategist — existing engine) and then
    Agent 2 (Critic/Reviewer) on top of it, with long-term project
    memory (Task 6) automatically pulled in and saved back.
    """
    if not is_authed(request):
        return auth_required_response()
    try:
        result = run_two_agent_workflow({
            "product_name"       : data.product_name,
            "product_description": data.product_description,
            "industry"           : data.industry,
            "region"             : data.region,
            "business_goal"      : data.business_goal,
            "brand_tone"         : data.brand_tone,
            "mode"               : data.mode,
            "client_name"        : data.client_name,
            "refinement"         : data.refinement,
        })
        return {"success": True, **result}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/agent-workflow", response_class=HTMLResponse)
async def agent_workflow_page(request: Request):
    """Agent workflow demo page."""
    if not is_authed(request):
        return RedirectResponse(url="/#login", status_code=303)
    template = jinja_env.get_template("agent_workflow.html")
    return HTMLResponse(content=template.render())


# ---------------------------------------------------------------------------
# Task 6 — Long-Term Memory / Project History (SQLite via project_memory.py)
# ---------------------------------------------------------------------------

@app.get("/project-history")
async def list_projects(request: Request):
    """All projects (clients or general products) that have saved memory."""
    if not is_authed(request):
        return auth_required_response()
    try:
        memory = get_memory()
        return {"success": True, "projects": memory.list_projects()}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/project-history/{project_name}")
async def project_history(project_name: str, request: Request, by: str = "client"):
    """
    by=client  (default) -> project_name is a client_name (Client Mode)
    by=product            -> project_name is a product_name (General Mode)
    """
    if not is_authed(request):
        return auth_required_response()
    try:
        memory = get_memory()
        if by == "product":
            history = memory.get_history(product_name=project_name, limit=50)
        else:
            history = memory.get_history(client_name=project_name, limit=50)
        return {"success": True, "history": history}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.delete("/project-history/{project_name}")
async def clear_project_history(project_name: str, request: Request, by: str = "client"):
    if not is_authed(request):
        return auth_required_response()
    try:
        memory = get_memory()
        if by == "product":
            deleted = memory.clear_project(product_name=project_name)
        else:
            deleted = memory.clear_project(client_name=project_name)
        return {"success": True, "deleted": deleted}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Run app
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting AuraGTM at http://localhost:8000")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )