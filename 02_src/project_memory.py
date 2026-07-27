"""
AuraGTM — Long-Term Memory / Project History
==============================================
Task 6 of the team project.

WHAT IT DOES
------------
AuraGTM's engine is stateless: every call to generate_gtm_strategy() /
ask_aura() starts from zero, with no memory of what was generated for a
given client/product before. This module adds a persistent history layer:

- Every strategy/answer produced for a project (client + product) is saved
  to the database, with full inputs + outputs + timestamps.
- Before generating something new for the same project, the workflow can
  pull the most recent N entries and feed them back in as
  "PROJECT MEMORY" context, so the system builds on previous work
  instead of contradicting or repeating itself.
- A simple history API lets the UI list / inspect / clear a project's past
  outputs.

STORAGE
-------
Originally implemented as a standalone SQLite file (memory/project_history.db),
chosen for zero-setup simplicity since it required no new dependency. This
was migrated to PostgreSQL (the same database used by the rest of the app)
because the SQLite file did not persist correctly on hosting providers with
an ephemeral filesystem (Render's free tier wipes local files on every
redeploy or restart). The table is now `agent_workflow_memory`, defined in
models.py, and uses the same SessionLocal/engine as the rest of the app.

A "project" is identified by `project_key`:
    - client_name, if provided (Client Mode)      -> e.g. "BeamData"
    - otherwise a slug of the product name (General Mode) -> e.g. "smart-pos"
This means General Mode projects also get long-term memory, not just
Client Mode ones.

IMPORTANT — this is a SEPARATE memory store from the main `project_history`
table that powers the regular "History" panel in the app. That table logs
every generated strategy for the user to review. This table is only read
and written by the Strategist + Critic agent workflow, to let Agent 1 build
on its own previous work for the same project.
"""

import re
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database import SessionLocal
from models import AgentWorkflowMemory


def _slugify(text: str) -> str:
    text = (text or "untitled").strip().lower()
    text = re.sub(r"[^a-z0-9\u0600-\u06FF]+", "-", text)
    return text.strip("-") or "untitled"


def derive_project_key(client_name: Optional[str], product_name: Optional[str]) -> str:
    """Client Mode projects are keyed by client. General Mode projects are
    keyed by a slug of the product name, so repeat work on the same product
    still benefits from memory even without a client KB."""
    client_name = (client_name or "").strip()
    if client_name:
        return f"client::{_slugify(client_name)}"
    return f"product::{_slugify(product_name or 'general')}"


class ProjectMemory:
    """Persistence layer for long-term agent-workflow memory, backed by
    PostgreSQL (same database as the rest of the app)."""

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_entry(
        self,
        *,
        client_name: Optional[str],
        product_name: Optional[str],
        industry: Optional[str] = None,
        region: Optional[str] = None,
        business_goal: Optional[str] = None,
        mode: Optional[str] = None,
        record_type: str = "strategy",
        final_content: str,
        draft_content: Optional[str] = None,
        critique_notes: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        project_key = derive_project_key(client_name, product_name)

        db = SessionLocal()
        try:
            entry = AgentWorkflowMemory(
                project_key=project_key,
                client_name=client_name,
                product_name=product_name,
                industry=industry,
                region=region,
                business_goal=business_goal,
                mode=mode,
                record_type=record_type,
                draft_content=draft_content,
                critique_notes=critique_notes,
                final_content=final_content,
                sources_json=json.dumps(sources or [], ensure_ascii=False),
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            return entry.id
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_history(
        self,
        client_name: Optional[str] = None,
        product_name: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Full history rows for a project, most recent first."""
        project_key = derive_project_key(client_name, product_name)

        db = SessionLocal()
        try:
            rows = (
                db.query(AgentWorkflowMemory)
                .filter(AgentWorkflowMemory.project_key == project_key)
                .order_by(AgentWorkflowMemory.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "project_key": r.project_key,
                    "client_name": r.client_name,
                    "product_name": r.product_name,
                    "industry": r.industry,
                    "region": r.region,
                    "business_goal": r.business_goal,
                    "mode": r.mode,
                    "record_type": r.record_type,
                    "draft_content": r.draft_content,
                    "critique_notes": r.critique_notes,
                    "final_content": r.final_content,
                    "sources_json": r.sources_json,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
        finally:
            db.close()

    def list_projects(self) -> List[Dict[str, Any]]:
        """All distinct projects that have at least one memory entry."""
        db = SessionLocal()
        try:
            rows = db.query(AgentWorkflowMemory).all()
            grouped: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                key = r.project_key
                if key not in grouped:
                    grouped[key] = {
                        "project_key": key,
                        "client_name": r.client_name or "",
                        "product_name": r.product_name or "",
                        "entries": 0,
                        "last_updated": r.created_at,
                    }
                grouped[key]["entries"] += 1
                if r.created_at and r.created_at > grouped[key]["last_updated"]:
                    grouped[key]["last_updated"] = r.created_at

            result = list(grouped.values())
            result.sort(key=lambda p: p["last_updated"] or datetime.min, reverse=True)
            for p in result:
                p["last_updated"] = p["last_updated"].isoformat() if p["last_updated"] else None
            return result
        finally:
            db.close()

    def get_memory_context(
        self,
        client_name: Optional[str] = None,
        product_name: Optional[str] = None,
        limit: int = 3,
        max_chars_each: int = 1200,
    ) -> str:
        """
        Build a compact text block summarizing the last `limit` outputs for
        this project, suitable for injecting into a prompt as long-term
        memory. Returns "" if there is no history yet (first run).
        """
        history = self.get_history(client_name, product_name, limit=limit)
        if not history:
            return ""

        blocks = []
        for row in reversed(history):  # oldest -> newest, reads more naturally
            when = (row["created_at"] or "")[:10]
            kind = row["record_type"]
            content = (row["final_content"] or "")[:max_chars_each]
            blocks.append(f"[Previous {kind} — {when}]\n{content}")
        return "\n\n---\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_project(self, client_name: Optional[str] = None, product_name: Optional[str] = None) -> int:
        project_key = derive_project_key(client_name, product_name)

        db = SessionLocal()
        try:
            deleted = (
                db.query(AgentWorkflowMemory)
                .filter(AgentWorkflowMemory.project_key == project_key)
                .delete()
            )
            db.commit()
            return deleted
        finally:
            db.close()


_memory: Optional[ProjectMemory] = None


def get_memory() -> ProjectMemory:
    """Singleton accessor, mirrors the `get_engine()` pattern already used
    in auragtm_engine.py so the rest of the codebase stays consistent."""
    global _memory
    if _memory is None:
        _memory = ProjectMemory()
    return _memory