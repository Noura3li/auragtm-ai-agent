from sqlalchemy.orm import declarative_base
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey
)
from datetime import datetime

Base = declarative_base()


# ==================================================
# USERS
# ==================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String(100), nullable=False)

    email = Column(
        String(255),
        unique=True,
        nullable=False
    )

    password_hash = Column(
        String(255),
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# ==================================================
# PROJECT HISTORY
# ==================================================

class ProjectHistory(Base):
    __tablename__ = "project_history"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    project_name = Column(String(255))
    client_name = Column(String(255))

    product_name = Column(String(255))
    product_description = Column(Text)

    industry = Column(String(255))
    region = Column(String(255))

    business_goal = Column(Text)
    brand_tone = Column(String(255))

    strategy_version = Column(String(50))

    recommended_strategy = Column(String(50))

    strategy_output = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)


# ==================================================
# PROJECT MEMORY
# ==================================================

class ProjectMemory(Base):
    __tablename__ = "project_memory"

    id = Column(Integer, primary_key=True, index=True)

    project_id = Column(
        Integer,
        ForeignKey("project_history.id")
    )

    memory_type = Column(String(100))

    memory_content = Column(Text)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# ==================================================
# AGENT WORKFLOW MEMORY (Strategist + Critic long-term memory)
# Migrated from a standalone SQLite file to PostgreSQL so it
# persists correctly on hosting providers with an ephemeral
# filesystem (e.g. Render's free tier wipes local files on redeploy).
# ==================================================

class AgentWorkflowMemory(Base):
    __tablename__ = "agent_workflow_memory"

    id = Column(Integer, primary_key=True, index=True)

    project_key = Column(String(255), nullable=False, index=True)

    client_name = Column(String(255))
    product_name = Column(String(255))

    industry = Column(String(255))
    region = Column(String(255))
    business_goal = Column(Text)
    mode = Column(String(50))

    record_type = Column(String(50))  # 'strategy' | 'gaps' | 'options' | 'answer' | 'agent_workflow'

    draft_content = Column(Text)
    critique_notes = Column(Text)
    final_content = Column(Text, nullable=False)

    sources_json = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)