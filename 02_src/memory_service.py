import re

from database import SessionLocal
from models import ProjectHistory


def extract_recommended_strategy(strategy_text):

    if not strategy_text:
        return None

    patterns = [
        r"Choice:\s*(Option\s*[ABC])",
        r"Recommended Option\s*[:\-]?\s*(Option\s*[ABC])",
        r"Recommended Strategy\s*[:\-]?\s*(Option\s*[ABC])",
        r"Recommendation\s*[:\-]?\s*(Option\s*[ABC])"
    ]

    for pattern in patterns:
        match = re.search(pattern, strategy_text, re.IGNORECASE)

        if match:
            option_text = match.group(1)
            letter_match = re.search(r"[ABC]", option_text, re.IGNORECASE)

            if letter_match:
                return f"Option {letter_match.group(0).upper()}"

    return None


def save_strategy(inputs, strategy_text, user_id=None, recommended_strategy=None):

    db = SessionLocal()

    try:
        query = db.query(ProjectHistory).filter(
            ProjectHistory.project_name == inputs["product_name"]
        )

        if user_id is not None:
            query = query.filter(ProjectHistory.user_id == user_id)

        existing_count = query.count()
        next_version = f"v{existing_count + 1}"

        if recommended_strategy is None:
            recommended_strategy = extract_recommended_strategy(strategy_text)

        project = ProjectHistory(
            user_id=user_id,

            project_name=inputs["product_name"],
            client_name=inputs.get("client_name", ""),

            product_name=inputs["product_name"],
            product_description=inputs["product_description"],

            industry=inputs["industry"],
            region=inputs["region"],

            business_goal=inputs["business_goal"],
            brand_tone=inputs["brand_tone"],

            strategy_version=next_version,
            recommended_strategy=recommended_strategy,

            strategy_output=strategy_text
        )

        db.add(project)
        db.commit()
        db.refresh(project)

        return project.id

    except Exception as e:
        db.rollback()
        print(f"Database Save Error: {e}")
        return None

    finally:
        db.close()


def get_user_history(user_id):

    db = SessionLocal()

    try:
        history = (
            db.query(ProjectHistory)
            .filter(ProjectHistory.user_id == user_id)
            .order_by(ProjectHistory.created_at.desc())
            .all()
        )

        return history

    finally:
        db.close()


def get_site_stats():
    """
    Public, anonymous stats — aggregate totals across ALL users.
    No individual user data, no per-user breakdowns. Safe to show on a public page.
    """

    db = SessionLocal()

    try:
        total_strategies = db.query(ProjectHistory).count()

        industries = (
            db.query(ProjectHistory.industry)
            .filter(ProjectHistory.industry.isnot(None))
            .filter(ProjectHistory.industry != "")
            .distinct()
            .count()
        )

        clients_served = (
            db.query(ProjectHistory.client_name)
            .filter(ProjectHistory.client_name.isnot(None))
            .filter(ProjectHistory.client_name != "")
            .distinct()
            .count()
        )

        return {
            "total_strategies": total_strategies,
            "industries_covered": industries,
            "clients_served": clients_served,
        }

    finally:
        db.close()