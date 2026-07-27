from database import SessionLocal
from models import User

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from sqlalchemy import or_, func


def create_user(username, email, password):

    db = SessionLocal()

    try:
        username = username.strip()
        email = email.strip().lower()

        if len(username) < 2:
            return False, "Username is too short"

        if "@" not in email or "." not in email:
            return False, "Invalid email"

        if len(password) < 8:
            return False, "Password must be at least 8 characters"

        existing = (
            db.query(User)
            .filter(
                or_(
                    User.email == email,
                    func.lower(User.username) == username.lower()
                )
            )
            .first()
        )

        if existing:
            return False, "Email or username already exists"

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )

        db.add(user)
        db.commit()

        return True, "User created"

    except Exception as e:
        db.rollback()
        return False, str(e)

    finally:
        db.close()


def login_user(identifier, password):

    db = SessionLocal()

    try:
        identifier = identifier.strip()

        user = (
            db.query(User)
            .filter(
                or_(
                    User.email == identifier.lower(),
                    func.lower(User.username) == identifier.lower()
                )
            )
            .first()
        )

        if not user:
            return None

        if not check_password_hash(user.password_hash, password):
            return None

        return user

    finally:
        db.close()