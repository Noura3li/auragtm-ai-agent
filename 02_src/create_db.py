from dotenv import load_dotenv
load_dotenv()

from database import engine
from models import Base

print("Creating AuraGTM tables...")

Base.metadata.create_all(bind=engine)

print("Tables created successfully!")