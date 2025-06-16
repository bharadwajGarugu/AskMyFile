import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from ..utils.logger import get_logger

# Initialize logger
logger = get_logger("database")

# Load environment variables
load_dotenv()
logger.info("Loading environment variables")

# Get database credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_DB = os.getenv("SUPABASE_DB")
SUPABASE_USER = os.getenv("SUPABASE_USER")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

# Construct database URL
DATABASE_URL = f"postgresql://{SUPABASE_USER}:{SUPABASE_PASSWORD}@{SUPABASE_URL}/{SUPABASE_DB}"
logger.info(f"Database URL constructed for {SUPABASE_DB}")

try:
    # Create database engine
    engine = create_engine(DATABASE_URL)
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {str(e)}")
    raise

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
logger.info("Database session maker configured")

# Create base class for models
Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        logger.debug("Database session started")
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        logger.debug("Database session closed")
        db.close() 