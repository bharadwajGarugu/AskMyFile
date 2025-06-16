from database.database import engine, SessionLocal
from database.models import Base, User, FAQ
from sqlalchemy import text
import os
from dotenv import load_dotenv
from utils.logger import get_logger

# Initialize logger
logger = get_logger("test_setup")

def test_database_connection():
    try:
        logger.info("Starting database connection test")
        
        # Test database connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.success("✅ Database connection successful!")
            
        # Test table creation
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.success("✅ Tables created successfully!")
        
        # Test user creation
        logger.info("Testing user creation...")
        db = SessionLocal()
        test_user = User(
            username="test_user",
            email="test@example.com",
            hashed_password="test_password"  # In real app, this would be hashed
        )
        db.add(test_user)
        db.commit()
        logger.success("✅ Test user created successfully!")
        
        # Test FAQ creation
        logger.info("Testing FAQ creation...")
        test_faq = FAQ(
            question="What is FAQBot?",
            answer="FAQBot is an intelligent FAQ system powered by AI.",
            category="General",
            author_id=test_user.id
        )
        db.add(test_faq)
        db.commit()
        logger.success("✅ Test FAQ created successfully!")
        
        # Clean up test data
        logger.info("Cleaning up test data...")
        db.delete(test_faq)
        db.delete(test_user)
        db.commit()
        logger.success("✅ Test data cleaned up successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during test setup: {str(e)}")
        raise
    finally:
        db.close()
        logger.info("Database session closed")

if __name__ == "__main__":
    logger.info("Starting FAQBot setup test...")
    test_database_connection()
    logger.success("Setup test completed successfully!") 