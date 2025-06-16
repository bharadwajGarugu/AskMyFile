import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.database import get_db
from src.utils.logger import get_logger

logger = get_logger("test_connection")

def test_connection():
    try:
        db = next(get_db())
        logger.info("Successfully connected to the database!")
        
        # Test query to check tables
        result = db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = [row[0] for row in result]
        logger.info(f"Available tables: {tables}")
        
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    test_connection() 