from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from pathlib import Path
import os

# Configure custom storage location
project_root = Path(__file__).parent
# Use string paths for storage to avoid Path-specific methods being called
storage_dir = str(project_root / "crewai_storage")

embedder_cohere={
                "provider": "cohere",
                "config":{
                    "model": "embed-multilingual-v3.0",
                }
            }

long_term_memory=LongTermMemory(
        storage=LTMSQLiteStorage(
            db_path=os.path.join(storage_dir, "ltm_database.db")
        )
    )

short_term_memory=ShortTermMemory(
        storage=RAGStorage(
            embedder_config=embedder_cohere,
            type="short_term",
            path=storage_dir
        )
    
    )

entity_memory=EntityMemory(
        storage=RAGStorage(
            embedder_config=embedder_cohere,
            type="short_term",
            path=storage_dir
        )
    )