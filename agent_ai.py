from crewai import Agent, Task, Crew, Process,LLM
import os
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from pathlib import Path

os.environ["GROQ_API_KEY"]=os.getenv("api_key")
os.environ["COHERE_API_KEY"] = os.getenv("cohere_api_key")
# Configure custom storage location
project_root = Path(__file__).parent
# Use string paths for storage to avoid Path-specific methods being called
storage_dir = str(project_root / "crewai_storage")

os.environ["CREWAI_STORAGE_DIR"] = storage_dir

llm=LLM("groq/llama-3.1-8b-instant")


def chat_with_llm(prompt: str):

    agent1=Agent(
        llm=llm,
        goal="you know everything about user",
        role="about user",
        backstory="You are a master at understanding and answering user questions accurately and helpfully.",
        
    )

    tache=Task(
        agent=agent1,
        description=f"Answer the question : {prompt}",
        expected_output="""Provide a detailed and accurate answer to the user's question based on your knowledge and understanding.""",
    )

    crew=Crew(
        agents=[agent1],
        tasks=[tache],
        process=Process.sequential,
        memory=True,
        embedder={
        "provider": "cohere",
        "config":{
            "model": "embed-multilingual-v3.0",
    }
},
    long_term_memory=LongTermMemory(
        storage=LTMSQLiteStorage(
            db_path=os.path.join(storage_dir, "ltm_database.db")
        )
    ),
    short_term_memory=ShortTermMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "cohere",
                "config":{
                    "model": "embed-multilingual-v3.0",
                }
            },
            type="short_term",
            path=storage_dir
        )
    
    ),
    entity_memory=EntityMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "cohere",
                "config":{
                    "model": "embed-multilingual-v3.0",
                }
            },
            type="short_term",
            path=storage_dir
        )
    )
    )

    response=crew.kickoff(inputs={"prompt": prompt})
    print(response)

if __name__ == "__main__":
    prompt=input("Entrez votre question : ")
    while prompt.lower() not in ["exit", "quit"]:
        chat_with_llm(prompt)
        prompt=input("Entrez votre question : ")