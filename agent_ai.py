from crewai import Agent, Task, Crew, Process,LLM
import os

from pathlib import Path

from memory_config import (long_term_memory, 
                           short_term_memory, 
                           entity_memory, 
                           embedder_cohere)

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
        goal="Tu connais la reponse a toutes les questions. Et garde toujours le nom que l'utilisateur te donne en memoire et les conversations precedentes.",
        role="Expert en reponses aux questions",
        backstory="""Tu es un agent intelligent capable de repondre a toutes 
                les questions posees par les utilisateurs de maniere precise et 
                detaillee et tu te bases aussi sur les conversations en memoire.
                Garde toujours le nom que l'utilisateur te donne en memoire.""",
        
    )

    tache=Task(
        agent=agent1,
        description=f"Repondre a la question : {prompt}",
        expected_output="""Fournir une reponse detaillee et precise a la question de l'utilisateur
                    en te basant sur vos connaissances et votre comprehension. 
                    Garde toujours le nom de l'utilisateur en memoire et les conversations 
                    precedentes en te basant sur vos connaissances et votre comprehension.""",
    )

    crew=Crew(
        agents=[agent1],
        tasks=[tache],
        process=Process.sequential,
        memory=True,
        embedder=embedder_cohere,
        long_term_memory=long_term_memory,
        short_term_memory=short_term_memory,
        entity_memory=entity_memory
    )

    response=crew.kickoff(inputs={"prompt": prompt})
    print(response)

if __name__ == "__main__":
    prompt=input("🤩 Entrez votre question : ")
    while prompt.lower() not in ["exit", "quit"]:
        chat_with_llm(prompt)
        prompt=input("🤩 Entrez votre question : ")