"""
ChromaDB-based vector store for persisting and retrieving research findings.

Uses ChromaDB's native embedding function to store findings with metadata
(task, timestamp, summary) and enables semantic search across past discoveries.
"""

import os
from datetime import datetime

import chromadb

from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env
load_dotenv(find_dotenv())

# Define the persistent storage path
CHROMA_DB_PATH = "./memory/chroma_db"

# Ensure the directory exists
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Collection name
COLLECTION_NAME = "aria_findings"


def _get_or_create_collection():
    """
    Get or create the findings collection.
    
    Returns:
        chromadb.Collection: The collection for storing findings.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def _generate_summary(output: str, max_length: int = 100) -> str:
    """
    Generate a brief summary from the output.
    
    Args:
        output (str): The full finding output.
        max_length (int): Maximum length of the summary.
    
    Returns:
        str: A truncated summary of the output.
    """
    summary = output[:max_length].replace("\n", " ").strip()
    if len(output) > max_length:
        summary += "..."
    return summary


def save_finding(task: str, output: str) -> None:
    """
    Save a research finding to the vector store with metadata.
    
    Embeds the finding and stores it along with task name, timestamp,
    and a brief summary for later retrieval and context.
    
    Args:
        task (str): Name or description of the task.
        output (str): The research output or finding to store.
    
    Returns:
        None
    """
    try:
        collection = _get_or_create_collection()
        
        # Generate metadata
        timestamp = datetime.now().isoformat()
        summary = _generate_summary(output)
        
        # Create a unique document ID based on timestamp and task
        doc_id = f"{task}_{timestamp}".replace(" ", "_").replace(":", "-")
        
        # Add to collection with metadata
        collection.add(
            ids=[doc_id],
            documents=[output],
            metadatas=[{
                "task": task,
                "timestamp": timestamp,
                "summary": summary
            }]
        )
    
    except Exception as e:
        print(f"Error saving finding to ChromaDB: {str(e)}")


def retrieve_relevant(query: str, n: int = 3) -> str:
    """
    Retrieve the top n most relevant past findings for a query.
    
    Performs semantic search using the query and returns formatted results
    with task names, timestamps, and content summaries.
    
    Args:
        query (str): The search query.
        n (int): Number of top results to retrieve (default 3).
    
    Returns:
        str: Formatted string of relevant findings, or message if none found.
    """
    try:
        collection = _get_or_create_collection()
        
        # Check if collection has any documents
        count = collection.count()
        if count == 0:
            return "No past findings in memory yet."
        
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=min(n, count)
        )
        
        # Check if results exist
        if not results or not results.get("documents") or not results["documents"][0]:
            return f"No relevant findings found for query: '{query}'"
        
        # Format results
        formatted_output = f"Relevant Past Findings for '{query}':\n\n"
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if results.get("distances") else []
        
        for idx, (doc, metadata) in enumerate(zip(documents, metadatas), 1):
            task = metadata.get("task", "Unknown Task")
            timestamp = metadata.get("timestamp", "Unknown Time")
            summary = metadata.get("summary", "No summary available")
            
            # Include distance/relevance score if available
            relevance_note = ""
            if distances and idx <= len(distances):
                score = 1 - distances[idx - 1]  # Convert distance to similarity (cosine)
                relevance_note = f" (relevance: {score:.2f})"
            
            formatted_output += f"Finding {idx}:{relevance_note}\n"
            formatted_output += f"  Task: {task}\n"
            formatted_output += f"  Timestamp: {timestamp}\n"
            formatted_output += f"  Summary: {summary}\n"
            formatted_output += f"  Content: {doc[:150]}...\n\n"
        
        return formatted_output
    
    except Exception as e:
        error_msg = f"Error retrieving findings from ChromaDB: {str(e)}"
        return error_msg
