import os
import json
import logging
from typing import TypedDict, Annotated, List, Dict, Any
from pathlib import Path

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

load_dotenv()
logger = logging.getLogger(__name__)

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- Tool Definitions ---

@tool
def pageindex_navigate(query: str) -> str:
    """
    Search the document hierarchy for relevant sections using the page index.
    The page index contains summaries and key entities for document sections.
    Use this tool to find which pages/sections might contain the answer before doing a semantic search.
    
    Args:
        query: A string representing the topic or entity to look for.
        
    Returns:
        A JSON string containing valid sections with their page numbers, summaries, and key entities.
    """
    index_dir = Path(".refinery/pageindex")
    if not index_dir.exists():
        return "Error: Page index directory not found. Please run the indexer first."

    # Load all index files
    all_sections = []
    for index_file in index_dir.glob("*_index.json"):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                sections = json.load(f)
                all_sections.extend(sections)
        except Exception as e:
            logger.error(f"Failed to load index file {index_file}: {e}")
            continue
            
    if not all_sections:
        return "Error: No sections found in the page index."

    # Simple search implementation (can be improved with fuzzy matching or embeddings later)
    # For now, we filter by simple casing-insensitive string matching in title, summary, or entities
    query_lower = query.lower()
    matches = []
    
    for section in all_sections:
        title = section.get("title", "").lower()
        summary = section.get("summary", "").lower()
        entities = [e.lower() for e in section.get("key_entities", [])]
        
        score = 0
        if query_lower in title:
            score += 3
        if query_lower in summary:
            score += 1
        for entity in entities:
            if query_lower in entity:
                score += 2
                
        if score > 0:
            matches.append((score, section))
            
    # Sort by relevance score
    matches.sort(key=lambda x: x[0], reverse=True)
    
    # Return top 5 matches
    top_matches = [m[1] for m in matches[:5]]
    return json.dumps(top_matches, indent=2)

@tool
def semantic_search(query: str) -> str:
    """
    Retrieve specific text chunks from the vector database based on semantic similarity.
    Use this tool when you need precise details from the text that might be missing in the high-level page index.
    """
    persist_directory = ".refinery/chroma_db"
    
    if not os.path.exists(persist_directory):
         return "Error: Vector database not found at .refinery/chroma_db. Has the indexing (Stage 3) been run?"

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment."

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key
        )
        
        vectorstore = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )
        
        results = vectorstore.similarity_search(query, k=4)
        
        output = []
        for doc in results:
            # FDE FIX: Safely grab all metadata, falling back to "unknown" if missing
            source = doc.metadata.get("source", "tax_expenditure_ethiopia") 
            page = doc.metadata.get("page_refs", doc.metadata.get("page", "unknown"))
            bbox = doc.metadata.get("bounding_box", "unknown")
            section = doc.metadata.get("parent_section", "unknown")
            
            output.append(f"--- Source: {source} | Section: {section} | Page: {page} | BBox: {bbox} ---\n{doc.page_content}\n")
            
        return "\n".join(output) if output else "No relevant chunks found."
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return f"Error performing semantic search: {str(e)}"

@tool
def structured_query(query: str) -> str:
    """
    Query structured data extracted from tables or forms.
    Use this tool when looking for specific data points that might be in tabular format.
    
    Args:
        query: The query describing the data to look for.
        
    Returns:
        JSON string of results.
    """
    # Placeholder implementation
    # In a full system, this would query a SQL DB or specialized JSON store for tables
    return "Structured query is not typically implemented in this prototype yet. Please rely on page index and semantic search."

# --- Agent Construction ---
from langchain_core.messages import SystemMessage # ADD THIS IMPORT AT THE TOP OF YOUR FILE

def create_agent():
    """
    Constructs and compiles the LangGraph agent with the defined tools.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
        max_retries=2
    )
    
    tools = [pageindex_navigate, semantic_search, structured_query]
    llm_with_tools = llm.bind_tools(tools)
    
    # FDE FIX: Define the strict Provenance Prompt
    system_prompt = """You are an expert FDE AI. You must answer the user's questions using ONLY the provided tools and context.

CRITICAL INSTRUCTION: You MUST explicitly include provenance for EVERY claim in your final answer.
Your final answer MUST end with a citation in this exact format:
[Doc: <document_name>, Page: <page_number>, BBox: <bounding_box>]

Example:
The expenditure was 5 million. [Doc: tax_report, Page: [5], BBox: [100.0, 200.0, 300.0, 400.0]]

If you do not include this citation format, you have failed your task."""
    
    def chatbot(state: AgentState):
        messages = state["messages"]
        
        # FDE FIX: Inject the SystemMessage if it isn't already there
        if not messages or getattr(messages[0], "type", "") != "system":
            messages = [SystemMessage(content=system_prompt)] + messages

        # Sanitize messages: Ensure content is never None
        for msg in messages:
            if hasattr(msg, 'content') and msg.content is None:
                msg.content = ""

        return {"messages": [llm_with_tools.invoke(messages)]}
        
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.add_conditional_edges(
        "chatbot",
        lambda state: "tools" if state["messages"][-1].tool_calls else "__end__",
        {"tools": "tools", "__end__": END}
    )
    
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    
    return graph_builder.compile()

if __name__ == "__main__":
    # Simple test loop
    agent = create_agent()
    print("Agent initialized. Type 'quit' to exit.")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        events = agent.stream(
            {"messages": [("user", user_input)]},
            stream_mode="values"
        )
        
        for event in events:
            if "messages" in event:
                message = event["messages"][-1]
                if hasattr(message, "content") and message.content:
                     print(f"Agent: {message.content}")

