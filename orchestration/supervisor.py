"""
LangGraph Supervisor Agent - Orchestration Layer
Routes and coordinates between Disease AI, Crop Planner, and AI Expert services
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
import operator
from services import classify_disease, parse_label, generate_cultivation_plan, answer_farming_question
from src.generator import generate_treatment


# Define the state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next: str


# Define tools for each service
@tool
def diagnose_disease_tool(image_path: str, user_context: str = "") -> str:
    """
    Diagnose crop disease from leaf image.
    
    Args:
        image_path: Path to the leaf image
        user_context: Additional context about the farm/location
        
    Returns:
        Diagnosis and treatment information
    """
    result = classify_disease(image_path)
    top = result["top_prediction"]
    crop, disease = parse_label(top["label"])
    confidence = top["confidence"]
    
    if disease.lower() == "healthy":
        diagnosis = (
            f"Crop: {crop}\n"
            f"Status: Healthy\n"
            f"Confidence: {confidence*100:.1f}%\n"
            "No disease detected."
        )
        treatment = "No treatment needed."
    else:
        alternatives = result["alternatives"]
        diagnosis = (
            f"Crop: {crop}\n"
            f"Disease: {disease}\n"
            f"Confidence: {confidence*100:.1f}%\n"
            f"Alternatives: {', '.join([parse_label(a['label'])[1] for a in alternatives])}"
        )
        treatment = generate_treatment(crop, disease, confidence, user_context)
    
    return f"DIAGNOSIS:\n{diagnosis}\n\nTREATMENT:\n{treatment}"


@tool
def plan_cultivation_tool(crop: str, region: str = "", season: str = "", soil_type: str = "", user_context: str = "") -> str:
    """
    Generate cultivation plan for a crop.
    
    Args:
        crop: Name of the crop
        region: Geographic region
        season: Growing season
        soil_type: Type of soil
        user_context: Additional context
        
    Returns:
        Comprehensive cultivation plan
    """
    plan = generate_cultivation_plan(crop, region, season, soil_type, user_context)
    return f"CULTIVATION PLAN FOR {crop.upper()}:\n\n{plan}"


@tool
def answer_farming_question_tool(question: str) -> str:
    """
    Answer farming-related questions using RAG.
    
    Args:
        question: Farmer's question
        
    Returns:
        Answer based on knowledge base
    """
    answer = answer_farming_question(question)
    return f"ANSWER:\n\n{answer}"


# Create tool node
tools = [diagnose_disease_tool, plan_cultivation_tool, answer_farming_question_tool]
tool_node = ToolNode(tools)


# Router function to determine which service to use
def route_request(state: AgentState) -> Literal["disease_ai", "crop_planner", "ai_expert", "end"]:
    """
    Route the request to the appropriate service based on user intent
    """
    last_message = state["messages"][-1]
    
    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()
        
        # Check for disease diagnosis intent
        if any(keyword in content for keyword in ["disease", "diagnose", "sick", "infected", "symptom", "leaf", "plant problem"]):
            return "disease_ai"
        
        # Check for crop planning intent
        elif any(keyword in content for keyword in ["plan", "cultivation", "grow", "planting", "sowing", "harvest", "yield"]):
            return "crop_planner"
        
        # Default to Q&A for general questions
        else:
            return "ai_expert"
    
    return "end"


# Disease AI node
def disease_ai_node(state: AgentState):
    """
    Handle disease diagnosis requests
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Extract image path and context from message
    # This is a simplified version - in production, you'd parse more carefully
    response = diagnose_disease_tool.invoke({"image_path": "", "user_context": ""})
    
    return {"messages": [AIMessage(content=response)]}


# Crop planner node
def crop_planner_node(state: AgentState):
    """
    Handle crop planning requests
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Extract parameters from message
    # This is a simplified version
    response = plan_cultivation_tool.invoke({
        "crop": "rice",  # Would extract from message
        "region": "",
        "season": "",
        "soil_type": "",
        "user_context": ""
    })
    
    return {"messages": [AIMessage(content=response)]}


# AI expert node
def ai_expert_node(state: AgentState):
    """
    Handle Q&A requests
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    response = answer_farming_question_tool.invoke({"question": last_message.content})
    
    return {"messages": [AIMessage(content=response)]}


# Build the graph
def build_supervisor_graph():
    """
    Build the LangGraph supervisor agent
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("disease_ai", disease_ai_node)
    workflow.add_node("crop_planner", crop_planner_node)
    workflow.add_node("ai_expert", ai_expert_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        route_request,
        {
            "disease_ai": "disease_ai",
            "crop_planner": "crop_planner",
            "ai_expert": "ai_expert",
            "end": END
        }
    )
    
    # Add edges from service nodes to end
    workflow.add_edge("disease_ai", END)
    workflow.add_edge("crop_planner", END)
    workflow.add_edge("ai_expert", END)
    
    return workflow.compile()


# Simplified supervisor for direct API usage
class SupervisorAgent:
    """
    Simplified supervisor agent that routes requests to appropriate services
    """
    
    def __init__(self):
        pass
    
    def process_request(self, request_type: str, **kwargs) -> str:
        """
        Process a request and route to appropriate service
        
        Args:
            request_type: Type of request ("diagnose", "plan", "qa")
            **kwargs: Arguments for the specific service
            
        Returns:
            Response from the appropriate service
        """
        if request_type == "diagnose":
            image_path = kwargs.get("image_path")
            user_context = kwargs.get("user_context", "")
            return diagnose_disease_tool.invoke({"image_path": image_path, "user_context": user_context})
        
        elif request_type == "plan":
            crop = kwargs.get("crop", "")
            region = kwargs.get("region", "")
            season = kwargs.get("season", "")
            soil_type = kwargs.get("soil_type", "")
            user_context = kwargs.get("user_context", "")
            return plan_cultivation_tool.invoke({
                "crop": crop,
                "region": region,
                "season": season,
                "soil_type": soil_type,
                "user_context": user_context
            })
        
        elif request_type == "qa":
            question = kwargs.get("question", "")
            return answer_farming_question_tool.invoke({"question": question})
        
        else:
            return "Invalid request type. Use 'diagnose', 'plan', or 'qa'."


# Create supervisor instance
supervisor = SupervisorAgent()
