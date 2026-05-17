"""
LangGraph StateGraph for the ARIA research agent.

Wires all cognitive nodes (planner, executor, critic, memory, terminator, reporter)
into a complete agentic research loop with conditional routing based on quality scores
and completion status.
"""

from langgraph.graph import StateGraph, END

from state import AgentState
from nodes.memory_reader import memory_reader_node
from nodes.planner import planner_node
from nodes.executor import executor_node
from nodes.critic import critic_node
from nodes.memory_writer import memory_writer_node
from nodes.terminator import terminator_node
from nodes.report_generator import report_generator_node


def _terminator_route(state: AgentState) -> str:
    """
    Conditional edge router from terminator node.
    
    Routes based on:
    - is_done == True → "report_generator" (all tasks done or max replans reached)
    - is_done == False AND last score < 7 → "planner" (replan with better strategy)
    - is_done == False AND last score >= 7 → "executor" (move to next task)
    
    Args:
        state (AgentState): Current agent state.
    
    Returns:
        str: Node name to route to.
    """
    is_done = state.get("is_done", False)
    
    if is_done:
        # All tasks completed or max replans reached
        return "report_generator"
    
    # Not done: check last result's quality score
    results = state.get("results", [])
    if results:
        last_score = results[-1].get("score", 5)
        
        if last_score < 7:
            # Low quality: replan
            return "planner"
        else:
            # Good quality: continue to next task
            return "executor"
    
    # Default: continue to executor if no results yet
    return "executor"


def build_graph() -> StateGraph:
    """
    Build and compile the ARIA research agent graph.
    
    Constructs a StateGraph with all cognitive nodes and conditional routing
    for quality-driven research execution and replanning.
    
    Returns:
        StateGraph: Compiled LangGraph StateGraph ready for invocation.
    """
    # Initialize StateGraph
    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("memory_reader", memory_reader_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("memory_writer", memory_writer_node)
    graph.add_node("terminator", terminator_node)
    graph.add_node("report_generator", report_generator_node)
    
    # Set entry point
    graph.set_entry_point("memory_reader")
    
    # Add linear edges for main execution flow
    graph.add_edge("memory_reader", "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "critic")
    graph.add_edge("critic", "memory_writer")
    graph.add_edge("memory_writer", "terminator")
    
    # Add conditional edge from terminator
    # Routes based on completion status and result quality
    graph.add_conditional_edges(
        "terminator",
        _terminator_route,
        {
            "report_generator": "report_generator",
            "planner": "planner",
            "executor": "executor"
        }
    )
    
    # Add edge from report generator to END
    graph.add_edge("report_generator", END)
    
    # Compile and return
    return graph.compile()


# Export the compiled graph
aria_graph = build_graph()
