"""
State definition for ARIA (Agent Research Intelligence Agent).

This module defines the AgentState TypedDict that tracks all information
flowing through the LangGraph research agent, including the research goal,
task breakdown, execution progress, memory context, and final outputs.
"""

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    """
    State schema for the ARIA research agent.
    
    Attributes:
        goal (str):
            The original research objective that the agent is working to fulfill.
            Set at initialization and never modified.
        
        subtasks (list[str]):
            The research plan broken down into actionable steps.
            Created during planning phase, may be regenerated if replanning occurs.
        
        current_task_index (int):
            Index indicating which subtask the agent is currently executing.
            Incremented as tasks are completed successfully.
        
        results (list[dict]):
            Cumulative list of completed task results. Each entry is a dictionary
            containing 'task' (str), 'output' (str), and 'score' (float).
            Uses operator.add for append operations to accumulate findings.
        
        memory_context (str):
            Retrieved past findings or relevant context injected before planning.
            Helps maintain consistency across long research sessions and avoids
            redundant work by referencing previous discoveries.
        
        final_report (str):
            The markdown-formatted research report generated at the end.
            Synthesizes all results into a cohesive, human-readable output.
        
        replan_count (int):
            Counter tracking how many times the agent has replanned the subtasks.
            Capped at 3 to prevent infinite replanning loops.
        
        is_done (bool):
            Termination flag indicating whether the agent has completed
            the research objective and should exit the graph.
    """
    
    goal: str
    subtasks: list[str]
    current_task_index: int
    results: Annotated[list[dict], operator.add]
    memory_context: str
    final_report: str
    replan_count: int
    is_done: bool
