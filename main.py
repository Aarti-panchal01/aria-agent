"""
Entry point for ARIA (Agent Research Intelligence Agent).

Prompts user for a research goal and runs the complete research workflow
through the LangGraph agent, printing outputs and saving artifacts.
"""

from graph import aria_graph
from state import AgentState


def main():
    """
    Main entry point for ARIA research agent.
    
    Collects research goal from user, initializes state, and runs
    the complete research workflow through the compiled graph.
    """
    print("=" * 60)
    print("ARIA - Agent Research Intelligence Agent")
    print("=" * 60)
    print()
    
    # Get research goal from user
    goal = input("Enter your research goal: ").strip()
    
    if not goal:
        print("Error: Research goal cannot be empty.")
        return
    
    print(f"\n🎯 Starting research on: {goal}")
    print("=" * 60)
    
    # Build initial state
    initial_state: AgentState = {
        "goal": goal,
        "subtasks": [],
        "current_task_index": 0,
        "results": [],
        "memory_context": "",
        "final_report": "",
        "replan_count": 0,
        "is_done": False
    }
    
    # Run the graph
    print("\n⏳ Executing research workflow...\n")
    try:
        final_state = aria_graph.invoke(initial_state)
        
        # Extract final report
        final_report = final_state.get("final_report", "")
        
        # Print final report
        print("\n" + "=" * 60)
        print("📊 RESEARCH REPORT")
        print("=" * 60)
        print(final_report)
        print("\n" + "=" * 60)
        
        # Print artifact locations
        print("\n✅ Research complete!")
        print(f"📄 Report saved to: ./output/report.md")
        print(f"📋 Full reasoning trace saved to: ./output/reasoning_trace.json")
        
    except Exception as e:
        print(f"\n❌ Error during research execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
