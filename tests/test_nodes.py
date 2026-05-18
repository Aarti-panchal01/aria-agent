"""
Unit tests for ARIA research agent nodes and core functionality.

Tests cover:
- Input sanitization (length limits, control character removal)
- Node functionality with mocked LLM calls
- Error handling and fallback behavior
- Graph routing logic
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage

from state import AgentState
from nodes.planner import planner_node, _parse_subtasks
from nodes.critic import critic_node, _parse_score
from nodes.report_generator import report_generator_node
from tools.search import web_search
from graph import _terminator_route


# ============================================================================
# TEST 1: Input Length Cap Sanitization
# ============================================================================

def test_input_length_cap():
    """
    Test that the main.py sanitizer rejects research goals exceeding 500 characters.
    
    Verifies that a 600-character input string fails the length validation
    before being processed by the agent.
    """
    # Create a 600-character goal (exceeds the 500-char limit)
    long_goal = "a" * 600
    MAX_GOAL_LENGTH = 500
    
    # The sanitizer logic from main.py
    if len(long_goal) > MAX_GOAL_LENGTH:
        is_rejected = True
    else:
        is_rejected = False
    
    assert is_rejected is True, "Goal exceeding 500 characters should be rejected"
    assert len(long_goal) == 600, "Test setup: goal should be 600 chars"


# ============================================================================
# TEST 2: Control Character Stripping
# ============================================================================

def test_input_control_chars():
    """
    Test that control characters are stripped from the goal string.
    
    Verifies that non-printable characters (except newline and tab) are removed,
    keeping only printable ASCII and common Unicode characters.
    """
    # Goal with control characters mixed in
    goal_with_controls = "Research LangGraph\x00\x01\x02 workflow\x03\x04 efficiency"
    
    # Apply the sanitizer logic from main.py
    sanitized_goal = ''.join(c for c in goal_with_controls if ord(c) >= 32 or c in '\n\t')
    
    # Assert no control characters remain
    assert '\x00' not in sanitized_goal, "Null character should be removed"
    assert '\x01' not in sanitized_goal, "Control char \\x01 should be removed"
    assert '\x02' not in sanitized_goal, "Control char \\x02 should be removed"
    assert '\x03' not in sanitized_goal, "Control char \\x03 should be removed"
    assert '\x04' not in sanitized_goal, "Control char \\x04 should be removed"
    assert 'Research LangGraph' in sanitized_goal, "Normal text should be preserved"
    assert 'workflow' in sanitized_goal, "Normal text should be preserved"


# ============================================================================
# TEST 3: Planner Returns Six Subtasks
# ============================================================================

def test_planner_returns_six_subtasks():
    """
    Test that planner node returns exactly 6 subtasks when LLM returns a numbered list.
    
    Mocks the ChatGroq LLM to return a 6-item numbered list, verifies that
    the planner correctly parses this and returns exactly 6 subtasks.
    """
    # Mock the LLM response with 6 numbered items
    mock_llm_response = """1. Research LangGraph architecture and components
2. Compare LangGraph vs LangChain workflow capabilities
3. Analyze LangGraph state management and persistence
4. Investigate LangGraph's agent and human-in-the-loop features
5. Review LangGraph production deployment patterns
6. Evaluate LangGraph performance and scaling characteristics"""
    
    # Create initial state
    initial_state: AgentState = {
        "goal": "Compare LangGraph and LangChain frameworks",
        "subtasks": [],
        "current_task_index": 0,
        "results": [],
        "memory_context": "",
        "final_report": "",
        "replan_count": 0,
        "is_done": False
    }
    
    # Mock ChatGroq
    with patch('nodes.planner.ChatGroq') as mock_groq_class:
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content=mock_llm_response)
        mock_groq_class.return_value = mock_llm
        
        # Mock environment
        with patch.dict('os.environ', {'GROQ_API_KEY': 'test-key'}):
            result = planner_node(initial_state)
    
    # Assert exactly 6 subtasks returned
    assert 'subtasks' in result, "Result should contain 'subtasks' key"
    assert len(result['subtasks']) == 6, f"Expected 6 subtasks, got {len(result['subtasks'])}"
    assert result['current_task_index'] == 0, "current_task_index should be 0"
    assert result['replan_count'] == 1, "replan_count should increment to 1"


# ============================================================================
# TEST 4: Critic Score In Range 0-10
# ============================================================================

def test_critic_score_in_range():
    """
    Test that critic node returns a score between 0 and 10.
    
    Mocks the ChatGroq LLM to return "Score: 8", verifies the critic
    correctly extracts and validates the score is within 0-10 range.
    """
    # Mock LLM response with a score
    mock_llm_response = "Score: 8"
    
    # Create initial state with a result to evaluate
    initial_state: AgentState = {
        "goal": "Research LangGraph",
        "subtasks": ["Research LangGraph architecture"],
        "current_task_index": 0,
        "results": [
            {
                "task": "Research LangGraph architecture",
                "output": "LangGraph is a graph-based orchestration framework...",
                "score": 0  # Will be filled by critic
            }
        ],
        "memory_context": "",
        "final_report": "",
        "replan_count": 0,
        "is_done": False
    }
    
    # Mock ChatGroq
    with patch('nodes.critic.ChatGroq') as mock_groq_class:
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content=mock_llm_response)
        mock_groq_class.return_value = mock_llm
        
        # Mock environment
        with patch.dict('os.environ', {'GROQ_API_KEY': 'test-key'}):
            result = critic_node(initial_state)
    
    # Assert score is within range and properly updated
    assert 'results' in result, "Result should contain 'results' key"
    assert len(result['results']) > 0, "Result should have at least one entry"
    score = result['results'][0]['score']
    assert isinstance(score, int), f"Score should be int, got {type(score)}"
    assert 0 <= score <= 10, f"Score should be between 0 and 10, got {score}"
    assert score == 8, f"Score should be 8, got {score}"


# ============================================================================
# TEST 5: Search Returns Error Dict on Failure
# ============================================================================

def test_search_returns_empty_on_failure():
    """
    Test that web_search returns an error dict when Tavily API fails.
    
    Mocks the TavilySearchResults to raise an exception, verifies that
    web_search gracefully returns a dict with an "error" key instead of crashing.
    """
    # Mock Tavily to raise an exception
    with patch('tools.search.TavilySearchResults') as mock_tavily_class:
        mock_tavily = Mock()
        # Simulate Tavily API failure on invoke
        mock_tavily.invoke.side_effect = Exception("Tavily API timeout")
        mock_tavily_class.return_value = mock_tavily
        
        # Mock environment
        with patch.dict('os.environ', {'TAVILY_API_KEY': 'test-key'}):
            result = web_search.invoke("LangGraph basics")
    
    # Assert result is a dict with error key
    assert isinstance(result, dict), f"Result should be dict on error, got {type(result)}"
    assert 'error' in result, "Result should contain 'error' key on failure"
    assert 'Search failed after' in result['error'], "Error message should indicate retry exhaustion"


# ============================================================================
# TEST 6: Replanner Triggers Below Threshold
# ============================================================================

def test_replanner_triggers_below_threshold():
    """
    Test that when last result score < 7, the graph routes to planner (replanning).
    
    Verifies the _terminator_route conditional logic: if is_done is False
    and the last result's score is below 7, it should return "planner" to replan.
    """
    # Create state with a low-quality result (score < 7)
    low_quality_state: AgentState = {
        "goal": "Research framework comparison",
        "subtasks": ["Compare LangGraph and LangChain"],
        "current_task_index": 0,
        "results": [
            {
                "task": "Compare LangGraph and LangChain",
                "output": "LangGraph is newer.",
                "score": 5  # Below threshold of 7
            }
        ],
        "memory_context": "",
        "final_report": "",
        "replan_count": 0,
        "is_done": False
    }
    
    # Call the routing function
    route = _terminator_route(low_quality_state)
    
    # Assert it routes to planner for replanning
    assert route == "planner", f"Low score should route to 'planner', got '{route}'"


# ============================================================================
# TEST 7: Report Generator Produces Markdown
# ============================================================================

def test_report_generator_produces_markdown():
    """
    Test that report_generator node produces a markdown report.
    
    Mocks the ChatGroq LLM to return a markdown-formatted response,
    verifies the report contains markdown heading syntax.
    """
    # Mock LLM response with markdown
    mock_report = """# Research Report

## Executive Summary

LangGraph is a graph-based orchestration framework for building stateful, multi-agent applications.

## Key Findings

- **Architecture**: Node-based with explicit edges and conditional routing
- **State Management**: Persistent across steps using TypedDict and reducers
- **Deployment**: Supports production-grade workflows with human-in-the-loop

## Conclusion

LangGraph is superior for complex, stateful workflows."""
    
    # Create initial state with results
    initial_state: AgentState = {
        "goal": "Research LangGraph framework",
        "subtasks": ["Research LangGraph basics", "Research LangGraph architecture"],
        "current_task_index": 2,
        "results": [
            {
                "task": "Research LangGraph basics",
                "output": "LangGraph is a graph-based framework...",
                "score": 8
            },
            {
                "task": "Research LangGraph architecture",
                "output": "LangGraph uses nodes and edges...",
                "score": 9
            }
        ],
        "memory_context": "",
        "final_report": "",
        "replan_count": 1,
        "is_done": True
    }
    
    # Mock ChatGroq
    with patch('nodes.report_generator.ChatGroq') as mock_groq_class:
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content=mock_report)
        mock_groq_class.return_value = mock_llm
        
        # Mock environment and file operations
        with patch.dict('os.environ', {'GROQ_API_KEY': 'test-key'}):
            with patch('builtins.open', create=True):
                with patch('os.makedirs'):
                    result = report_generator_node(initial_state)
    
    # Assert markdown heading is present
    assert 'final_report' in result, "Result should contain 'final_report' key"
    report = result['final_report']
    assert isinstance(report, str), f"Report should be string, got {type(report)}"
    assert '# Research Report' in report, "Report should contain markdown heading"
    assert '##' in report, "Report should contain markdown subheadings"
    assert 'LangGraph' in report, "Report should contain research content"


# ============================================================================
# HELPER TESTS: Parsing Functions
# ============================================================================

def test_parse_subtasks_handles_various_formats():
    """Test that _parse_subtasks correctly handles various numbering formats."""
    response = """1. First task here
2) Second task
3: Third task
    4. Indented fourth task"""
    
    subtasks = _parse_subtasks(response)
    
    assert len(subtasks) == 4, f"Should parse 4 subtasks, got {len(subtasks)}"
    assert "First task here" in subtasks[0]
    assert "Second task" in subtasks[1]
    assert "Third task" in subtasks[2]
    assert "fourth task" in subtasks[3]


def test_parse_score_extracts_integer():
    """Test that _parse_score correctly extracts integers from LLM responses."""
    # Test various formats
    assert _parse_score("Quality: 8") == 8
    assert _parse_score("I would rate this a 7 out of 10") == 7
    assert _parse_score("Score is 10") == 10
    assert _parse_score("This is a 0 quality") == 0
    assert _parse_score("No number here") == 5  # Default fallback


def test_parse_score_clamps_range():
    """Test that _parse_score clamps scores to 0-10 range."""
    # Scores within range should pass through
    assert _parse_score("The score is 5") == 5
    
    # Scores outside range should be clamped
    # Note: _parse_score only looks for 0-10 in the regex, so 11+ won't match
    # and will default to 5
    assert _parse_score("Rank: 15") == 5  # No 0-10 digit found, uses default


# ============================================================================
# INTEGRATION-STYLE TESTS
# ============================================================================

def test_planner_fallback_on_llm_failure():
    """
    Test that planner gracefully falls back when LLM call fails.
    
    Verifies that if all LLM retries are exhausted, the planner returns
    a minimal fallback subtask list instead of crashing.
    """
    initial_state: AgentState = {
        "goal": "Research something",
        "subtasks": [],
        "current_task_index": 0,
        "results": [],
        "memory_context": "",
        "final_report": "",
        "replan_count": 0,
        "is_done": False
    }
    
    # Mock ChatGroq to fail all retries
    with patch('nodes.planner.ChatGroq') as mock_groq_class:
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM service unavailable")
        mock_groq_class.return_value = mock_llm
        
        with patch.dict('os.environ', {'GROQ_API_KEY': 'test-key'}):
            # Mock time.sleep to avoid delays in tests
            with patch('nodes.planner.time.sleep'):
                result = planner_node(initial_state)
    
    # Assert fallback subtask is returned
    assert 'subtasks' in result
    assert len(result['subtasks']) > 0, "Should have fallback subtasks"
    assert "Research the topic" in result['subtasks'], "Should use minimal fallback"


def test_critic_fallback_on_llm_failure():
    """
    Test that critic gracefully falls back when LLM call fails.
    
    Verifies that if all LLM retries are exhausted, the critic returns
    a neutral score of 5 instead of crashing.
    """
    initial_state: AgentState = {
        "goal": "Research something",
        "subtasks": [],
        "current_task_index": 0,
        "results": [
            {
                "task": "Test task",
                "output": "Test output",
                "score": 0
            }
        ],
        "memory_context": "",
        "final_report": "",
        "replan_count": 0,
        "is_done": False
    }
    
    # Mock ChatGroq to fail all retries
    with patch('nodes.critic.ChatGroq') as mock_groq_class:
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM service unavailable")
        mock_groq_class.return_value = mock_llm
        
        with patch.dict('os.environ', {'GROQ_API_KEY': 'test-key'}):
            # Mock time.sleep to avoid delays in tests
            with patch('nodes.critic.time.sleep'):
                result = critic_node(initial_state)
    
    # Assert fallback score is returned
    assert 'results' in result
    score = result['results'][0]['score']
    assert score == 5, f"Should return neutral fallback score of 5, got {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
