import pytest
from agents.graph import planner_node
from agents.state import AgentState

''' Unit test for the planner_node in the research agent graph'''

def test_planner_updates_state():
    # test state
    initial_state: AgentState = {
        "task": "Test research",
        "plan": [],
        "research_notes": [],
        "loop_count": 0
    }
    
    output = planner_node(initial_state)
    # Validate the output
    assert "plan" in output
    assert len(output["plan"]) > 0
    assert isinstance(output["plan"], list)