import pytest
from main import app # Your LangGraph app

test_cases = [
    {
        "task": "What is the return policy for electronics?",
        "must_include": ["30 days", "original packaging", "restocking fee"],
    },
    {
        "task": "How do I reset my admin password?",
        "must_include": ["IT portal", "two-factor authentication"],
    }
]

def test_agent_accuracy():
    for case in test_cases:
        # Run the agent
        result = app.invoke({"task": case["task"], "loop_count": 0})
        report = result["report"]
        
        # Check if the "must-include" facts are in the final report
        found_facts = [fact in report.lower() for fact in case["must_include"]]
        accuracy = sum(found_facts) / len(found_facts)
        
        print(f"Task: {case['task']} | Accuracy: {accuracy * 100}%")
        assert accuracy >= 0.8  # Fail the test if less than 80% accurate