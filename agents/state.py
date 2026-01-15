import operator
from typing import Annotated, List, TypedDict, Optional

'''
This is the agent's shared memory. It defines a TypedDict that every node in the graph will share to read and write data.
'''

class AgentState(TypedDict):
    task: str
    plan: List[str]  # steps to take created by Planner
    context: Annotated[List[str], operator.add]  # operator.add so new context is appended, not overwritten
    research_notes: Annotated[List[str], operator.add] 
    report: Optional[str]
    score: int  # quality score (0-10) from Grader
    loop_count: int  # Track iterations to prevent infinite loops