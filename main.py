import asyncio
from agents.graph import app 
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv() # Ensure keys are loaded before the graph runs

async def run_research(user_query: str):
    # Unique thread_id allows the graph to "remember" this specific execution
    config = {"configurable": {"thread_id": str(uuid4())}}
    
    initial_state = {
        "task": user_query,
        "plan": [],          
        "research_notes": [],
        "report": None,      
        "loop_count": 0,
        "score": 0        
    }

    print(f"--- Starting Research for: {user_query} ---")

    # Using stream_mode="updates" shows exactly which node just finished
    async for event in app.astream(initial_state, config, stream_mode="updates"):
        for node_name, output in event.items():
            print(f"\n[Node Execution] Finished: {node_name}")
            # Print partial updates if you want
            if "research_notes" in output:
               print(f" -> Found {len(output['research_notes'])} new facts.")

    # Fetch the final consolidated state after all nodes finish
    final_state = await app.aget_state(config)
    report = final_state.values.get("report")

    if report:
        print("\n" + "="*50)
        print("FINAL RESEARCH REPORT")
        print("="*50 + "\n")
        print(report)
    else:
        print("\n[Error] No report was generated. Check your node logic.")

if __name__ == "__main__":
    query = "Research the impact of generative AI on PostgreSQL performance optimization."
    try:
        print(app.get_graph().print_ascii())
        asyncio.run(run_research(query))
    except KeyboardInterrupt:
        print("\nResearch cancelled by user.")