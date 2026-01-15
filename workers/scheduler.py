from apscheduler.schedulers.blocking import BlockingScheduler
from main import run_research
import asyncio

'''
Scheduler to run research tasks at specified intervals'''

# This wrapper allows the sync scheduler to call your async agent
def job_wrapper(query):
    print(f"--- Triggering Scheduled Task: {query} ---")
    asyncio.run(run_research(query))

scheduler = BlockingScheduler()

# My usage: a deep dive into recent AI developments every morning at 9:00 AM
scheduler.add_job(
    job_wrapper, 
    'cron', 
    hour=9, 
    minute=0, 
    args=["Summarize top 5 AI breakthroughs (like a new YOLOE promptable model or new AI tool) from the last 24 hours."]
)

if __name__ == "__main__":
    print("Scheduler started. Press Ctrl+C to exit.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass