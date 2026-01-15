from celery import Celery
from main import run_research
import asyncio

'''
Celery app to handle task queue for running research in the background
'''

app = Celery('research_tasks', broker='redis://localhost:6379/0')

@app.task
def background_research_task(user_query):
    # This runs in a separate worker process
    asyncio.run(run_research(user_query))
    return f"Research completed for: {user_query}"