import os
from langchain_community.tools.tavily_search import TavilySearchResults

'''
wrapper for Tavily API to perform web searches
'''

def web_search(query: str):
    tool = TavilySearchResults(k=3)
    return tool.run(query)