from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
import os

'''
connect to MongoDB Atlas Vector Store, includes retriever and helper function to get the vector store
'''

def get_vector_store():
    client = MongoClient(os.getenv("MONGO_URI"))
    collection = client["research_db"]["documents"]
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=OpenAIEmbeddings(),
        index_name="vector_index"
    )

def get_retriever():
    return get_vector_store().as_retriever(search_kwargs={"k": 3})