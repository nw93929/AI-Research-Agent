import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

def get_pinecone_retriever():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load existing index
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name, 
        embedding=embeddings
    )
    
    # Returns a retriever that pulls the top 3 relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 3})