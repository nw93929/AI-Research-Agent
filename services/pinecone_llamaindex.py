import os
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

def query_pinecone_llamaindex(query_text):
    # Setup Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    # Setup Vector Store for LlamaIndex
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # Set up the index connection
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    # Query the index
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    
    return str(response)