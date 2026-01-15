from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
import os

def process_document(file_path):
    """
    Processes a financial PDF document using LlamaIndex standards.
    This ensures compatibility with the researcher_node retrieval logic.
    """
    # 1. Load the document (SimpleDirectoryReader handles PDFs automatically)
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    
    # 2. Chunking (Senior Tip: TokenTextSplitter is more accurate for LLMs than character splitters)
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # 3. Embedding
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    # Extract text and generate vectors
    texts = [node.get_content() for node in nodes]
    vectors = [embed_model.get_text_embedding(text) for text in texts]
    
    return {"text_chunks": texts, "vectors": vectors}