"""
Document Ingestion Pipeline for AI Research Agent

This script uploads SEC filings, financial PDFs, and other documents to your Pinecone
vector database so the RAG system can retrieve them during research.

Usage:
    1. Place your PDF files in the './finance_docs' folder
    2. Run: python scripts/ingest_documents.py
    3. The script will chunk, embed, and upload them to Pinecone

Requirements:
    - OPENAI_API_KEY in .env (for embeddings)
    - PINECONE_API_KEY in .env
    - PINECONE_INDEX_NAME in .env
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone


def validate_environment():
    """Check that all required environment variables are set."""
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"❌ ERROR: Missing required environment variables: {', '.join(missing)}")
        print("\nPlease add these to your .env file:")
        for var in missing:
            print(f"  {var}=your_key_here")
        sys.exit(1)

    print("✅ Environment variables validated")


def check_documents_folder(docs_path):
    """Check if the finance_docs folder exists and has files."""
    if not docs_path.exists():
        print(f"❌ ERROR: Documents folder not found at {docs_path}")
        print(f"\nPlease create the folder and add your PDF files:")
        print(f"  mkdir {docs_path}")
        sys.exit(1)

    # Check for supported file types
    supported_extensions = ['.pdf', '.txt', '.docx', '.doc']
    files = [f for f in docs_path.iterdir() if f.suffix.lower() in supported_extensions]

    if not files:
        print(f"⚠️  WARNING: No documents found in {docs_path}")
        print(f"\nSupported file types: {', '.join(supported_extensions)}")
        print("\nPlease add your documents to this folder and run the script again.")
        sys.exit(0)

    print(f"✅ Found {len(files)} document(s) to process:")
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  • {file.name} ({size_mb:.2f} MB)")

    return files


def ingest_documents(docs_path, index_name):
    """
    Main ingestion pipeline: Load → Chunk → Embed → Upload to Pinecone
    """
    print("\n" + "="*60)
    print("Starting Document Ingestion Pipeline")
    print("="*60)

    # 1. Connect to Pinecone
    print("\n[1/4] Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    try:
        pinecone_index = pc.Index(index_name)
        stats = pinecone_index.describe_index_stats()
        print(f"✅ Connected to index '{index_name}'")
        print(f"   Current vector count: {stats.get('total_vector_count', 0)}")
    except Exception as e:
        print(f"❌ ERROR: Could not connect to Pinecone index '{index_name}'")
        print(f"   Error: {str(e)}")
        print("\nPlease verify:")
        print("  1. Your PINECONE_API_KEY is correct")
        print("  2. Your PINECONE_INDEX_NAME exists in your Pinecone dashboard")
        sys.exit(1)

    # 2. Setup Vector Store (the "bridge" between LlamaIndex and Pinecone)
    print("\n[2/4] Setting up vector store connection...")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Configure embeddings (same model used by query system)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    print("✅ Using OpenAI text-embedding-3-small model")

    # 3. Load documents from folder
    print(f"\n[3/4] Loading documents from {docs_path}...")
    try:
        documents = SimpleDirectoryReader(
            input_dir=str(docs_path),
            recursive=True,  # Include subdirectories
            required_exts=['.pdf', '.txt', '.docx', '.doc']  # Supported file types
        ).load_data()

        print(f"✅ Loaded {len(documents)} document(s)")

        # Show document metadata
        for i, doc in enumerate(documents[:5], 1):  # Show first 5
            filename = doc.metadata.get('file_name', 'unknown')
            print(f"   {i}. {filename}")
        if len(documents) > 5:
            print(f"   ... and {len(documents) - 5} more")

    except Exception as e:
        print(f"❌ ERROR: Failed to load documents")
        print(f"   Error: {str(e)}")
        sys.exit(1)

    # 4. Create index (This automatically chunks, embeds, and uploads)
    print(f"\n[4/4] Processing documents (chunking, embedding, uploading)...")
    print("⏳ This may take several minutes depending on document size...")

    try:
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True  # Shows progress bar
        )

        print("\n✅ SUCCESS! Documents uploaded to Pinecone")

        # Get updated stats
        stats = pinecone_index.describe_index_stats()
        new_count = stats.get('total_vector_count', 0)
        print(f"\nPinecone Index Statistics:")
        print(f"  Total vectors: {new_count}")
        print(f"  Index name: {index_name}")

    except Exception as e:
        print(f"\n❌ ERROR: Failed to upload documents")
        print(f"   Error: {str(e)}")
        sys.exit(1)

    print("\n" + "="*60)
    print("✨ Ingestion Complete!")
    print("="*60)
    print("\nYour documents are now available to the research agent.")
    print("You can test retrieval by running:")
    print("  python main.py")


def main():
    """Main entry point for document ingestion."""
    print("🚀 AI Research Agent - Document Ingestion Pipeline")
    print()

    # Setup paths
    docs_path = project_root / "finance_docs"

    # Validate environment
    validate_environment()

    # Check for documents
    files = check_documents_folder(docs_path)

    # Get Pinecone index name
    index_name = os.getenv("PINECONE_INDEX_NAME")

    # Confirm before proceeding
    print(f"\n📤 Ready to upload {len(files)} document(s) to Pinecone index '{index_name}'")
    response = input("Continue? (y/n): ").lower().strip()

    if response != 'y':
        print("❌ Ingestion cancelled")
        sys.exit(0)

    # Run ingestion
    ingest_documents(docs_path, index_name)


if __name__ == "__main__":
    main()
