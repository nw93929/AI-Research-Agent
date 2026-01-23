# Document Ingestion Guide - RAG Setup

This guide explains how to add your own SEC filings, financial PDFs, and research documents to the AI Research Agent's knowledge base.

## 📚 What is Document Ingestion?

**Document Ingestion** is the process of preparing your documents so the AI agent can retrieve and reference them during research. This is the foundation of RAG (Retrieval-Augmented Generation).

### The Pipeline

```
Your PDFs → Chunking → Embedding → Pinecone Database → Research Agent
```

**What happens:**
1. **Load:** Read PDF/text files from `finance_docs/` folder
2. **Chunk:** Split documents into 512-token pieces (overlapping for context)
3. **Embed:** Convert each chunk to a 1536-dimensional vector using OpenAI embeddings
4. **Upload:** Store vectors in Pinecone with metadata (filename, page number, etc.)
5. **Retrieve:** When the agent needs info, it searches Pinecone for relevant chunks

## 🚀 Quick Start (5 Minutes)

### Step 1: Add Your Documents

Place your files in the `finance_docs/` folder:

```bash
finance_docs/
├── AAPL_10K_2024.pdf
├── TSLA_Earnings_Q3.pdf
└── Warren_Buffett_Letter_2024.pdf
```

**Supported formats:**
- PDF (`.pdf`) - Most common for SEC filings
- Text (`.txt`) - Plain text documents
- Word (`.docx`, `.doc`) - Microsoft Word files

### Step 2: Run the Ingestion Script

```bash
python scripts/ingest_documents.py
```

**What you'll see:**
```
🚀 AI Research Agent - Document Ingestion Pipeline

✅ Environment variables validated
✅ Found 3 document(s) to process:
  • AAPL_10K_2024.pdf (2.34 MB)
  • TSLA_Earnings_Q3.pdf (0.87 MB)
  • Warren_Buffett_Letter_2024.pdf (1.12 MB)

📤 Ready to upload 3 document(s) to Pinecone index 'financial-research'
Continue? (y/n): y

[1/4] Connecting to Pinecone...
✅ Connected to index 'financial-research'
   Current vector count: 0

[2/4] Setting up vector store connection...
✅ Using OpenAI text-embedding-3-small model

[3/4] Loading documents from finance_docs...
✅ Loaded 3 document(s)

[4/4] Processing documents (chunking, embedding, uploading)...
⏳ This may take several minutes...
[████████████████████] 100%

✅ SUCCESS! Documents uploaded to Pinecone

Pinecone Index Statistics:
  Total vectors: 847
  Index name: financial-research

✨ Ingestion Complete!
```

### Step 3: Test Retrieval

```bash
python main.py
# Enter query: "What was Apple's revenue growth in 2024?"
```

The agent will now search your uploaded documents and cite them in the report!

## 📋 Detailed Workflow

### Prerequisites

Before running ingestion, ensure you have:

1. **Required API Keys in `.env`:**
   ```env
   OPENAI_API_KEY=sk-...           # For embeddings
   PINECONE_API_KEY=your_key       # For vector storage
   PINECONE_INDEX_NAME=your_index  # Your Pinecone index name
   ```

2. **Pinecone Index Created:**
   - Go to [app.pinecone.io](https://app.pinecone.io)
   - Create a new index:
     - **Dimension:** 1536 (for text-embedding-3-small)
     - **Metric:** cosine
     - **Environment:** Free tier is sufficient for testing

3. **Documents Ready:**
   - Downloaded SEC filings from [EDGAR](https://www.sec.gov/edgar)
   - Earnings reports from company investor relations pages
   - Analyst reports or proprietary research

### Understanding the Script

The `scripts/ingest_documents.py` script performs these steps:

#### 1. Environment Validation
```python
validate_environment()
# Checks for OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
```

#### 2. Document Discovery
```python
SimpleDirectoryReader("./finance_docs").load_data()
# Recursively finds all PDFs, TXTs, DOCX in finance_docs/
```

#### 3. Automatic Chunking
```python
# Behind the scenes, LlamaIndex:
# - Splits documents into 512-token chunks
# - Adds 50-token overlap (prevents context loss at boundaries)
# - Preserves metadata (filename, page number)
```

#### 4. Embedding Generation
```python
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Converts each chunk to a 1536-dimensional vector
# Cost: ~$0.0001 per 1000 tokens (~$0.05 for a 200-page 10-K)
```

#### 5. Upload to Pinecone
```python
VectorStoreIndex.from_documents(documents, storage_context=storage_context)
# Uploads vectors + metadata to Pinecone
# Shows progress bar for long uploads
```

## 🔄 Common Use Cases

### Use Case 1: Analyzing Specific Companies

**Scenario:** You're researching Apple and want deep access to their filings.

**Documents to add:**
```
finance_docs/AAPL/
├── 10-K_2024.pdf       # Annual report
├── 10-Q_Q1_2024.pdf    # Quarterly filings
├── 10-Q_Q2_2024.pdf
├── 10-Q_Q3_2024.pdf
└── Earnings_Transcript_Q4_2024.txt
```

**Ingestion:**
```bash
python scripts/ingest_documents.py
# Uploads all Apple documents
```

**Query:**
```bash
python main.py
Query: "What are Apple's biggest risks according to their 10-K?"
```

**Result:** The agent retrieves the "Risk Factors" section and summarizes it.

### Use Case 2: Sector Analysis

**Scenario:** You're comparing tech companies.

**Documents to add:**
```
finance_docs/
├── tech_sector/
│   ├── AAPL_10K.pdf
│   ├── MSFT_10K.pdf
│   ├── GOOGL_10K.pdf
│   └── META_10K.pdf
```

**Query:**
```bash
Query: "Compare R&D spending as % of revenue for FAANG companies"
```

**Result:** The agent searches all uploaded filings and compares R&D ratios.

### Use Case 3: Historical Analysis

**Scenario:** You want to track a company's evolution over time.

**Documents to add:**
```
finance_docs/TSLA_Historical/
├── 10-K_2020.pdf
├── 10-K_2021.pdf
├── 10-K_2022.pdf
├── 10-K_2023.pdf
└── 10-K_2024.pdf
```

**Query:**
```bash
Query: "How has Tesla's gross margin changed from 2020 to 2024?"
```

**Result:** The agent retrieves data from all 5 filings and shows the trend.

## 🛠️ Advanced Configuration

### Customizing Chunk Size

Edit `scripts/ingest_documents.py` to change chunking parameters:

```python
# Default (balanced):
splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

# Smaller chunks (more precise retrieval, but less context):
splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=25)

# Larger chunks (more context, but less precise):
splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=100)
```

**When to use smaller chunks:**
- Extracting specific numbers (revenue, P/E ratios)
- Answering narrow factual questions

**When to use larger chunks:**
- Understanding complex narratives (MD&A sections)
- Analyzing qualitative information (risk factors, strategy)

### Using Different Embedding Models

```python
# Default (cost-effective):
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Cost: $0.02 per 1M tokens
# Dimension: 1536

# Higher quality (more expensive):
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
# Cost: $0.13 per 1M tokens
# Dimension: 3072 (requires recreating Pinecone index)
```

**Note:** If you change models, you must:
1. Create a new Pinecone index with matching dimensions
2. Re-ingest all documents

### Filtering Documents by Type

To only upload PDFs (skip TXT, DOCX):

```python
documents = SimpleDirectoryReader(
    input_dir="./finance_docs",
    required_exts=['.pdf']  # Only PDFs
).load_data()
```

## 📊 Cost Estimates

### Embedding Costs (OpenAI text-embedding-3-small)

| Document Type | Pages | Tokens | Cost |
|---------------|-------|--------|------|
| 10-K (Annual Report) | 200 | ~100K | $0.002 |
| 10-Q (Quarterly) | 50 | ~25K | $0.0005 |
| Earnings Transcript | 20 | ~10K | $0.0002 |
| Analyst Report | 30 | ~15K | $0.0003 |

**Example:** Uploading 10 company 10-Ks = $0.02 in embedding costs

### Pinecone Storage Costs

**Free Tier:**
- 100,000 vectors (~50 10-K reports)
- 1 index
- No credit card required

**Starter Plan ($70/month):**
- 5 million vectors (~2,500 10-K reports)
- Sufficient for most individual researchers

## 🐛 Troubleshooting

### Error: "No documents found in finance_docs"

**Cause:** The folder is empty or contains unsupported file types.

**Fix:**
```bash
# Check folder contents
ls finance_docs/

# Ensure files have correct extensions: .pdf, .txt, .docx
```

### Error: "Could not connect to Pinecone index"

**Cause:** Index name doesn't exist or API key is wrong.

**Fix:**
1. Log in to [app.pinecone.io](https://app.pinecone.io)
2. Verify your index name
3. Check `.env` file has correct `PINECONE_INDEX_NAME`

### Error: "Rate limit exceeded" (OpenAI)

**Cause:** Uploading too many documents too quickly.

**Fix:**
- Wait 60 seconds and try again
- Upgrade to OpenAI Tier 2 (requires $50 credit purchase)
- Process documents in smaller batches

### Error: "Dimension mismatch" (Pinecone)

**Cause:** Your Pinecone index dimension doesn't match embedding model.

**Fix:**
- `text-embedding-3-small` requires dimension = 1536
- `text-embedding-3-large` requires dimension = 3072
- Recreate your Pinecone index with correct dimension

## 🔐 Security Best Practices

### Never Commit Proprietary Documents

The `.gitignore` file is configured to exclude all documents:

```gitignore
# Excludes:
finance_docs/*.pdf
finance_docs/**/*.pdf

# Keeps (for structure):
finance_docs/README.md
finance_docs/.gitkeep
```

**Verify before pushing:**
```bash
git status
# Should NOT show any PDFs or documents
```

### Handling Sensitive Data

If your documents contain sensitive information:

1. **Use a private Pinecone index** (not shared with others)
2. **Set up access controls** in Pinecone dashboard
3. **Rotate API keys regularly** (every 90 days)
4. **Never log document contents** in API calls

## 🚦 Next Steps

After successful ingestion:

1. **Test retrieval quality:**
   ```bash
   python main.py
   # Try queries that should reference your uploaded docs
   ```

2. **Check citations:**
   - Does the agent cite specific documents?
   - Are the page numbers accurate?

3. **Iterate on queries:**
   - Start with factual questions ("What was X's revenue?")
   - Progress to analytical questions ("Why did margins decline?")

4. **Monitor Pinecone usage:**
   - Track vector count growth
   - Set up alerts if approaching tier limits

5. **Automate with n8n (optional):**
   - See n8n integration guide for automatic ingestion
   - Trigger ingestion when new PDFs arrive via email/Google Drive

## 📚 Related Documentation

- [QUICK_START.md](QUICK_START.md) - Initial setup guide
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Full architecture overview
- [finance_docs/README.md](finance_docs/README.md) - Document folder guide
- [Pinecone Documentation](https://docs.pinecone.io/) - Vector database details
- [LlamaIndex Documentation](https://docs.llamaindex.ai/) - Ingestion framework

---

**Questions?** Check the [finance_docs/README.md](finance_docs/README.md) for folder-specific guidance.
