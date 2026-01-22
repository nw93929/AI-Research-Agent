# Finance Documents Folder

This folder is where you place your **SEC filings, financial PDFs, and other documents** that you want the AI Research Agent to access via RAG (Retrieval-Augmented Generation).

## 📁 What to Put Here

Place any of these file types:
- **PDF files** (10-K, 10-Q, 8-K, earnings reports, analyst reports)
- **Text files** (.txt)
- **Word documents** (.docx, .doc)

## 📂 Folder Structure (Optional)

You can organize documents in subdirectories for easier management:

```
finance_docs/
├── sec_filings/
│   ├── AAPL_10K_2024.pdf
│   └── TSLA_10Q_Q3_2024.pdf
├── earnings_reports/
│   ├── GOOGL_Q4_2024_Earnings.pdf
│   └── MSFT_Annual_Report_2024.pdf
└── analyst_reports/
    └── JPMorgan_Tech_Sector_2024.pdf
```

## 🚀 How to Upload Documents to Pinecone

Once you've added your documents to this folder:

1. **Run the ingestion script:**
   ```bash
   python scripts/ingest_documents.py
   ```

2. **The script will:**
   - ✅ Validate your API keys (.env file)
   - ✅ Count and list all documents found
   - ✅ Ask for confirmation before uploading
   - ✅ Chunk documents into 512-token pieces
   - ✅ Generate embeddings using OpenAI `text-embedding-3-small`
   - ✅ Upload vectors to your Pinecone database

3. **Wait for completion:**
   - Processing time depends on file size
   - Typical 10-K (200 pages): ~2-3 minutes
   - You'll see a progress bar during processing

## 🔄 Re-running Ingestion

If you add more documents later:
- Just drop new files in this folder
- Run `python scripts/ingest_documents.py` again
- New documents will be added to existing vectors in Pinecone

**Note:** The script does NOT delete old vectors. If you want to start fresh, manually clear your Pinecone index from the dashboard.

## 🧪 Testing Retrieval

After ingestion, test that your documents are accessible:

```bash
python main.py
# Query: "What was Apple's revenue in 2024?"
```

The researcher_node will automatically search Pinecone and retrieve relevant chunks from your uploaded documents.

## ⚠️ File Size Limits

- **Per file:** No hard limit, but large files (>50MB) may take longer
- **Total folder:** Depends on your Pinecone plan (Free tier: 100K vectors ~= 50MB of text)
- **Recommendation:** Start with 10-20 key documents, then scale up

## 🔒 Security Note

This folder is **gitignored** by default. Your proprietary documents will NOT be committed to version control.

Only the README.md file is tracked in Git to preserve folder structure.
