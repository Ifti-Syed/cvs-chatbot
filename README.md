# CVS Chatbot – Central Ventilation System

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot for Central Ventilation System (CVS). Upload product manuals, installation guides, and technical specifications as PDFs; ask questions and get accurate, source-cited answers powered by OpenAI GPT-4o mini and Qdrant vector search.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Local Development Setup](#3-local-development-setup)
4. [Docker Setup](#4-docker-setup)
5. [Qdrant Setup](#5-qdrant-setup)
6. [Google Cloud Run Deployment](#6-google-cloud-run-deployment)
7. [API Documentation](#7-api-documentation)
8. [Environment Variables Reference](#8-environment-variables-reference)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Project Overview

| Feature | Details |
|---|---|
| **Framework** | FastAPI (Python 3.11) |
| **LLM** | OpenAI GPT-4o mini |
| **Embeddings** | OpenAI text-embedding-3-small (1536 dims) |
| **Vector DB** | Qdrant (local Docker or Qdrant Cloud) |
| **Frontend** | Vanilla HTML / CSS / JS (dark theme, ChatGPT-style) |
| **Deployment** | Google Cloud Run + Qdrant Cloud |

### Key Capabilities

- **RAG Pipeline**: Upload PDFs → extract text → chunk → embed → store in Qdrant → answer questions with cited sources.
- **Document Management**: Upload, list, and delete ingested documents via the UI or API.
- **Conversation History**: Last 10 messages sent as context for multi-turn dialogue.
- **Single-service deployment**: One Docker container serves both the REST API and the frontend.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Frontend  (HTML + CSS + JS)  – Dark ChatGPT UI     │   │
│   │  - Chat interface with message bubbles              │   │
│   │  - PDF upload modal with drag & drop                │   │
│   │  - Document list in sidebar                         │   │
│   └────────────────────┬────────────────────────────────┘   │
└────────────────────────│────────────────────────────────────┘
                         │ HTTP (same-origin)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│                                                              │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────┐   │
│  │  /api/chat  │   │/api/upload-  │   │ /api/documents │   │
│  │  (POST)     │   │ document     │   │ (GET / DELETE) │   │
│  └──────┬──────┘   └──────┬───────┘   └───────┬────────┘   │
│         │                 │                    │            │
│  ┌──────▼──────────────────▼──────────────────▼──────────┐ │
│  │              Service Layer                             │ │
│  │  ChatService  |  IngestionPipeline  |  QdrantService  │ │
│  └──────┬───────────────────┬───────────────────┬────────┘ │
│         │                   │                   │          │
│  ┌──────▼──────┐    ┌───────▼──────┐    ┌───────▼───────┐ │
│  │OpenAIService│    │ PDFProcessor │    │  QdrantClient │ │
│  │(Embeddings +│    │  (pypdf)     │    │               │ │
│  │  Chat API)  │    └──────────────┘    └───────┬───────┘ │
│  └──────┬──────┘                                │         │
└─────────│──────────────────────────────────────-│─────────┘
          │                                        │
          ▼                                        ▼
  ┌───────────────┐                    ┌───────────────────┐
  │   OpenAI API  │                    │  Qdrant (local    │
  │  (GPT-4o mini │                    │  Docker or Cloud) │
  │   embeddings) │                    │                   │
  └───────────────┘                    └───────────────────┘
```

### RAG Flow

```
User Question
     │
     ▼
Generate Embedding (OpenAI text-embedding-3-small)
     │
     ▼
Semantic Search in Qdrant (top-k=5 chunks)
     │
     ▼
Build Context String (chunk text + source metadata)
     │
     ▼
Compose Messages (system prompt + context + history + question)
     │
     ▼
OpenAI Chat Completion (GPT-4o mini)
     │
     ▼
Return Answer + Sources to Frontend
```

---

## 3. Local Development Setup

### Prerequisites

- Python 3.11+
- pip
- Docker (for Qdrant)
- An OpenAI API key

### Step 1 – Clone and create virtual environment

```bash
git clone <your-repo-url>
cd CVS_Chatbot

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 – Configure environment

```bash
cp .env .env
```

Edit `.env` and set at minimum:

```
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
```

### Step 4 – Start Qdrant locally

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

### Step 5 – Run the application

```bash
# From the CVS_Chatbot directory (project root)
python -m uvicorn backend.main:app --reload --port 8000
```

Open your browser at **http://localhost:8000**

- Frontend UI: http://localhost:8000/
- API docs (Swagger): http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

---

## 4. Docker Setup

### Build and run locally with Docker

```bash
# Build the image (run from project root)
docker build -f docker/Dockerfile -t cvs-chatbot:latest .

# Run the container
docker run -d \
  --name cvs-chatbot \
  -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -e QDRANT_URL=https://your-cluster.qdrant.io \
  -e QDRANT_API_KEY=your-qdrant-api-key \
  cvs-chatbot:latest
```

Open http://localhost:8080

### Docker Compose (app + local Qdrant)

Create `docker-compose.yml` in the project root:

```yaml
version: '3.9'

services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  chatbot:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8080:8080"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_COLLECTION_NAME=cvs_documents
      - ENVIRONMENT=production
    depends_on:
      - qdrant

volumes:
  qdrant_data:
```

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f chatbot

# Stop
docker-compose down
```

---

## 5. Qdrant Setup

### Option A – Local Qdrant with Docker

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

Use `QDRANT_URL=http://localhost:6333` (no API key needed).

### Option B – Qdrant Cloud (recommended for production)

1. Sign up at [https://cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a free cluster
3. Copy your **Cluster URL** and **API Key**
4. Set environment variables:
   ```
   QDRANT_URL=https://your-cluster-id.aws.cloud.qdrant.io
   QDRANT_API_KEY=your_qdrant_api_key
   ```

The application automatically uses API key authentication when `QDRANT_API_KEY` is set.

---

## 6. Google Cloud Run Deployment

### Prerequisites

- Google Cloud SDK installed and authenticated (`gcloud auth login`)
- A GCP project with billing enabled
- Artifact Registry or Container Registry enabled
- A Qdrant Cloud cluster (or VPC-peered Qdrant instance)

### Step 1 – Set your project

```bash
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1
export SERVICE_NAME=cvs-chatbot
export IMAGE_NAME=gcr.io/${PROJECT_ID}/${SERVICE_NAME}

gcloud config set project ${PROJECT_ID}
```

### Step 2 – Enable required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  containerregistry.googleapis.com \
  cloudbuild.googleapis.com
```

### Step 3 – Build and push Docker image

```bash
# From the CVS_Chatbot project root
gcloud builds submit \
  --tag ${IMAGE_NAME}:latest \
  --dockerfile docker/Dockerfile \
  .
```

Alternatively, build locally and push:

```bash
docker build -f docker/Dockerfile -t ${IMAGE_NAME}:latest .
docker push ${IMAGE_NAME}:latest
```

### Step 4 – Deploy to Cloud Run

```bash
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --region ${REGION} \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 300 \
  --set-env-vars "OPENAI_API_KEY=sk-your-key-here" \
  --set-env-vars "QDRANT_URL=https://your-cluster.qdrant.io" \
  --set-env-vars "QDRANT_API_KEY=your-qdrant-api-key" \
  --set-env-vars "QDRANT_COLLECTION_NAME=cvs_documents" \
  --set-env-vars "OPENAI_EMBEDDING_MODEL=text-embedding-3-small" \
  --set-env-vars "OPENAI_CHAT_MODEL=gpt-4o-mini" \
  --set-env-vars "ENVIRONMENT=production" \
  --set-env-vars "PORT=8080"
```

### Step 5 – Verify deployment

```bash
# Get the service URL
gcloud run services describe ${SERVICE_NAME} \
  --region ${REGION} \
  --format 'value(status.url)'
```

Open the URL in your browser. The CVS Chatbot should be live.

### Step 6 – Update a deployed service

After making code changes:

```bash
# Rebuild and push
gcloud builds submit --tag ${IMAGE_NAME}:latest --dockerfile docker/Dockerfile .

# Deploy new revision
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --region ${REGION}
```

### Using Secret Manager for API Keys (recommended for production)

```bash
# Store secrets
echo -n "sk-your-openai-key" | gcloud secrets create OPENAI_API_KEY --data-file=-
echo -n "your-qdrant-api-key" | gcloud secrets create QDRANT_API_KEY --data-file=-

# Grant Cloud Run service account access
gcloud secrets add-iam-policy-binding OPENAI_API_KEY \
  --member="serviceAccount:${PROJECT_ID}@appspot.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Deploy with secrets
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --region ${REGION} \
  --set-secrets "OPENAI_API_KEY=OPENAI_API_KEY:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest"
```

---

## 7. API Documentation

Interactive docs available at `/api/docs` (Swagger UI) and `/api/redoc` when the app is running.

### POST /api/chat

Ask a question to the CVS Assistant.

**Request:**
```json
{
  "message": "What ventilation products does CVS offer?",
  "conversation_history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I help you?"}
  ]
}
```

**Response:**
```json
{
  "answer": "CVS offers a range of ventilation products including...",
  "sources": [
    {
      "document_name": "CVS_Product_Catalog_2024",
      "page_number": 5,
      "score": 0.8923
    }
  ]
}
```

---

### POST /api/upload-document

Upload and ingest a PDF document.

**Request:** `multipart/form-data` with field `file` (PDF file)

**Response:**
```json
{
  "status": "success",
  "document_name": "CVS_Product_Catalog_2024",
  "total_chunks": 142,
  "pages": 24,
  "message": "Document 'CVS_Product_Catalog_2024' ingested successfully. 142 chunks stored from 24 pages."
}
```

---

### GET /api/documents

List all ingested documents.

**Response:**
```json
[
  {"document_name": "CVS_Product_Catalog_2024"},
  {"document_name": "Installation_Manual_v3"}
]
```

---

### DELETE /api/documents/{document_name}

Delete all vectors for a specific document.

**Response:**
```json
{
  "status": "success",
  "message": "Document 'CVS_Product_Catalog_2024' deleted successfully."
}
```

---

### GET /health

Health check endpoint.

**Response:**
```json
{"status": "healthy", "service": "CVS Chatbot"}
```

---

## 8. Environment Variables Reference

| Variable | Default | Required | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | – | Yes | OpenAI API key |
| `QDRANT_URL` | `http://localhost:6333` | Yes | Qdrant instance URL |
| `QDRANT_API_KEY` | – | No | Qdrant Cloud API key |
| `QDRANT_COLLECTION_NAME` | `cvs_documents` | No | Qdrant collection name |
| `GCS_BUCKET_NAME` | – | No | GCS bucket for PDF storage |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | No | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | No | Chat completion model |
| `CHUNK_SIZE` | `512` | No | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | No | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | No | Number of chunks retrieved per query |
| `MAX_TOKENS` | `1024` | No | Max tokens in chat response |
| `ENVIRONMENT` | `development` | No | `development` or `production` |
| `PORT` | `8000` | No | Server port |

---

## 9. Troubleshooting

### "Connection refused" to Qdrant

**Cause:** Qdrant is not running or the URL is wrong.

**Fix:**
```bash
# Check Qdrant container is running
docker ps | grep qdrant

# Start if not running
docker start qdrant

# Test connectivity
curl http://localhost:6333/healthz
```

---

### "OPENAI_API_KEY not found" or validation error on startup

**Cause:** `.env` file is missing or `OPENAI_API_KEY` is not set.

**Fix:**
```bash
cp .env .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

---

### PDF upload returns "No extractable text found"

**Cause:** The PDF is scanned (image-only) and has no machine-readable text.

**Fix:** Use an OCR tool such as Adobe Acrobat or `ocrmypdf` to add a text layer before uploading:
```bash
pip install ocrmypdf
ocrmypdf input_scan.pdf output_searchable.pdf
```

---

### Chat responses are generic / not using document context

**Cause:** No documents have been ingested, or the query does not semantically match any chunks.

**Fix:**
1. Upload at least one relevant PDF document.
2. Check the document list in the sidebar to confirm ingestion succeeded.
3. Try rephrasing your question to match terminology used in the document.

---

### Cloud Run container fails to start

**Cause:** Missing environment variables or inability to reach Qdrant.

**Fix:**
```bash
# View Cloud Run logs
gcloud run services logs read cvs-chatbot --region us-central1 --limit 50
```

Ensure all required env vars are set and Qdrant Cloud cluster is accessible from Cloud Run (no VPC restrictions).

---

### ImportError: No module named 'backend'

**Cause:** `PYTHONPATH` is not set to the project root.

**Fix:**
```bash
# Run from the CVS_Chatbot root directory
PYTHONPATH=. python -m uvicorn backend.main:app --reload
# Or set in .env: PYTHONPATH=/path/to/CVS_Chatbot
```

---

### High memory usage during ingestion of large PDFs

**Cause:** Large PDFs with many pages generate many embeddings.

**Fix:** Embeddings are batched in groups of 100. For very large documents (500+ pages), consider increasing Cloud Run memory to 2 Gi:
```bash
gcloud run services update cvs-chatbot --memory 2Gi --region us-central1
```

---

*Built with FastAPI, OpenAI, and Qdrant for Central Ventilation System.*
