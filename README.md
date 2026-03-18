# AI Knowledge Assistant

Production-ready full-stack RAG application with a LangChain tool-calling agent.

## Stack

- Next.js 15 (App Router)
- TypeScript
- LangChain JS
- Gemini + OpenAI provider support (Gemini default)
- ChromaDB vector database
- TailwindCSS

## Core Architecture

- `src/lib/rag.ts`
  - Ingestion pipeline: chunking + embeddings + Chroma storage
  - Retrieval pipeline: similarity search over knowledge chunks
- `src/lib/agent.ts`
  - Tool-calling agent with `search_knowledge_base`
  - Agent decides when to call retrieval tool vs answer directly
- `src/app/api/ingest/route.ts`
  - Accepts pasted text and/or uploaded `.txt` file
  - Stores chunk embeddings in Chroma
- `src/app/api/chat/route.ts`
  - Runs tool-calling agent
  - Streams SSE events with answer tokens + retrieval visibility metadata

## Environment Variables

Copy `.env.example` to `.env.local` and configure keys.

```bash
cp .env.example .env.local
```

Required:

- `LLM_PROVIDER=gemini` (default) or `openai`
- `GEMINI_API_KEY` when provider is `gemini`
- `OPENAI_API_KEY` when provider is `openai`

Chroma:

- Leave `CHROMA_URL` empty (or set `memory`) for in-memory dev mode
- Set `CHROMA_URL=http://localhost:8000` to use a local Chroma server
- Collection name and tenant/database are configurable

## Run Chroma Locally

Use the included compose file when running Chroma server mode:

```bash
docker compose up -d
```

Default app behavior with empty `CHROMA_URL` is in-memory vector storage in the app process.
For Chroma server mode, optional persistence is available by toggling `IS_PERSISTENT=TRUE` and mounting a host volume in `docker-compose.yml`.

## Run the App

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## API Contracts

### `POST /api/ingest`

- Multipart form data:
  - `text` (optional)
  - `file` (optional, `.txt`)
  - `source` (optional)
- JSON:
  - `{ "text": "...", "source": "..." }`

Response includes `totalChunks` and per-item chunk counts.

### `POST /api/chat`

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What does the uploaded document say about X?"
    }
  ]
}
```

Returns a Server-Sent Events stream (`text/event-stream`) with these events:

- `meta`:
  - `{ "usedKnowledgeBaseTool": boolean, "sources": [{ "source": string, "score": number, "snippet": string, "id"?: string }] }`
- `token`:
  - `{ "token": string }`
- `done`:
  - `{ "completed": true }`
- `error`:
  - `{ "error": string }`

This enables the UI to show whether retrieval was used and display source snippets for each assistant response.

## Notes

- This app is intentionally single-user and has no authentication.
- Retrieval and agent logic are modular and reusable.
- The vector store is initialized once and reused across requests.
