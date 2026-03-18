import { ChromaClient, type Collection } from "chromadb";
import { promises as fs } from "node:fs";
import path from "node:path";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { getAppConfig } from "@/lib/config";
import { getEmbeddingsModel } from "@/lib/models";

type VectorMetadata = Record<string, unknown>;
type ChromaMetadataValue =
  | string
  | number
  | boolean
  | null
  | string[]
  | number[]
  | boolean[];
type ChromaMetadata = Record<string, ChromaMetadataValue>;

interface QueryResponse {
  ids: string[][];
  documents: (string | null)[][];
  metadatas: VectorMetadata[][];
  distances: (number | null)[][];
}

interface VectorCollectionClient {
  upsert(args: {
    ids: string[];
    embeddings: number[][];
    documents: string[];
    metadatas: VectorMetadata[];
  }): Promise<void>;
  query(args: {
    queryEmbeddings: number[][];
    nResults: number;
    include?: unknown[];
  }): Promise<QueryResponse>;
}

type ChromaQueryArgs = Parameters<Collection["query"]>[0];

export interface VectorStoreContext {
  collection: VectorCollectionClient;
  embeddings: EmbeddingsInterface;
}

let vectorStorePromise: Promise<VectorStoreContext> | null = null;
const inMemoryRecords: Array<{
  id: string;
  document: string;
  metadata: VectorMetadata;
  embedding: number[];
}> = [];
const LOCAL_VECTOR_STORE_DIR = path.join(process.cwd(), ".rag");
const LOCAL_VECTOR_STORE_FILE = path.join(
  LOCAL_VECTOR_STORE_DIR,
  "in-memory-vectors.json",
);

let inMemoryLoadedFromDisk = false;

async function loadInMemoryRecordsFromDisk(): Promise<void> {
  if (inMemoryLoadedFromDisk) {
    return;
  }

  try {
    const raw = await fs.readFile(LOCAL_VECTOR_STORE_FILE, "utf8");
    const parsed = JSON.parse(raw) as Array<{
      id?: unknown;
      document?: unknown;
      metadata?: unknown;
      embedding?: unknown;
    }>;

    if (!Array.isArray(parsed)) {
      inMemoryLoadedFromDisk = true;
      return;
    }

    for (const item of parsed) {
      if (
        typeof item.id !== "string" ||
        typeof item.document !== "string" ||
        !Array.isArray(item.embedding) ||
        !item.embedding.every((value) => typeof value === "number")
      ) {
        continue;
      }

      inMemoryRecords.push({
        id: item.id,
        document: item.document,
        metadata:
          item.metadata && typeof item.metadata === "object"
            ? (item.metadata as VectorMetadata)
            : {},
        embedding: item.embedding,
      });
    }
  } catch {
    // Missing/invalid file means we start with an empty local store.
  } finally {
    inMemoryLoadedFromDisk = true;
  }
}

async function persistInMemoryRecordsToDisk(): Promise<void> {
  await fs.mkdir(LOCAL_VECTOR_STORE_DIR, { recursive: true });
  await fs.writeFile(
    LOCAL_VECTOR_STORE_FILE,
    JSON.stringify(inMemoryRecords),
    "utf8",
  );
}

function cosineDistance(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) {
    return 1;
  }

  let dot = 0;
  let aNorm = 0;
  let bNorm = 0;
  for (let index = 0; index < a.length; index += 1) {
    const av = a[index] ?? 0;
    const bv = b[index] ?? 0;
    dot += av * bv;
    aNorm += av * av;
    bNorm += bv * bv;
  }

  if (aNorm === 0 || bNorm === 0) {
    return 1;
  }

  const cosineSimilarity = dot / (Math.sqrt(aNorm) * Math.sqrt(bNorm));
  return 1 - cosineSimilarity;
}

function createInMemoryCollection(): VectorCollectionClient {
  return {
    async upsert({ ids, embeddings, documents, metadatas }) {
      await loadInMemoryRecordsFromDisk();

      for (let index = 0; index < ids.length; index += 1) {
        const record = {
          id: ids[index],
          embedding: embeddings[index],
          document: documents[index],
          metadata: metadatas[index] ?? {},
        };

        const existingIndex = inMemoryRecords.findIndex(
          (item) => item.id === record.id,
        );
        if (existingIndex >= 0) {
          inMemoryRecords[existingIndex] = record;
        } else {
          inMemoryRecords.push(record);
        }
      }

      await persistInMemoryRecordsToDisk();
    },
    async query({ queryEmbeddings, nResults }) {
      await loadInMemoryRecordsFromDisk();

      const queryEmbedding = queryEmbeddings[0];
      if (!queryEmbedding || inMemoryRecords.length === 0) {
        return {
          ids: [[]],
          documents: [[]],
          metadatas: [[]],
          distances: [[]],
        };
      }

      const ranked = [...inMemoryRecords]
        .map((record) => ({
          ...record,
          distance: cosineDistance(queryEmbedding, record.embedding),
        }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, nResults);

      return {
        ids: [ranked.map((item) => item.id)],
        documents: [ranked.map((item) => item.document)],
        metadatas: [ranked.map((item) => item.metadata)],
        distances: [ranked.map((item) => item.distance)],
      };
    },
  };
}

function createChromaAdapter(collection: Collection): VectorCollectionClient {
  function toChromaMetadata(metadata: VectorMetadata): ChromaMetadata {
    const normalized: ChromaMetadata = {};

    for (const [key, value] of Object.entries(metadata)) {
      if (
        typeof value === "string" ||
        typeof value === "number" ||
        typeof value === "boolean" ||
        value === null
      ) {
        normalized[key] = value;
        continue;
      }

      if (
        Array.isArray(value) &&
        value.every(
          (item) =>
            typeof item === "string" ||
            typeof item === "number" ||
            typeof item === "boolean",
        )
      ) {
        normalized[key] = value as string[] | number[] | boolean[];
      }
    }

    return normalized;
  }

  return {
    async upsert(args) {
      await collection.upsert({
        ...args,
        metadatas: args.metadatas.map((metadata) => toChromaMetadata(metadata)),
      });
    },
    async query(args) {
      const result = await collection.query({
        queryEmbeddings: args.queryEmbeddings,
        nResults: args.nResults,
        include: args.include as ChromaQueryArgs["include"],
      });

      return {
        ids: result.ids ?? [[]],
        documents: (result.documents as (string | null)[][] | undefined) ?? [
          [],
        ],
        metadatas: (result.metadatas as VectorMetadata[][] | undefined) ?? [[]],
        distances: result.distances ?? [[]],
      };
    },
  };
}

/**
 * Initializes the vector store once and reuses it across all requests.
 * This keeps ingestion and retrieval consistent within the running process.
 */
export async function getVectorStore(): Promise<VectorStoreContext> {
  if (vectorStorePromise) {
    return vectorStorePromise;
  }

  const config = getAppConfig();
  const embeddings = getEmbeddingsModel();

  vectorStorePromise = (async () => {
    if (config.useInMemoryVectorStore) {
      return {
        collection: createInMemoryCollection(),
        embeddings,
      };
    }

    const client = new ChromaClient({
      path: config.chromaUrl,
      tenant: config.chromaTenant,
      database: config.chromaDatabase,
    });

    const collection = await client.getOrCreateCollection({
      name: config.chromaCollectionName,
    });

    return {
      collection: createChromaAdapter(collection),
      embeddings,
    };
  })();

  return vectorStorePromise;
}
