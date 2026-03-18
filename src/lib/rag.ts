import { randomUUID } from "node:crypto";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import type { IngestResult, RetrievalChunk } from "@/types/rag";
import { getVectorStore } from "@/lib/vectorStore";

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 900,
  chunkOverlap: 180,
});

export interface IngestInput {
  content: string;
  source: string;
}

/**
 * Splits raw content into chunks and stores embeddings in Chroma.
 */
export async function ingestKnowledge(
  input: IngestInput,
): Promise<IngestResult> {
  const normalized = input.content.trim();
  if (!normalized) {
    throw new Error("Document content is empty.");
  }

  const chunks = await splitter.createDocuments(
    [normalized],
    [
      {
        source: input.source,
        ingestedAt: new Date().toISOString(),
      },
    ],
  );

  const ids = chunks.map(() => randomUUID());
  const { collection, embeddings } = await getVectorStore();
  const embeddingVectors = await embeddings.embedDocuments(
    chunks.map((chunk) => chunk.pageContent),
  );

  await collection.upsert({
    ids,
    embeddings: embeddingVectors,
    documents: chunks.map((chunk) => chunk.pageContent),
    metadatas: chunks.map((chunk) => chunk.metadata),
  });

  return {
    source: input.source,
    chunkCount: chunks.length,
  };
}

/**
 * Retrieves the most similar chunks for the user query.
 * The returned chunks are the grounding context used by chat answers.
 */
export async function retrieveRelevantChunks(
  query: string,
  topK = 4,
): Promise<RetrievalChunk[]> {
  const normalized = query.trim();
  if (!normalized) {
    return [];
  }

  const { collection, embeddings } = await getVectorStore();
  const queryEmbedding = await embeddings.embedQuery(normalized);
  const result = await collection.query({
    queryEmbeddings: [queryEmbedding],
    nResults: topK,
    include: ["documents", "metadatas", "distances"],
  });

  const documents = result.documents?.[0] ?? [];
  const metadatas = result.metadatas?.[0] ?? [];
  const ids = result.ids?.[0] ?? [];
  const distances = result.distances?.[0] ?? [];

  return documents.flatMap((content, index) => {
    if (typeof content !== "string") {
      return [];
    }

    const metadata = metadatas[index] ?? {};
    const distance =
      typeof distances[index] === "number" ? distances[index] : 1;

    return [
      {
        id: ids[index],
        content,
        source:
          metadata && typeof metadata.source === "string"
            ? metadata.source
            : "unknown-source",
        score: 1 / (1 + distance),
      },
    ];
  });
}

export function formatChunksForPrompt(chunks: RetrievalChunk[]): string {
  if (!chunks.length) {
    return "No relevant knowledge-base context found.";
  }

  return chunks
    .map((chunk, index) => {
      return [
        `Chunk ${index + 1}`,
        `Source: ${chunk.source}`,
        `Score: ${chunk.score.toFixed(4)}`,
        chunk.content,
      ].join("\n");
    })
    .join("\n\n---\n\n");
}
