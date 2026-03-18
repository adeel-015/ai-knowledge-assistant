import { z } from "zod";
import type { Provider } from "@/types/rag";

const envSchema = z.object({
  LLM_PROVIDER: z.enum(["gemini", "openai"]).default("gemini"),
  OPENAI_API_KEY: z.string().optional(),
  GEMINI_API_KEY: z.string().optional(),
  OPENAI_CHAT_MODEL: z.string().default("gpt-4o-mini"),
  OPENAI_EMBEDDING_MODEL: z.string().default("text-embedding-3-small"),
  GEMINI_CHAT_MODEL: z.string().default("gemini-2.0-flash"),
  GEMINI_EMBEDDING_MODEL: z.string().default("gemini-embedding-001"),
  CHROMA_URL: z.string().optional(),
  CHROMA_COLLECTION_NAME: z.string().default("ai-knowledge-assistant"),
  CHROMA_TENANT: z.string().optional(),
  CHROMA_DATABASE: z.string().optional(),
});

export interface AppConfig {
  provider: Provider;
  openAIApiKey?: string;
  geminiApiKey?: string;
  openAIChatModel: string;
  openAIEmbeddingModel: string;
  geminiChatModel: string;
  geminiEmbeddingModel: string;
  chromaUrl?: string;
  chromaCollectionName: string;
  chromaTenant?: string;
  chromaDatabase?: string;
  useInMemoryVectorStore: boolean;
}

let cachedConfig: AppConfig | null = null;

export function getAppConfig(): AppConfig {
  if (cachedConfig) {
    return cachedConfig;
  }

  const parsed = envSchema.parse({
    LLM_PROVIDER: process.env.LLM_PROVIDER,
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    GEMINI_API_KEY: process.env.GEMINI_API_KEY,
    OPENAI_CHAT_MODEL: process.env.OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL: process.env.OPENAI_EMBEDDING_MODEL,
    GEMINI_CHAT_MODEL: process.env.GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL: process.env.GEMINI_EMBEDDING_MODEL,
    CHROMA_URL: process.env.CHROMA_URL,
    CHROMA_COLLECTION_NAME: process.env.CHROMA_COLLECTION_NAME,
    CHROMA_TENANT: process.env.CHROMA_TENANT,
    CHROMA_DATABASE: process.env.CHROMA_DATABASE,
  });

  const rawChromaUrl = parsed.CHROMA_URL?.trim();
  const useInMemoryVectorStore =
    !rawChromaUrl || rawChromaUrl.toLowerCase() === "memory";

  if (!useInMemoryVectorStore) {
    z.string().url().parse(rawChromaUrl);
  }

  if (parsed.LLM_PROVIDER === "gemini" && !parsed.GEMINI_API_KEY) {
    throw new Error("LLM_PROVIDER is gemini but GEMINI_API_KEY is missing.");
  }

  if (parsed.LLM_PROVIDER === "openai" && !parsed.OPENAI_API_KEY) {
    throw new Error("LLM_PROVIDER is openai but OPENAI_API_KEY is missing.");
  }

  cachedConfig = {
    provider: parsed.LLM_PROVIDER,
    openAIApiKey: parsed.OPENAI_API_KEY,
    geminiApiKey: parsed.GEMINI_API_KEY,
    openAIChatModel: parsed.OPENAI_CHAT_MODEL,
    openAIEmbeddingModel: parsed.OPENAI_EMBEDDING_MODEL,
    geminiChatModel: parsed.GEMINI_CHAT_MODEL,
    geminiEmbeddingModel: parsed.GEMINI_EMBEDDING_MODEL,
    chromaUrl: useInMemoryVectorStore ? undefined : rawChromaUrl,
    chromaCollectionName: parsed.CHROMA_COLLECTION_NAME,
    chromaTenant: parsed.CHROMA_TENANT,
    chromaDatabase: parsed.CHROMA_DATABASE,
    useInMemoryVectorStore,
  };

  return cachedConfig;
}
