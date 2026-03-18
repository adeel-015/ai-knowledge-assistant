export type Provider = "gemini" | "openai";

export type ChatMessageRole = "user" | "assistant";

export interface ChatMessage {
  role: ChatMessageRole;
  content: string;
  metadata?: AssistantMessageMetadata;
}

export interface SourceSnippet {
  id?: string;
  source: string;
  score: number;
  snippet: string;
}

export interface AssistantMessageMetadata {
  usedKnowledgeBaseTool: boolean;
  sources: SourceSnippet[];
}

export interface IngestResult {
  source: string;
  chunkCount: number;
}

export interface RetrievalChunk {
  id?: string;
  content: string;
  source: string;
  score: number;
}

export interface AgentChatResult {
  answer: string;
  usedKnowledgeBaseTool: boolean;
  retrievedChunks: RetrievalChunk[];
}
