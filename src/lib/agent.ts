import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import type { AgentChatResult, ChatMessage, RetrievalChunk } from "@/types/rag";
import { formatChunksForPrompt, retrieveRelevantChunks } from "@/lib/rag";
import { getChatModel } from "@/lib/models";

const MIN_RETRIEVAL_SCORE_FOR_GROUNDED_ANSWER = 0.12;
const HIGH_CONFIDENCE_RETRIEVAL_SCORE = 0.75;

function looksLikeKnowledgeQuestion(input: string): boolean {
  const normalized = input.trim().toLowerCase();
  if (!normalized) {
    return false;
  }

  const trivialChatPatterns = [
    /^hi\b/,
    /^hello\b/,
    /^hey\b/,
    /^thanks?\b/,
    /^how are you\b/,
    /^good (morning|afternoon|evening)\b/,
  ];

  if (trivialChatPatterns.some((pattern) => pattern.test(normalized))) {
    return false;
  }

  const knowledgeCues = [
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "explain",
    "define",
    "difference",
    "capital",
    "meaning",
  ];

  return (
    normalized.includes("?") ||
    knowledgeCues.some((cue) => normalized.includes(cue))
  );
}

function hasUsefulRetrievalMatch(chunks: RetrievalChunk[]): boolean {
  return chunks.some(
    (chunk) => chunk.score >= MIN_RETRIEVAL_SCORE_FOR_GROUNDED_ANSWER,
  );
}

function tokenizeForOverlap(value: string): string[] {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);
}

function normalizeToken(token: string): string {
  if (token.endsWith("ing") && token.length > 5) {
    return token.slice(0, -3);
  }
  if (token.endsWith("es") && token.length > 4) {
    return token.slice(0, -2);
  }
  if (token.endsWith("s") && token.length > 3) {
    return token.slice(0, -1);
  }
  return token;
}

function hasKeywordOverlap(input: string, chunks: RetrievalChunk[]): boolean {
  if (!chunks.length) {
    return false;
  }

  const stopWords = new Set([
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "tell",
    "me",
  ]);

  const queryTokens = tokenizeForOverlap(input)
    .map(normalizeToken)
    .filter((token) => !stopWords.has(token));
  if (!queryTokens.length) {
    return false;
  }

  const chunkTokenSet = new Set(
    chunks
      .slice(0, 2)
      .flatMap((chunk) => tokenizeForOverlap(chunk.content).map(normalizeToken))
      .filter((token) => !stopWords.has(token)),
  );

  return queryTokens.some((token) => chunkTokenSet.has(token));
}

function shouldUseKnowledgeBase(
  input: string,
  chunks: RetrievalChunk[],
  isKnowledgeQuestion: boolean,
): boolean {
  if (!chunks.length) {
    return false;
  }

  const bestScore = Math.max(...chunks.map((chunk) => chunk.score));
  const hasOverlap = hasKeywordOverlap(input, chunks);

  // Deterministic route for knowledge questions: lexical overlap wins.
  if (isKnowledgeQuestion && hasOverlap) {
    return true;
  }

  // Fallback score-based route for non-overlap cases.
  if (!hasUsefulRetrievalMatch(chunks)) {
    return false;
  }

  return bestScore >= HIGH_CONFIDENCE_RETRIEVAL_SCORE;
}

function toLangChainHistory(messages: ChatMessage[]): BaseMessage[] {
  return messages
    .filter((message) => message.content.trim().length > 0)
    .map((message) => {
      if (message.role === "assistant") {
        return new AIMessage(message.content);
      }
      return new HumanMessage(message.content);
    });
}

/**
 * Deterministic router:
 * 1) always retrieve candidate chunks,
 * 2) use score + lexical overlap gate,
 * 3) route to grounded RAG or direct LLM.
 */
export async function runKnowledgeAgent(
  input: string,
  history: ChatMessage[],
): Promise<AgentChatResult> {
  let retrievedChunks: RetrievalChunk[] = [];
  let usedKnowledgeBaseTool = false;

  const systemInstruction = new SystemMessage(
    [
      "You are AI Knowledge Assistant.",
      "Keep responses concise by default: 3-5 short lines maximum.",
      "Only provide detailed answers when the user explicitly asks for detail, examples, or step-by-step guidance.",
      "Prefer plain, clean formatting with short paragraphs or compact bullet points.",
      "If provided with knowledge-base context, prioritize it.",
      "If context is weak or missing, answer directly and avoid pretending context exists.",
    ].join("\n"),
  );

  const priorMessages = toLangChainHistory(history);
  const model = getChatModel();
  const isKnowledgeQuestion = looksLikeKnowledgeQuestion(input);

  // Always retrieve first to make routing deterministic.
  if (isKnowledgeQuestion) {
    retrievedChunks = await retrieveRelevantChunks(input, 4);
  }

  const bestScore = retrievedChunks.length
    ? Math.max(...retrievedChunks.map((chunk) => chunk.score))
    : 0;
  const overlap = hasKeywordOverlap(input, retrievedChunks);

  const hasRelevantKbContext = shouldUseKnowledgeBase(
    input,
    retrievedChunks,
    isKnowledgeQuestion,
  );

  console.log("[RAG Router]", {
    input,
    isKnowledgeQuestion,
    retrievedCount: retrievedChunks.length,
    bestScore: Number(bestScore.toFixed(4)),
    overlap,
    route: hasRelevantKbContext ? "knowledge-base" : "direct",
  });

  if (hasRelevantKbContext) {
    usedKnowledgeBaseTool = true;

    const groundedResponse = await model.invoke([
      systemInstruction,
      ...priorMessages,
      new HumanMessage(
        [
          `User question: ${input}`,
          "",
          "Knowledge base context:",
          formatChunksForPrompt(retrievedChunks.slice(0, 2)),
          "",
          "Answer using only the provided knowledge base context.",
          "Keep the answer concise (3-5 short lines).",
          "If the context does not directly answer the question, say that briefly instead of guessing.",
        ].join("\n"),
      ),
    ]);

    return {
      answer: stringifyMessageContent(groundedResponse.content),
      usedKnowledgeBaseTool,
      retrievedChunks,
    };
  }

  const directResponse = await model.invoke([
    systemInstruction,
    ...priorMessages,
    new HumanMessage(input),
  ]);

  return {
    answer: stringifyMessageContent(directResponse.content),
    usedKnowledgeBaseTool: false,
    retrievedChunks,
  };
}

function stringifyMessageContent(content: AIMessage["content"]): string {
  if (typeof content === "string") {
    return content;
  }

  return content
    .map((part) => {
      if (typeof part === "string") {
        return part;
      }
      if ("text" in part && typeof part.text === "string") {
        return part.text;
      }
      return "";
    })
    .join(" ")
    .trim();
}
