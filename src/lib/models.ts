import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { getAppConfig } from "@/lib/config";

class HybridEmbeddings implements EmbeddingsInterface {
  private readonly openaiEmbeddings?: OpenAIEmbeddings;
  private readonly geminiEmbeddings?: GoogleGenerativeAIEmbeddings;
  private readonly primarySource: "openai" | "gemini";

  constructor(
    preferredPrimary: "openai" | "gemini",
    openaiApiKey: string | undefined,
    openaiModel: string,
    geminiApiKey: string | undefined,
    geminiModel: string,
  ) {
    const canUseOpenAI = Boolean(openaiApiKey);
    const canUseGemini = Boolean(geminiApiKey);

    if (!canUseOpenAI && !canUseGemini) {
      throw new Error("No embedding API key provided (OpenAI or Gemini)");
    }

    const effectivePrimary =
      preferredPrimary === "gemini"
        ? canUseGemini
          ? "gemini"
          : "openai"
        : canUseOpenAI
          ? "openai"
          : "gemini";

    this.primarySource = effectivePrimary;

    if (canUseOpenAI) {
      this.openaiEmbeddings = new OpenAIEmbeddings({
        apiKey: openaiApiKey,
        model: openaiModel,
        maxRetries: 0,
        timeout: 15000,
      });
    }

    if (canUseGemini) {
      this.geminiEmbeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: geminiApiKey,
        model: geminiModel,
        maxRetries: 0,
      });
    }
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    if (this.primarySource === "openai" && this.openaiEmbeddings) {
      try {
        return await this.openaiEmbeddings.embedDocuments(texts);
      } catch (error) {
        if (!this.geminiEmbeddings) throw error;
        return this.geminiEmbeddings.embedDocuments(texts);
      }
    }

    if (this.geminiEmbeddings) {
      try {
        return await this.geminiEmbeddings.embedDocuments(texts);
      } catch (error) {
        if (!this.openaiEmbeddings) throw error;
        return this.openaiEmbeddings.embedDocuments(texts);
      }
    }

    throw new Error("No embedding provider available");
  }

  async embedQuery(text: string): Promise<number[]> {
    if (this.primarySource === "openai" && this.openaiEmbeddings) {
      try {
        return await this.openaiEmbeddings.embedQuery(text);
      } catch (error) {
        if (!this.geminiEmbeddings) throw error;
        return this.geminiEmbeddings.embedQuery(text);
      }
    }

    if (this.geminiEmbeddings) {
      try {
        return await this.geminiEmbeddings.embedQuery(text);
      } catch (error) {
        if (!this.openaiEmbeddings) throw error;
        return this.openaiEmbeddings.embedQuery(text);
      }
    }

    throw new Error("No embedding provider available");
  }
}

export function getChatModel(): ChatOpenAI | ChatGoogleGenerativeAI {
  const config = getAppConfig();

  if (config.provider === "openai") {
    return new ChatOpenAI({
      apiKey: config.openAIApiKey,
      model: config.openAIChatModel,
      temperature: 0.2,
      maxTokens: 120,
      maxRetries: 0,
      timeout: 15000,
    });
  }

  return new ChatGoogleGenerativeAI({
    apiKey: config.geminiApiKey,
    model: config.geminiChatModel,
    temperature: 0.2,
    maxOutputTokens: 120,
    maxRetries: 0,
  });
}

export function getEmbeddingsModel(): EmbeddingsInterface {
  const config = getAppConfig();

  return new HybridEmbeddings(
    config.provider,
    config.openAIApiKey,
    config.openAIEmbeddingModel,
    config.geminiApiKey,
    config.geminiEmbeddingModel,
  );
}
