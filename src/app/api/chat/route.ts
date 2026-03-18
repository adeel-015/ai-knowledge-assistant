import { NextResponse } from "next/server";
import { z } from "zod";
import { runKnowledgeAgent } from "@/lib/agent";
import type { ChatMessage, SourceSnippet } from "@/types/rag";

export const runtime = "nodejs";

const chatBodySchema = z.object({
  messages: z
    .array(
      z.object({
        role: z.enum(["user", "assistant"]),
        content: z.string().min(1),
      }),
    )
    .min(1),
});

type SseEventName = "meta" | "token" | "done" | "error";

function formatSseEvent(event: SseEventName, payload: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
}

function toSourceSnippets(
  chunks: Array<{
    id?: string;
    source: string;
    score: number;
    content: string;
  }>,
): SourceSnippet[] {
  const deduped = new Map<string, SourceSnippet>();

  for (const chunk of chunks) {
    const snippet = chunk.content.replace(/\s+/g, " ").trim().slice(0, 260);
    const key = `${chunk.source}::${snippet}`;
    const score = Number(chunk.score.toFixed(4));
    const existing = deduped.get(key);

    if (!existing || score > existing.score) {
      deduped.set(key, {
        id: chunk.id,
        source: chunk.source,
        score,
        snippet,
      });
    }
  }

  return [...deduped.values()].sort((a, b) => b.score - a.score).slice(0, 5);
}

function streamSseResponse(args: {
  answer: string;
  usedKnowledgeBaseTool: boolean;
  sources: SourceSnippet[];
}): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const tokens = args.answer.split(/(\s+)/).filter((part) => part.length > 0);

  return new ReadableStream<Uint8Array>({
    start(controller) {
      try {
        controller.enqueue(
          encoder.encode(
            formatSseEvent("meta", {
              usedKnowledgeBaseTool: args.usedKnowledgeBaseTool,
              sources: args.sources,
            }),
          ),
        );

        for (const token of tokens) {
          controller.enqueue(
            encoder.encode(formatSseEvent("token", { token })),
          );
        }

        controller.enqueue(
          encoder.encode(formatSseEvent("done", { completed: true })),
        );
      } catch (error) {
        const rawMessage =
          error instanceof Error
            ? error.message
            : "Unexpected streaming error.";
        controller.enqueue(
          encoder.encode(formatSseEvent("error", { error: rawMessage })),
        );
      } finally {
        controller.close();
      }
    },
  });
}

export async function POST(request: Request): Promise<Response> {
  try {
    const body = chatBodySchema.parse(await request.json());

    const messages = body.messages as ChatMessage[];
    if (!messages.some((message) => message.role === "user")) {
      return NextResponse.json(
        { error: "No user message found in request." },
        { status: 400 },
      );
    }

    const lastMessage = messages[messages.length - 1];
    if (lastMessage.role !== "user" || !lastMessage.content.trim()) {
      return NextResponse.json(
        { error: "Last message must be a user message with content." },
        { status: 400 },
      );
    }

    const input = lastMessage.content.trim();
    const history = messages.slice(0, -1);
    const result = await runKnowledgeAgent(input, history);
    const sourceSnippets = result.usedKnowledgeBaseTool
      ? toSourceSnippets(result.retrievedChunks)
      : [];

    return new Response(
      streamSseResponse({
        answer: result.answer,
        usedKnowledgeBaseTool: result.usedKnowledgeBaseTool,
        sources: sourceSnippets,
      }),
      {
        headers: {
          "Content-Type": "text/event-stream; charset=utf-8",
          "Cache-Control": "no-cache, no-transform",
          Connection: "keep-alive",
        },
      },
    );
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: error.issues.map((issue) => issue.message).join("; ") },
        { status: 400 },
      );
    }

    const rawMessage =
      error instanceof Error ? error.message : "Unexpected chat error.";
    const lowerMessage = rawMessage.toLowerCase();

    if (
      lowerMessage.includes("429") ||
      lowerMessage.includes("quota") ||
      lowerMessage.includes("rate limit")
    ) {
      return NextResponse.json(
        {
          error:
            "LLM quota/rate limit reached. Update billing/quota or switch LLM_PROVIDER to openai with a valid key.",
          details: rawMessage,
        },
        { status: 429 },
      );
    }

    return NextResponse.json({ error: rawMessage }, { status: 500 });
  }
}
