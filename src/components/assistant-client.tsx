"use client";

import { useMemo, useRef, useState } from "react";
import type {
    AssistantMessageMetadata,
    ChatMessage,
    SourceSnippet,
} from "@/types/rag";

const CHAT_REQUEST_TIMEOUT_MS = 20000;

interface SseEvent {
    event: string;
    data: string;
}

function parseSseBlock(block: string): SseEvent | null {
    const lines = block.split("\n");
    let event = "message";
    const dataLines: string[] = [];

    for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        if (!line || line.startsWith(":")) {
            continue;
        }
        if (line.startsWith("event:")) {
            event = line.slice(6).trim();
            continue;
        }
        if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trimStart());
        }
    }

    if (!dataLines.length) {
        return null;
    }

    return {
        event,
        data: dataLines.join("\n"),
    };
}

async function readSseStream(
    response: Response,
    onEvent: (event: SseEvent) => void
): Promise<void> {
    if (!response.body) {
        const text = await response.text();
        const parsed = parseSseBlock(text);
        if (parsed) {
            onEvent(parsed);
        }
        return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }

        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

        let delimiterIndex = buffer.indexOf("\n\n");
        while (delimiterIndex !== -1) {
            const block = buffer.slice(0, delimiterIndex).trim();
            buffer = buffer.slice(delimiterIndex + 2);
            if (block.length > 0) {
                const parsed = parseSseBlock(block);
                if (parsed) {
                    onEvent(parsed);
                }
            }
            delimiterIndex = buffer.indexOf("\n\n");
        }
    }

    const trailing = buffer.trim();
    if (trailing.length > 0) {
        const parsed = parseSseBlock(trailing);
        if (parsed) {
            onEvent(parsed);
        }
    }
}

function parseSourceSnippets(value: unknown): SourceSnippet[] {
    if (!Array.isArray(value)) {
        return [];
    }

    return value.flatMap((item) => {
        if (!item || typeof item !== "object") {
            return [];
        }

        const id = "id" in item && typeof item.id === "string" ? item.id : undefined;
        const source = "source" in item && typeof item.source === "string" ? item.source : "unknown-source";
        const snippet = "snippet" in item && typeof item.snippet === "string" ? item.snippet : "";
        const score = "score" in item && typeof item.score === "number" ? item.score : 0;

        return [{
            id,
            source,
            snippet,
            score,
        }];
    });
}

export function AssistantClient() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const [ingestText, setIngestText] = useState("");
    const [sourceName, setSourceName] = useState("pasted-text");
    const [file, setFile] = useState<File | null>(null);
    const [isIngesting, setIsIngesting] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [status, setStatus] = useState<string | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const canIngest = useMemo(() => {
        return ingestText.trim().length > 0 || !!file;
    }, [ingestText, file]);

    async function handleIngest() {
        if (!canIngest) {
            return;
        }

        setIsIngesting(true);
        setStatus(null);

        try {
            const formData = new FormData();
            if (ingestText.trim()) {
                formData.append("text", ingestText.trim());
            }
            if (sourceName.trim()) {
                formData.append("source", sourceName.trim());
            }
            if (file) {
                formData.append("file", file);
            }

            const response = await fetch("/api/ingest", {
                method: "POST",
                body: formData,
            });

            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.error ?? "Ingestion failed.");
            }

            setStatus(`Ingested ${payload.totalChunks} chunk(s).`);
            setIngestText("");
            setFile(null);
        } catch (error) {
            setStatus(error instanceof Error ? error.message : "Ingestion failed.");
        } finally {
            setIsIngesting(false);
        }
    }

    async function handleSend() {
        const content = input.trim();
        if (!content || isSending) {
            return;
        }

        const nextMessages: ChatMessage[] = [...messages, { role: "user", content }];
        const assistantIndex = nextMessages.length;
        setMessages([...nextMessages, { role: "assistant", content: "" }]);
        setInput("");
        setIsSending(true);
        setStatus(null);
        const abortController = new AbortController();
        const timeoutId = window.setTimeout(() => {
            abortController.abort();
        }, CHAT_REQUEST_TIMEOUT_MS);

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ messages: nextMessages }),
                signal: abortController.signal,
            });

            if (!response.ok) {
                const payload = await response.json();
                throw new Error(payload.error ?? "Chat request failed.");
            }

            const contentType = response.headers.get("content-type") ?? "";

            let streamed = "";
            let metadata: AssistantMessageMetadata | undefined;

            if (contentType.includes("text/event-stream")) {
                await readSseStream(response, (sseEvent) => {
                    if (sseEvent.event === "meta") {
                        try {
                            const payload = JSON.parse(sseEvent.data) as {
                                usedKnowledgeBaseTool?: unknown;
                                sources?: unknown;
                            };

                            metadata = {
                                usedKnowledgeBaseTool: Boolean(payload.usedKnowledgeBaseTool),
                                sources: parseSourceSnippets(payload.sources),
                            };

                            setMessages((current) => {
                                const updated = [...current];
                                const currentAssistant = updated[assistantIndex];
                                if (!currentAssistant || currentAssistant.role !== "assistant") {
                                    return current;
                                }
                                updated[assistantIndex] = {
                                    role: "assistant",
                                    content: streamed,
                                    metadata,
                                };
                                return updated;
                            });
                        } catch {
                            // Ignore malformed metadata events and keep streaming text.
                        }
                        return;
                    }

                    if (sseEvent.event === "token") {
                        try {
                            const payload = JSON.parse(sseEvent.data) as { token?: unknown };
                            if (typeof payload.token === "string") {
                                streamed += payload.token;
                            }
                        } catch {
                            // Ignore malformed token events.
                        }

                        setMessages((current) => {
                            const updated = [...current];
                            const currentAssistant = updated[assistantIndex];
                            if (!currentAssistant || currentAssistant.role !== "assistant") {
                                return current;
                            }
                            updated[assistantIndex] = {
                                role: "assistant",
                                content: streamed,
                                metadata,
                            };
                            return updated;
                        });
                        return;
                    }

                    if (sseEvent.event === "error") {
                        try {
                            const payload = JSON.parse(sseEvent.data) as { error?: unknown };
                            if (typeof payload.error === "string" && payload.error.trim()) {
                                setStatus(payload.error);
                            }
                        } catch {
                            // Ignore malformed error events.
                        }
                    }
                });
            } else {
                // Compatibility fallback for old non-SSE response shape.
                if (!response.body) {
                    streamed = await response.text();
                } else {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) {
                            break;
                        }
                        streamed += decoder.decode(value, { stream: true });
                        setMessages((current) => {
                            const updated = [...current];
                            const currentAssistant = updated[assistantIndex];
                            if (!currentAssistant || currentAssistant.role !== "assistant") {
                                return current;
                            }
                            updated[assistantIndex] = {
                                role: "assistant",
                                content: streamed,
                                metadata,
                            };
                            return updated;
                        });
                    }
                }
            }

            if (!streamed.trim()) {
                setMessages((current) => {
                    const updated = [...current];
                    const currentAssistant = updated[assistantIndex];
                    if (!currentAssistant || currentAssistant.role !== "assistant") {
                        return current;
                    }
                    updated[assistantIndex] = {
                        role: "assistant",
                        content: "I could not produce a response.",
                        metadata,
                    };
                    return updated;
                });
            }
        } catch (error) {
            const message = error instanceof Error ? error.message : "Chat failed.";
            const friendlyMessage =
                message.includes("aborted") || message.includes("AbortError")
                    ? "Chat request timed out. Check API key/quota or try again."
                    : message;
            setStatus(friendlyMessage);
            setMessages((current) => {
                const updated = [...current];
                const currentAssistant = updated[assistantIndex];
                if (!currentAssistant || currentAssistant.role !== "assistant") {
                    return current;
                }
                updated[assistantIndex] = {
                    role: "assistant",
                    content: `Error: ${friendlyMessage}`,
                };
                return updated;
            });
        } finally {
            window.clearTimeout(timeoutId);
            setIsSending(false);
            inputRef.current?.focus();
        }
    }

    return (
        <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-4 px-4 py-6 md:px-8">
            <header className="rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm backdrop-blur">
                <h1 className="text-2xl font-semibold text-slate-900">AI Knowledge Assistant</h1>
                <p className="mt-1 text-sm text-slate-600">
                    Upload or paste knowledge, then ask grounded questions through the agent.
                </p>
            </header>

            <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-700">
                    Document Ingestion
                </h2>
                <div className="mt-3 grid gap-3 md:grid-cols-2">
                    <label className="flex flex-col gap-2">
                        <span className="text-xs font-medium text-slate-600">Source name</span>
                        <input
                            className="rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-emerald-100 focus:ring"
                            value={sourceName}
                            onChange={(event) => setSourceName(event.target.value)}
                            placeholder="knowledge-note"
                        />
                    </label>
                    <label className="flex flex-col gap-2">
                        <span className="text-xs font-medium text-slate-600">Upload text file (.txt)</span>
                        <input
                            className="rounded-lg border border-slate-300 px-3 py-2 text-sm"
                            type="file"
                            accept=".txt,text/plain"
                            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                        />
                    </label>
                </div>
                <label className="mt-3 flex flex-col gap-2">
                    <span className="text-xs font-medium text-slate-600">Or paste text</span>
                    <textarea
                        className="min-h-28 rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-emerald-100 focus:ring"
                        value={ingestText}
                        onChange={(event) => setIngestText(event.target.value)}
                        placeholder="Paste content to store in the knowledge base..."
                    />
                </label>
                <div className="mt-3 flex items-center gap-3">
                    <button
                        className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-emerald-300"
                        onClick={handleIngest}
                        disabled={!canIngest || isIngesting}
                    >
                        {isIngesting ? "Ingesting..." : "Ingest"}
                    </button>
                    {status ? <p className="text-sm text-slate-600">{status}</p> : null}
                </div>
            </section>

            <section className="flex min-h-[40vh] flex-1 flex-col rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-700">Chat</h2>
                <div className="mt-3 flex-1 space-y-3 overflow-y-auto rounded-lg border border-slate-100 bg-slate-50 p-3">
                    {!messages.length ? (
                        <p className="text-sm text-slate-500">
                            Ask a question after ingesting content. The agent decides when to call
                            search_knowledge_base.
                        </p>
                    ) : null}
                    {messages.map((message, index) => (
                        <div
                            key={`${message.role}-${index}`}
                            className={`max-w-[90%] rounded-xl px-3 py-2 text-sm leading-6 ${message.role === "user"
                                ? "ml-auto bg-emerald-600 text-white"
                                : "bg-white text-slate-800"
                                }`}
                        >
                            <p className="whitespace-pre-wrap wrap-break-words">{message.content.replace(/\\n/g, "\n")}</p>
                            {message.role === "assistant" && message.metadata ? (
                                <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50/80 px-2 py-2 text-xs text-slate-700">
                                    <p className="font-semibold">
                                        {message.metadata.usedKnowledgeBaseTool
                                            ? "Knowledge Base Used"
                                            : "Direct Answer"}
                                    </p>
                                    {message.metadata.usedKnowledgeBaseTool && message.metadata.sources.length > 0 ? (
                                        <div className="mt-2 space-y-2">
                                            {message.metadata.sources.map((source, sourceIndex) => (
                                                <div
                                                    key={`${source.id ?? source.source}-${sourceIndex}`}
                                                    className="rounded-md border border-slate-200 bg-white px-2 py-2"
                                                >
                                                    <p className="font-medium text-slate-800">
                                                        {source.source}
                                                    </p>
                                                    <p className="text-slate-500">
                                                        Relevance: {source.score.toFixed(3)}
                                                    </p>
                                                    <p className="mt-1 text-slate-700">{source.snippet}</p>
                                                </div>
                                            ))}
                                        </div>
                                    ) : null}
                                </div>
                            ) : null}
                        </div>
                    ))}
                </div>
                <div className="mt-3 flex gap-2">
                    <input
                        ref={inputRef}
                        className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-emerald-100 focus:ring"
                        value={input}
                        onChange={(event) => setInput(event.target.value)}
                        onKeyDown={(event) => {
                            if (event.key === "Enter") {
                                event.preventDefault();
                                void handleSend();
                            }
                        }}
                        placeholder="Ask a question..."
                    />
                    <button
                        className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-400"
                        onClick={handleSend}
                        disabled={isSending || !input.trim()}
                    >
                        {isSending ? "Thinking..." : "Send"}
                    </button>
                </div>
            </section>
        </div>
    );
}
