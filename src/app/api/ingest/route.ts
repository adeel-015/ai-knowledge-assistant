import { NextResponse } from "next/server";
import { z } from "zod";
import { ingestKnowledge } from "@/lib/rag";

export const runtime = "nodejs";

const ingestBodySchema = z.object({
  text: z.string().min(1),
  source: z.string().optional(),
});

function badRequest(message: string) {
  return NextResponse.json({ error: message }, { status: 400 });
}

export async function POST(request: Request): Promise<NextResponse> {
  try {
    const contentType = request.headers.get("content-type") ?? "";

    if (contentType.includes("multipart/form-data")) {
      const formData = await request.formData();
      const textInput = String(formData.get("text") ?? "").trim();
      const sourceInput = String(formData.get("source") ?? "").trim();
      const file = formData.get("file");

      const ingestionTargets: Array<{ content: string; source: string }> = [];

      if (textInput.length > 0) {
        ingestionTargets.push({
          content: textInput,
          source: sourceInput || "pasted-text",
        });
      }

      if (file instanceof File && file.size > 0) {
        const fileText = (await file.text()).trim();
        if (fileText.length > 0) {
          ingestionTargets.push({
            content: fileText,
            source: file.name || "uploaded-file",
          });
        }
      }

      if (!ingestionTargets.length) {
        return badRequest("Provide non-empty text or a text file to ingest.");
      }

      const results = await Promise.all(
        ingestionTargets.map((target) => ingestKnowledge(target)),
      );

      return NextResponse.json({
        message: "Knowledge ingested successfully.",
        items: results,
        totalChunks: results.reduce((sum, item) => sum + item.chunkCount, 0),
      });
    }

    const body = ingestBodySchema.parse(await request.json());
    const result = await ingestKnowledge({
      content: body.text,
      source: body.source?.trim() || "pasted-text",
    });

    return NextResponse.json({
      message: "Knowledge ingested successfully.",
      items: [result],
      totalChunks: result.chunkCount,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return badRequest(error.issues.map((issue) => issue.message).join("; "));
    }

    const message =
      error instanceof Error ? error.message : "Unexpected ingestion error.";

    return NextResponse.json({ error: message }, { status: 500 });
  }
}
