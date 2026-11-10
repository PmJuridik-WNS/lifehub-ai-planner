import { ANDREA_FAST_MODEL, ANDREA_GATEWAY_MODEL } from "@/lib/ai-models";
import { createGoogleAiStudioFetch, googleAiKey } from "@/lib/google-ai.server";
import { AiCallError, httpAiError, logAiSkipped, withAiLogging, type AiOperation } from "./ai-logger.server";
export type { AiProvider } from "./ai-logger.server";
import type { AiProvider } from "./ai-logger.server";

type JsonSchema = { name: string; schema: Record<string, unknown> };
type Message = { role: "system" | "user"; content: string };
type TextOptions = { apiKey?: string; system: string; input: string; jsonSchema?: JsonSchema; model?: string; feature?: string };
export type AiAttachment = { filename: string; mimeType: string; data: string };
type Block = { type: "text"; text: string } | { type: "image_url"; image_url: { url: string } } | { type: "file"; file: { filename: string; file_data: string } };
type RequestFetch = (url: string, init: RequestInit) => Promise<Response>;

async function readStream(body: ReadableStream<Uint8Array>): Promise<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  const consume = (line: string) => {
    if (!line.startsWith("data:")) return;
    const payload = line.slice(5).trim();
    if (!payload || payload === "[DONE]") return;
    let event: { error?: unknown; choices?: { delta?: { content?: unknown } }[] };
    try { event = JSON.parse(payload); }
    catch { throw new AiCallError("invalid_response"); }
    if (!event || typeof event !== "object" || event.error) throw new AiCallError("invalid_response");
    const delta = event.choices?.[0]?.delta?.content;
    if (typeof delta === "string") text += delta;
  };
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) consume(line);
    }
    buffer += decoder.decode();
    if (buffer.trim()) consume(buffer);
    return text.trim();
  } finally {
    try { await reader.cancel(); } catch { /* best effort */ }
    reader.releaseLock();
  }
}

function jsonFormat(schema?: JsonSchema) {
  return schema ? { response_format: { type: "json_schema", json_schema: { name: schema.name, strict: true, schema: schema.schema } } } : {};
}

async function requestText(opts: {
  provider: AiProvider; operation: AiOperation; attempt?: number; url: string;
  key?: string; payload: Record<string, unknown>; request?: RequestFetch;
  clean?: (text: string) => string;
}): Promise<string | null> {
  if (!opts.key) {
    logAiSkipped({ provider: opts.provider, operation: opts.operation, attempt: opts.attempt });
    return null;
  }
  return withAiLogging({ provider: opts.provider, operation: opts.operation, attempt: opts.attempt }, async () => {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (opts.provider === "lovable") {
      headers["Lovable-API-Key"] = opts.key!;
      headers["X-Lovable-AIG-SDK"] = "fetch";
    } else headers.Authorization = `Bearer ${opts.key}`;
    const request = opts.request ?? ((url: string, init: RequestInit) => fetch(url, init));
    const res = await request(opts.url, { method: "POST", headers, body: JSON.stringify(opts.payload) });
    if (!res.ok) {
      try { await res.body?.cancel(); } catch { /* best effort */ }
      throw httpAiError(res.status);
    }
    if (!res.body) throw new AiCallError("empty_response", res.status);
    let text = await readStream(res.body);
    if (opts.clean) text = opts.clean(text);
    if (!text.trim()) throw new AiCallError("empty_response", res.status);
    return text;
  });
}

async function optionalText(opts: Parameters<typeof requestText>[0]): Promise<string | null> {
  try { return await requestText(opts); }
  catch { return null; } // Already logged once by requestText.
}

function pickPerplexityModel(input: string): string {
  const q = input.toLocaleLowerCase("sv-SE");
  if (/(planera|plan |schema|kalender|prioritera|steg för steg|varför)/.test(q)) return "sonar-reasoning-pro";
  if (/(nyhet|senaste|idag|pris|väder|vem är|vad hände|aktuell)/.test(q)) return "sonar-pro";
  return "sonar";
}

async function lovableText(messages: unknown[], schema?: JsonSchema, operation: AiOperation = "text", attempt = 1): Promise<string | null> {
  return requestText({ provider: "lovable", operation, attempt,
    url: "https://ai.gateway.lovable.dev/v1/chat/completions", key: process.env["LOVABLE_API_KEY"],
    payload: { model: ANDREA_GATEWAY_MODEL, stream: true, messages, ...jsonFormat(schema) } });
}

export async function completeTextDetailed(opts: TextOptions): Promise<{ text: string; provider: AiProvider }> {
  const messages: Message[] = [{ role: "system", content: opts.system }, { role: "user", content: opts.input }];
  const delivered = async (text: string, provider: AiProvider) => {
    const { recordAiUsage } = await import("./ai-usage.server");
    await recordAiUsage(provider, opts.feature ?? "text");
    return { text, provider };
  };
  const geminiKey = opts.apiKey ?? googleAiKey();
  const google = await optionalText({ provider: "google", operation: "text",
    key: geminiKey, url: "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    request: geminiKey ? createGoogleAiStudioFetch() : undefined,
    payload: { model: opts.model ?? ANDREA_FAST_MODEL, stream: true, messages, ...jsonFormat(opts.jsonSchema) } });
  if (google) return delivered(google, "google");
  let gateway: string | null = null;
  try { gateway = await lovableText(messages, opts.jsonSchema); } catch { /* Already logged. */ }
  if (gateway) return delivered(gateway, "lovable");
  const openai = await optionalText({ provider: "openai", operation: "text", key: process.env["OPENAI_API_KEY"],
    url: "https://api.openai.com/v1/chat/completions",
    payload: { model: "gpt-5-mini", stream: true, messages, ...jsonFormat(opts.jsonSchema) } });
  if (openai) return delivered(openai, "openai");
  const keys = [process.env["PERPLEXITY_API_KEY_1"], process.env["PERPLEXITY_API_KEY"]].filter((k): k is string => Boolean(k));
  if (!keys.length) logAiSkipped({ provider: "perplexity", operation: "text" });
  for (const [index, key] of keys.entries()) {
    const text = await optionalText({ provider: "perplexity", operation: "text", attempt: index + 1, key,
      url: "https://api.perplexity.ai/chat/completions",
      payload: { model: opts.jsonSchema ? "sonar" : pickPerplexityModel(opts.input), stream: true, messages,
        ...(opts.jsonSchema ? { response_format: { type: "json_schema", json_schema: { schema: opts.jsonSchema.schema } } } : {}) },
      clean: (value) => value.replace(/<think>[\s\S]*?<\/think>/g, "").trim() });
    if (text) return delivered(text, "perplexity");
  }
  try {
    const text = await lovableText(messages, opts.jsonSchema, "text", 2);
    if (text) return delivered(text, "lovable");
  } catch { /* Already logged; never expose provider error bodies. */ }
  throw new Error("AI-tjänsten är inte tillgänglig just nu. Försök igen om en stund.");
}

export async function completeText(opts: TextOptions): Promise<string> {
  return (await completeTextDetailed(opts)).text;
}

export async function completeVision(opts: { system: string; input: string; attachments: AiAttachment[]; jsonSchema?: JsonSchema; feature?: string }): Promise<string> {
  // Preserve the existing requirement for the gateway key.
  if (!process.env["LOVABLE_API_KEY"]) {
    logAiSkipped({ provider: "lovable", operation: "vision" });
    throw new Error("AI-tjänsten är inte tillgänglig just nu. Försök igen om en stund.");
  }
  const blocks: Block[] = [{ type: "text", text: opts.input }];
  let imagesOnly = opts.attachments.length > 0;
  for (const file of opts.attachments) {
    const url = `data:${file.mimeType};base64,${file.data}`;
    if (file.mimeType.startsWith("image/")) blocks.push({ type: "image_url", image_url: { url } });
    else { imagesOnly = false; blocks.push({ type: "file", file: { filename: file.filename, file_data: url } }); }
  }
  const messages = [{ role: "system", content: opts.system }, { role: "user", content: blocks }];
  const geminiKey = googleAiKey();
  if (imagesOnly) {
    const text = await optionalText({ provider: "google", operation: "vision", key: geminiKey,
      url: "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
      request: geminiKey ? createGoogleAiStudioFetch() : undefined,
      payload: { model: ANDREA_FAST_MODEL, stream: true, messages, ...jsonFormat(opts.jsonSchema) } });
    if (text) {
      const { recordAiUsage } = await import("./ai-usage.server");
      await recordAiUsage("google", opts.feature ?? "vision");
      return text;
    }
  }
  let text: string | null = null;
  try { text = await lovableText(messages, opts.jsonSchema, "vision"); }
  catch { /* Already logged. */ }
  if (!text) throw new Error("AI-tjänsten kunde inte läsa bilagan just nu.");
  const { recordAiUsage } = await import("./ai-usage.server");
  await recordAiUsage("lovable", opts.feature ?? "vision");
  return text;
}
