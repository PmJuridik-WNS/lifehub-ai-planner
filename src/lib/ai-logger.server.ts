export type AiProvider = "google" | "lovable" | "openai" | "perplexity";
export type AiOperation = "text" | "vision" | "websearch";
export type AiFailureKind = "auth" | "payment" | "forbidden" | "rate_limit" | "server" | "http" | "network" | "timeout" | "aborted" | "empty_response" | "invalid_response" | "unknown";

export class AiCallError extends Error {
  constructor(public readonly kind: AiFailureKind, public readonly status?: number) {
    super("AI request failed");
    this.name = "AiCallError";
  }
}

export function classifyAiError(error: unknown): AiFailureKind {
  if (error instanceof AiCallError) return error.kind;
  if (error instanceof Error) {
    if (error.name === "TimeoutError") return "timeout";
    if (error.name === "AbortError") return "aborted";
    if (error instanceof SyntaxError) return "invalid_response";
    if (error instanceof TypeError) return "network";
  }
  return "unknown";
}

export function httpAiError(status: number): AiCallError {
  const kind: AiFailureKind = status === 401 ? "auth" : status === 402 ? "payment" : status === 403 ? "forbidden" : status === 429 ? "rate_limit" : status >= 500 && status <= 599 ? "server" : "http";
  return new AiCallError(kind, status);
}

type Context = { provider: AiProvider; operation: AiOperation; attempt?: number };

function writeLog(context: Context, event: "ai_call_success" | "ai_call_failed" | "ai_call_skipped", durationMs: number, error?: unknown): void {
  // Never spread caller objects or include error messages, response bodies or headers.
  const record = {
    level: event === "ai_call_failed" ? "warn" : "info",
    event,
    provider: context.provider,
    operation: context.operation,
    attempt: context.attempt ?? 1,
    status: error instanceof AiCallError ? error.status ?? null : null,
    errorType: event === "ai_call_failed" ? classifyAiError(error) : null,
    reason: event === "ai_call_skipped" ? "missing_key" : null,
    durationMs: Math.max(0, Math.round(durationMs)),
    timestamp: new Date().toISOString(),
  };
  try {
    const line = JSON.stringify(record);
    if (event === "ai_call_failed") console.warn(line);
    else console.info(line);
  } catch {
    // Logging must not interrupt the application.
  }
}

export function logAiSkipped(context: Context): void {
  writeLog(context, "ai_call_skipped", 0);
}

export async function withAiLogging<T>(context: Context, task: () => Promise<T>): Promise<T> {
  const start = performance.now();
  try {
    const result = await task();
    writeLog(context, "ai_call_success", performance.now() - start);
    return result;
  } catch (error) {
    writeLog(context, "ai_call_failed", performance.now() - start, error);
    throw error;
  }
}
