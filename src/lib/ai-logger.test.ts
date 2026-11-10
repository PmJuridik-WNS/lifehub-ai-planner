import { afterEach, describe, expect, it, vi } from "vitest";
import { AiCallError, classifyAiError, httpAiError, logAiSkipped, withAiLogging } from "./ai-logger.server";

const context = { provider: "perplexity" as const, operation: "text" as const };
afterEach(() => vi.restoreAllMocks());

describe("AI metadata logging", () => {
  it.each([[401, "auth"], [402, "payment"], [403, "forbidden"], [429, "rate_limit"], [503, "server"], [400, "http"]])("classifies HTTP %s", (status, kind) => {
    expect(classifyAiError(httpAiError(Number(status)))).toBe(kind);
  });
  it("does not infer HTTP status from error text", () => {
    expect(classifyAiError(new Error("401 429 500 private content"))).toBe("unknown");
    expect(classifyAiError(new TypeError("private URL"))).toBe("network");
    expect(classifyAiError(new SyntaxError("private response"))).toBe("invalid_response");
    const timeout = new Error("private content");
    timeout.name = "TimeoutError";
    expect(classifyAiError(timeout)).toBe("timeout");
  });
  it("logs one failure without copying error content or extra context fields", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const error = new Error("pplx-secret private patient prompt");
    const extraContext = { ...context, prompt: "private prompt", apiKey: "pplx-secret" };
    await expect(withAiLogging(extraContext, async () => { throw error; })).rejects.toBe(error);
    expect(warn).toHaveBeenCalledTimes(1);
    const line = String(warn.mock.calls[0][0]);
    expect(line).not.toContain("pplx-secret");
    expect(line).not.toContain("private");
    expect(line).not.toContain("stack");
    expect(JSON.parse(line)).toMatchObject({ event: "ai_call_failed", status: null, errorType: "unknown" });
  });
  it("records status and attempt without raw response data", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    await expect(withAiLogging({ ...context, attempt: 2 }, async () => { throw httpAiError(429); })).rejects.toBeInstanceOf(AiCallError);
    expect(JSON.parse(String(warn.mock.calls[0][0]))).toMatchObject({ status: 429, errorType: "rate_limit", attempt: 2 });
  });
  it("returns the result but never logs the result", async () => {
    const info = vi.spyOn(console, "info").mockImplementation(() => {});
    expect(await withAiLogging(context, async () => "private answer")).toBe("private answer");
    expect(info).toHaveBeenCalledTimes(1);
    expect(String(info.mock.calls[0][0])).not.toContain("private answer");
  });
  it("marks missing keys as skipped with zero duration", () => {
    const info = vi.spyOn(console, "info").mockImplementation(() => {});
    logAiSkipped(context);
    expect(JSON.parse(String(info.mock.calls[0][0]))).toMatchObject({ event: "ai_call_skipped", reason: "missing_key", durationMs: 0 });
  });
  it("does not fail a successful task when the log sink throws", async () => {
    vi.spyOn(console, "info").mockImplementation(() => { throw new Error("sink unavailable"); });
    expect(await withAiLogging(context, async () => "ok")).toBe("ok");
  });
});
