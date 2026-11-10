import { AiCallError, httpAiError, logAiSkipped, withAiLogging } from "./ai-logger.server";

export type WebSearchResult = { answer: string; sources: string[]; model: string };

export async function searchWeb(query: string, mode: "web" | "academic" = "web"): Promise<WebSearchResult> {
  const keys = [process.env["PERPLEXITY_API_KEY_1"], process.env["PERPLEXITY_API_KEY"]].filter((k): k is string => Boolean(k));
  if (!keys.length) {
    logAiSkipped({ provider: "perplexity", operation: "websearch" });
    throw new Error("Perplexity är inte kopplat.");
  }
  for (const [index, key] of keys.entries()) {
    try {
      return await withAiLogging({ provider: "perplexity", operation: "websearch", attempt: index + 1 }, async () => {
        const res = await fetch("https://api.perplexity.ai/chat/completions", {
          method: "POST",
          headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
          body: JSON.stringify({ model: "sonar-pro", search_mode: mode === "academic" ? "academic" : undefined,
            messages: [{ role: "system", content: "Svara kort och konkret på svenska. Max 6 meningar. Ta med datum och siffror när det är relevant." }, { role: "user", content: query }] }),
        });
        if (!res.ok) {
          try { await res.body?.cancel(); } catch { /* best effort */ }
          throw httpAiError(res.status);
        }
        const data = await res.json() as {
          choices?: { message?: { content?: unknown } }[];
          citations?: unknown;
          search_results?: { url?: unknown }[];
        };
        if (!data || typeof data !== "object") throw new AiCallError("invalid_response", res.status);
        const answer = data.choices?.[0]?.message?.content;
        if (typeof answer !== "string" || !answer.trim()) throw new AiCallError("empty_response", res.status);
        const candidates = Array.isArray(data.citations) ? data.citations : (Array.isArray(data.search_results) ? data.search_results.map((r) => r?.url) : []);
        const sources = candidates.filter((url): url is string => typeof url === "string");
        return { answer, sources: sources.slice(0, 6), model: "sonar-pro" };
      });
    } catch { /* Already logged once; try the next configured key. */ }
  }
  throw new Error("Webbsökningen kunde inte genomföras just nu. Kontrollera anslutningen eller försök igen senare.");
}
