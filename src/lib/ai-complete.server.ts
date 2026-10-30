import { ANDREA_FAST_MODEL, ANDREA_GATEWAY_MODEL } from "@/lib/ai-models";
import { createGoogleAiStudioFetch, googleAiKey } from "@/lib/google-ai.server";

type JsonSchema = { name: string; schema: Record<string, unknown> };

const LOVABLE_FALLBACK_MODEL = ANDREA_GATEWAY_MODEL;

type Message = { role: "system" | "user"; content: string };

/** Läser ut texten ur ett OpenAI-kompatibelt SSE-svar. */
async function readStream(body: ReadableStream<Uint8Array>): Promise<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;
      try {
        const evt = JSON.parse(payload) as {
          choices?: { delta?: { content?: string }; message?: { content?: string } }[];
        };
        const delta = evt.choices?.[0]?.delta?.content ?? "";
        if (delta) text += delta;
      } catch {
        // ofullständigt event – ignorera
      }
    }
  }
  return text.trim();
}

function jsonFormat(jsonSchema?: JsonSchema) {
  if (!jsonSchema) return {};
  return {
    response_format: {
      type: "json_schema",
      json_schema: { name: jsonSchema.name, strict: true, schema: jsonSchema.schema },
    },
  };
}

/**
 * Förstahandsval: Google Gemini via Lovables AI-gateway – den Gemini-väg som
 * alltid fungerar oavsett om användarens egen nyckel är begränsad.
 * Returnerar null vid tillfälliga fel så reservkedjan kan ta vid; anroparen
 * kastar felet om inget annat svarar.
 */
async function completeViaLovable(
  messages: Message[],
  jsonSchema?: JsonSchema,
): Promise<string> {
  const key = process.env["LOVABLE_API_KEY"];
  if (!key) throw new Error("AI-tjänsten är inte tillgänglig just nu. Försök igen om en stund.");

  const res = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Lovable-API-Key": key,
      "X-Lovable-AIG-SDK": "fetch",
    },
    body: JSON.stringify({
      model: LOVABLE_FALLBACK_MODEL,
      stream: true,
      messages,
      ...jsonFormat(jsonSchema),
    }),
  });

  if (!res.ok || !res.body) {
    if (res.status === 429)
      throw new Error("AI-tjänsten är hårt belastad just nu. Försök igen om en stund.");
    if (res.status === 402 || res.status === 403)
      throw new Error("AI-krediterna är slut – fyll på för att fortsätta använda Andrea.");
    throw new Error("AI-tjänsten kunde inte nås just nu. Försök igen om en stund.");
  }

  return readStream(res.body);
}

/**
 * Andrahandsval: användarens eget ChatGPT-konto (OPENAI_API_KEY).
 * Returnerar null om nyckeln saknas eller anropet misslyckas, så att
 * anroparen kan gå vidare i reservkedjan.
 */
async function completeViaOpenAI(
  messages: Message[],
  jsonSchema?: JsonSchema,
): Promise<string | null> {
  const apiKey = process.env["OPENAI_API_KEY"];
  if (!apiKey) return null;
  try {
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "gpt-5.6-terra",
        stream: true,
        messages,
        ...jsonFormat(jsonSchema),
      }),
    });
    if (!res.ok || !res.body) {
      const detail = await res.text().catch(() => "");
      console.warn(`ChatGPT svarade ${res.status}: ${detail.slice(0, 300)}`);
      return null;
    }
    const text = await readStream(res.body);
    return text || null;
  } catch (error) {
    console.warn("ChatGPT misslyckades, provar nästa tjänst.", error);
    return null;
  }
}

/** Förstahandsval: användarens Google-konto (GOOGLE_API_KEY, annars GEMINI_API_KEY). */
async function completeViaGemini(
  messages: Message[],
  jsonSchema: JsonSchema | undefined,
  model: string,
  apiKeyOverride?: string,
): Promise<string | null> {
  const apiKey = apiKeyOverride ?? googleAiKey();

  if (!apiKey) return null;
  try {
    const googleFetch = createGoogleAiStudioFetch();
    const res = await googleFetch(
      "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          stream: true,
          messages,
          ...jsonFormat(jsonSchema),
        }),
      },
    );
    if (!res.ok || !res.body) {
      const detail = await res.text().catch(() => "");
      console.warn(`Google AI Studio svarade ${res.status}: ${detail.slice(0, 300)}`);
      return null;
    }
    const text = await readStream(res.body);
    return text || null;
  } catch (error) {
    console.warn("Google AI Studio misslyckades, provar reservtjänsten.", error);
    return null;
  }
}

/**
 * Kör ett strömmande textanrop i prioritetsordning:
 * Gemini (eget konto, förstahandsval) → Gemini via Lovables gateway →
 * ChatGPT (eget konto) → Perplexity.
 */
export type AiProvider = "openai" | "google" | "perplexity" | "lovable";

/** Som completeText, men berättar även vilken AI-tjänst som levererade svaret. */
export async function completeTextDetailed(opts: {
  apiKey?: string;
  system: string;
  input: string;
  jsonSchema?: JsonSchema;
  model?: string;
  feature?: string;
}): Promise<{ text: string; provider: AiProvider }> {
  const messages: Message[] = [
    { role: "system", content: opts.system },
    { role: "user", content: opts.input },
  ];

  const delivered = async (text: string, provider: AiProvider) => {
    const { recordAiUsage } = await import("./ai-usage.server");
    await recordAiUsage(provider, opts.feature ?? "text");
    return { text, provider };
  };

  // Först: användarens egen Gemini-nyckel (GOOGLE_API_KEY).
  const geminiText = await completeViaGemini(
    messages,
    opts.jsonSchema,
    opts.model ?? ANDREA_FAST_MODEL,
    opts.apiKey,
  );
  if (geminiText) return delivered(geminiText, "google");

  // Reserv: Gemini via Lovables gateway – fungerar även om den egna nyckeln är begränsad.
  try {
    const gatewayText = await completeViaLovable(messages, opts.jsonSchema);
    if (gatewayText) return delivered(gatewayText, "lovable");
  } catch (error) {
    console.warn("Gemini via Lovable misslyckades, provar reservkedjan.", error);
  }

  const openAiText = await completeViaOpenAI(messages, opts.jsonSchema);
  if (openAiText) return delivered(openAiText, "openai");

  const pplxText = await completeViaPerplexity(messages, opts.jsonSchema, opts.input);
  if (pplxText) return delivered(pplxText, "perplexity");

  return delivered(await completeViaLovable(messages, opts.jsonSchema), "lovable");
}

export async function completeText(opts: {
  apiKey?: string;
  system: string;
  input: string;
  jsonSchema?: JsonSchema;
  model?: string;
  feature?: string;
}): Promise<string> {
  return (await completeTextDetailed(opts)).text;
}

/** Välj Perplexity-modell efter typ av fråga. */
function pickPerplexityModel(input: string): string {
  const q = input.toLocaleLowerCase("sv-SE");
  if (/(planera|plan |schema|kalender|prioritera|steg för steg|varför)/.test(q))
    return "sonar-reasoning-pro";
  if (/(nyhet|senaste|idag|pris|väder|vem är|vad hände|aktuell)/.test(q)) return "sonar-pro";
  return "sonar";
}

/** Tredjehandsval: användarens Perplexity-konto, före Lovable-reserven. */
async function completeViaPerplexity(
  messages: Message[],
  jsonSchema: JsonSchema | undefined,
  input: string,
): Promise<string | null> {
  const keys = [process.env["PERPLEXITY_API_KEY_1"], process.env["PERPLEXITY_API_KEY"]].filter(
    (k): k is string => Boolean(k),
  );
  const model = jsonSchema ? "sonar" : pickPerplexityModel(input);
  for (const key of keys) {
    try {
      const res = await fetch("https://api.perplexity.ai/chat/completions", {
        method: "POST",
        headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          stream: true,
          messages,
          ...(jsonSchema
            ? { response_format: { type: "json_schema", json_schema: { schema: jsonSchema.schema } } }
            : {}),
        }),
      });
      if (!res.ok || !res.body) {
        const detail = await res.text().catch(() => "");
        console.warn(`Perplexity svarade ${res.status}: ${detail.slice(0, 300)}`);
        continue;
      }
      let text = await readStream(res.body);
      text = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
      if (text) return text;
    } catch (error) {
      console.warn("Perplexity misslyckades, provar reservtjänsten.", error);
    }
  }
  return null;
}

export type AiAttachment = { filename: string; mimeType: string; data: string };

type Block =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } }
  | { type: "file"; file: { filename: string; file_data: string } };

/**
 * Multimodalt anrop (PDF och bilder). Används när ett mejl har bilagor –
 * där ligger oftast kvittot eller fakturan. Prioritet: Gemini (eget konto,
 * endast bilder) → Lovable AI (reserv, stöder även PDF).
 */
export async function completeVision(opts: {
  system: string;
  input: string;
  attachments: AiAttachment[];
  jsonSchema?: JsonSchema;
  feature?: string;
}): Promise<string> {
  const key = process.env["LOVABLE_API_KEY"];
  if (!key) throw new Error("AI-tjänsten är inte tillgänglig just nu. Försök igen om en stund.");

  const blocks: Block[] = [{ type: "text", text: opts.input }];
  let imagesOnly = opts.attachments.length > 0;
  for (const file of opts.attachments) {
    const url = `data:${file.mimeType};base64,${file.data}`;
    if (file.mimeType.startsWith("image/")) blocks.push({ type: "image_url", image_url: { url } });
    else {
      imagesOnly = false;
      blocks.push({ type: "file", file: { filename: file.filename, file_data: url } });
    }
  }

  // Förstahandsval för bilder: användarens eget Google-konto.
  const geminiKey = googleAiKey();

  if (imagesOnly && geminiKey) {
    try {
      const googleFetch = createGoogleAiStudioFetch();
      const geminiRes = await googleFetch(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${geminiKey}`,
          },
          body: JSON.stringify({
            model: ANDREA_FAST_MODEL,
            stream: true,
            messages: [
              { role: "system", content: opts.system },
              { role: "user", content: blocks },
            ],
            ...jsonFormat(opts.jsonSchema),
          }),
        },
      );
      if (geminiRes.ok && geminiRes.body) {
        const text = await readStream(geminiRes.body);
        if (text) {
          const { recordAiUsage } = await import("./ai-usage.server");
          await recordAiUsage("google", opts.feature ?? "vision");
          return text;
        }
      } else {
        const detail = await geminiRes.text().catch(() => "");
        console.warn(`Gemini (bilaga) svarade ${geminiRes.status}: ${detail.slice(0, 300)}`);
      }
    } catch (error) {
      console.warn("Gemini (bilaga) misslyckades, provar reservtjänsten.", error);
    }
  }

  const res = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Lovable-API-Key": key,
      "X-Lovable-AIG-SDK": "fetch",
    },
    body: JSON.stringify({
      model: LOVABLE_FALLBACK_MODEL,
      stream: true,
      messages: [
        { role: "system", content: opts.system },
        { role: "user", content: blocks },
      ],
      ...jsonFormat(opts.jsonSchema),
    }),
  });

  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    console.warn(`Lovable AI (bilaga) svarade ${res.status}: ${detail.slice(0, 300)}`);
    if (res.status === 402 || res.status === 403)
      throw new Error("AI-krediterna är slut – fyll på för att fortsätta använda Andrea.");
    throw new Error("AI-tjänsten kunde inte läsa bilagan just nu.");
  }

  const text = await readStream(res.body);
  if (text) {
    const { recordAiUsage } = await import("./ai-usage.server");
    await recordAiUsage("lovable", opts.feature ?? "vision");
  }
  return text;
}
