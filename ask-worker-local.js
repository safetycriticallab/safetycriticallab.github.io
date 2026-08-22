/**
 * SCL Ask Worker (local model variant) — answers questions about SCL grounded
 * in the published FAQ, using a self-hosted Ollama model instead of the
 * Anthropic API. Same contract as ask-worker.js, so search.html needs no
 * changes beyond ASK_ENDPOINT.
 *
 * Architecture:
 *   browser -> this worker (CORS, validation, grounding)
 *           -> Cloudflare Tunnel hostname (protected by Cloudflare Access)
 *           -> Ollama on the host machine (llama3.1:8b)
 *
 * Deploy:
 *   1. Create a new Cloudflare Worker named "scl-ask" and paste this file.
 *   2. Settings > Variables:
 *        OLLAMA_URL              plain var, e.g. https://ollama.safetycriticallabs.com
 *        CF_ACCESS_CLIENT_ID     secret, from the Zero Trust service token
 *        CF_ACCESS_CLIENT_SECRET secret, from the Zero Trust service token
 *        OLLAMA_MODEL            optional plain var, defaults to llama3.1:latest
 *   3. Required before going live: a WAF rate-limiting rule for this worker's
 *      route (e.g. 10 requests/minute per IP). Ollama serializes requests, so
 *      without it one scripted loop can monopolize the model.
 *   4. Put the deployed URL into ASK_ENDPOINT in search.html.
 *
 * Tunnel notes (on the machine running Ollama):
 *   - cloudflared ingress must set httpHostHeader to localhost:11434 or
 *     Ollama rejects the forwarded Host header.
 *   - Protect the hostname with a Cloudflare Access application whose policy
 *     is Service Auth + the service token above. Ollama itself has no auth.
 *
 * Contract: POST {question: string} -> 200 {answer: string}
 *           4xx/5xx {error: string}
 */

const DEFAULT_MODEL = 'llama3.1:latest';
const MAX_QUESTION_CHARS = 500;
const FAQ_URL = 'https://safetycriticallabs.com/faq.json';
const UPSTREAM_TIMEOUT_MS = 90000; // cold start: model load + prompt eval can near a minute

const ALLOWED_ORIGINS = [
  'https://safetycriticallabs.com',
  'https://www.safetycriticallabs.com',
  'https://safetycriticallabs.github.io',
];

// Exact-host match; a bare startsWith('http://localhost') would also admit
// registrable domains like http://localhost.evil.com.
const LOCALHOST_ORIGIN_RE = /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/;

function originAllowed(origin) {
  return ALLOWED_ORIGINS.includes(origin) || LOCALHOST_ORIGIN_RE.test(origin);
}

function corsHeaders(origin) {
  return {
    'Access-Control-Allow-Origin': originAllowed(origin) ? origin : ALLOWED_ORIGINS[0],
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin',
    'Content-Type': 'application/json',
  };
}

function reply(status, body, origin) {
  return new Response(JSON.stringify(body), { status, headers: corsHeaders(origin) });
}

const SYSTEM_INSTRUCTIONS = `You are the question-answering assistant on the public website of Safety Critical Labs (SCL), an independent certification authority for AI in safety-critical systems. SCL publishes the AI Requirements Framework: ten core requirement areas (AI-1 through AI-10) plus three conditional architecture and paradigm areas (AI-11 multi-model, AI-12 neural networks, AI-13 continuous learning), anchored in standards like DO-178C, ISO 26262, and NPR 7150.2D.

Answer using ONLY the reference entries provided below. Rules:
- Keep answers to 2 to 5 short sentences, plain text, no markdown formatting, no em dashes.
- If the reference entries do not cover the question, say so plainly and point the visitor to the contact page at /contact.html. Never guess or invent facts, certifications, clients, partnerships, or status.
- Do not overstate SCL's status. SCL is pre-accreditation: ANAB intake is on file and a fee estimate was received, but formal engagement is deferred until certification volume supports it.
- If asked something unrelated to SCL, AI assurance, or safety-critical certification, politely decline and redirect to what you can help with.

Reference entries follow.`;

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }
    if (request.method !== 'POST') {
      return reply(405, { error: 'POST only' }, origin);
    }
    if (origin && !originAllowed(origin)) {
      return reply(403, { error: 'Origin not allowed' }, origin);
    }
    if (!env.OLLAMA_URL) {
      return reply(500, { error: 'Worker not configured' }, origin);
    }

    // Bound the body before buffering it; a legitimate question is under 4KB.
    const bodyLen = parseInt(request.headers.get('Content-Length') || '0', 10);
    if (!bodyLen || bodyLen > 4096) {
      return reply(413, { error: 'Missing or oversized request body' }, origin);
    }

    let question;
    try {
      const body = await request.json();
      question = typeof body.question === 'string' ? body.question.trim() : '';
    } catch (e) {
      return reply(400, { error: 'Invalid JSON body' }, origin);
    }
    if (!question) {
      return reply(400, { error: 'Missing question' }, origin);
    }
    if (question.length > MAX_QUESTION_CHARS) {
      return reply(400, { error: 'Question too long (max ' + MAX_QUESTION_CHARS + ' characters)' }, origin);
    }

    // FAQ is the grounding corpus; edge-cache it so we do not refetch per request.
    let faqText;
    try {
      const faqResp = await fetch(FAQ_URL, { cf: { cacheTtl: 300, cacheEverything: true } });
      if (!faqResp.ok) throw new Error('faq ' + faqResp.status);
      faqText = await faqResp.text();
    } catch (e) {
      return reply(503, { error: 'Reference material unavailable, try again shortly' }, origin);
    }

    const upstreamHeaders = { 'Content-Type': 'application/json' };
    // Service-token auth for the Cloudflare Access application in front of
    // the tunnel; without these, Access turns requests away before Ollama.
    if (env.CF_ACCESS_CLIENT_ID && env.CF_ACCESS_CLIENT_SECRET) {
      upstreamHeaders['CF-Access-Client-Id'] = env.CF_ACCESS_CLIENT_ID;
      upstreamHeaders['CF-Access-Client-Secret'] = env.CF_ACCESS_CLIENT_SECRET;
    } else {
      // Legitimate only when testing against an unprotected endpoint; in
      // production a missing secret shows up here, not as a fake outage.
      console.log('CF Access credentials not set; calling upstream unauthenticated');
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

    let apiResp;
    try {
      apiResp = await fetch(env.OLLAMA_URL.replace(/\/+$/, '') + '/api/chat', {
        method: 'POST',
        headers: upstreamHeaders,
        signal: controller.signal,
        body: JSON.stringify({
          model: env.OLLAMA_MODEL || DEFAULT_MODEL,
          stream: false,
          keep_alive: '1h',
          // num_ctx must clear the ~3.4k-token FAQ context; Ollama's default
          // window would silently truncate the grounding.
          options: { temperature: 0.2, num_ctx: 8192, num_predict: 300 },
          messages: [
            { role: 'system', content: SYSTEM_INSTRUCTIONS + '\n\n' + faqText },
            { role: 'user', content: question },
          ],
        }),
      });
    } catch (e) {
      clearTimeout(timer);
      if (e && e.name === 'AbortError') {
        return reply(504, { error: 'The assistant took too long, try again' }, origin);
      }
      return reply(502, { error: 'The assistant is offline right now, try again later' }, origin);
    }

    if (!apiResp.ok) {
      clearTimeout(timer);
      console.log('upstream error', apiResp.status);
      return reply(502, { error: 'The assistant could not process that question' }, origin);
    }

    // Keep the abort timer armed until the body is fully read; aborting the
    // controller also cancels a trickling response stream.
    let data;
    try {
      data = await apiResp.json();
    } catch (e) {
      return reply(502, { error: 'The assistant could not process that question' }, origin);
    } finally {
      clearTimeout(timer);
    }

    const answer = (data.message && typeof data.message.content === 'string')
      ? data.message.content.trim()
      : '';

    if (!answer) {
      return reply(502, { error: 'Empty response from the assistant' }, origin);
    }

    return reply(200, { answer: answer }, origin);
  },
};
