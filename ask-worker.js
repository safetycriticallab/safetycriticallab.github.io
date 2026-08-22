/**
 * SCL Ask Worker — answers questions about SCL grounded in the published FAQ.
 *
 * Deploy (separate from the scl-cors-proxy worker):
 *   1. Create a new Cloudflare Worker (e.g. "scl-ask") and paste this file.
 *   2. Set the API key as a secret:  wrangler secret put ANTHROPIC_API_KEY
 *      (or Dashboard > Worker > Settings > Variables > Add secret)
 *   3. Recommended: add a Cloudflare WAF rate-limiting rule for this worker's
 *      route (e.g. 10 requests/minute per IP) so a scraper cannot run up the bill.
 *   4. Put the deployed URL into ASK_ENDPOINT in search.html.
 *
 * Contract: POST {question: string} -> 200 {answer: string}
 *           4xx/5xx {error: string}
 *
 * Model: claude-opus-4-8. For a lower-cost tier swap MODEL to "claude-haiku-4-5"
 * ($1/$5 per MTok vs $5/$25); answers stay grounded either way since the FAQ
 * context does the heavy lifting.
 */

const MODEL = 'claude-opus-4-8';
const MAX_TOKENS = 600;
const MAX_QUESTION_CHARS = 500;
const FAQ_URL = 'https://safetycriticallabs.com/faq.json';
const ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages';

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

    let apiResp;
    try {
      apiResp = await fetch(ANTHROPIC_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': env.ANTHROPIC_API_KEY,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: MODEL,
          max_tokens: MAX_TOKENS,
          // Stable prefix first with a cache breakpoint; the per-request
          // question rides in messages so the system block stays cacheable.
          system: [
            {
              type: 'text',
              text: SYSTEM_INSTRUCTIONS + '\n\n' + faqText,
              cache_control: { type: 'ephemeral' },
            },
          ],
          messages: [{ role: 'user', content: question }],
        }),
      });
    } catch (e) {
      return reply(502, { error: 'Upstream unavailable' }, origin);
    }

    if (apiResp.status === 429 || apiResp.status === 529) {
      return reply(503, { error: 'The assistant is busy, try again in a minute' }, origin);
    }
    if (!apiResp.ok) {
      return reply(502, { error: 'The assistant could not process that question' }, origin);
    }

    const data = await apiResp.json();

    if (data.stop_reason === 'refusal') {
      return reply(200, { answer: 'That is outside what this assistant can help with. For anything about SCL certification, the framework, or the assessment process, ask away or reach out via the contact page.' }, origin);
    }

    const answer = (data.content || [])
      .filter((b) => b.type === 'text')
      .map((b) => b.text)
      .join('')
      .trim();

    if (!answer) {
      return reply(502, { error: 'Empty response from the assistant' }, origin);
    }

    return reply(200, { answer: answer }, origin);
  },
};
