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
const FRAMEWORK_URL = 'https://safetycriticallabs.com/framework.json';
const UPSTREAM_TIMEOUT_MS = 90000; // cold start: model load + prompt eval can near a minute

// Framework excerpts appended per question, capped so the whole prompt stays
// inside num_ctx 8192: ~350 instructions + ~3.4k FAQ + <=2.5k excerpts + question.
// Scoring must stay in lockstep with scoring_mirror.py (the offline test bench).
const EXCERPT_BUDGET_CHARS = 8000;
const EXCERPT_MAX_PICK = 5;
const EXCERPT_ABS_MIN = 4.5;
const EXCERPT_REL_MIN = 0.35;
const MAX_QUERY_TOKENS = 20; // CPU guard: a 500-char question can hold ~100 tokens,
                             // and scan cost scales linearly with token count

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

Answer using ONLY the reference entries provided below. The entries are SCL's FAQ, sometimes followed by verbatim excerpts from the AI Requirements Framework v3.3 standard. Rules:
- Keep answers to 2 to 6 short sentences, plain text, no markdown formatting, no em dashes.
- When you answer from framework excerpts, cite the requirement IDs you used, for example (AI-4.1). Never cite an ID that is not present in the provided excerpts, and never invent requirement text.
- If the reference entries do not cover the question, say so plainly and point the visitor to the contact page at /contact.html. Never guess or invent facts, certifications, clients, partnerships, or status.
- Do not overstate SCL's status. SCL is pre-accreditation: ANAB intake is on file and a fee estimate was received, but formal engagement is deferred until certification volume supports it.
- If asked something unrelated to SCL, AI assurance, or safety-critical certification, politely decline and redirect to what you can help with.

Reference entries follow.`;

/* ── Framework retrieval: score chunks against the question, keep the best
   few under a hard character budget. Company questions score below the
   threshold and get no excerpts, keeping those prompts short and fast. ── */

const STOPWORDS = new Set(['the','a','an','is','are','was','were','be','been','of','for','to','on','in','by','with','at','as','from','and','or','but','if','then','so','also','do','does','did','can','could','will','would','should','may','might','i','my','me','you','your','we','our','us','they','them','it','its','this','that','these','those','there','here','not','please','just','like','some','any','what','how','where','when','why','who','which','about','tell','need','want','have','has','framework','scl','much','safety','critical','labs']);

function tokenize(s) {
  return s.toLowerCase().replace(/[^a-z0-9.\- ]+/g, ' ').split(/\s+/)
    .filter(function (w) { return w.length >= 3 && !STOPWORDS.has(w); });
}

function stemToken(t) {
  // light suffix stemming so "relying" reaches "reliance", "applies" reaches
  // "applicability"; matches scoring_mirror.py exactly
  if (t.length > 5) t = t.replace(/(ings|ing|ed|es|ly|s)$/, '');
  else t = t.replace(/s$/, '');
  if (t.length > 4 && t.charAt(t.length - 1) === 'y') t = t.slice(0, -1);
  return t;
}

function selectExcerpts(question, framework) {
  var entries = (framework && framework.entries) || [];
  if (!entries.length) return '';
  var qTokens = tokenize(question).slice(0, MAX_QUERY_TOKENS);
  var qNorm = question.toLowerCase();
  var stems = qTokens.map(stemToken);

  // precompute searchable fields
  var fields = entries.map(function (c) {
    return { title: c.title.toLowerCase(), kw: (c.keywords || []).join(' ').toLowerCase(), text: c.text.toLowerCase() };
  });

  // Single pass over the corpus per stem: document frequency for the rarity
  // weight AND the per-entry hit set, so the scoring loop never rescans texts
  // (the scans are the dominant CPU cost at 270KB of corpus).
  var weights = [];
  var textHits = [];
  for (var si = 0; si < stems.length; si++) {
    var hits = new Set();
    for (var fi = 0; fi < fields.length; fi++) {
      if (fields[fi].text.indexOf(stems[si]) !== -1) hits.add(fi);
    }
    weights.push(hits.size > 0 ? 1 / (1 + Math.log(hits.size)) : 0.6);
    textHits.push(hits);
  }

  var scored = [];
  var top = 0;
  for (var i = 0; i < entries.length; i++) {
    var c = entries[i], f = fields[i];
    var score = 0;
    // direct requirement-id mention ("ai-4.1", "ai-12") is a strong signal;
    // reject prefix collisions ("ai-1" inside "ai-12") by refusing matches
    // followed by a digit (a following "." is the parent-area case, kept).
    var idLower = c.id.toLowerCase();
    var p = qNorm.indexOf(idLower);
    while (p !== -1 && /\d/.test(qNorm.charAt(p + idLower.length))) p = qNorm.indexOf(idLower, p + 1);
    if (p !== -1) score += 100;
    for (var j = 0; j < stems.length; j++) {
      // pair bonus first: it uses the raw tokens (always >=3 chars), so a
      // short stem must not suppress it
      if (j + 1 < qTokens.length) {
        var pair = qTokens[j] + ' ' + qTokens[j + 1];
        if (f.title.indexOf(pair) !== -1 || f.kw.indexOf(pair) !== -1) score += 12;
      }
      var st = stems[j];
      if (st.length < 3) continue;
      var w = weights[j];
      if (f.title.indexOf(st) !== -1) score += 10 * w;
      if (f.kw.indexOf(st) !== -1) score += 8 * w;
      if (textHits[j].has(i)) score += 2 * w;
    }
    // whole-phrase alias: multi-word keyword appearing verbatim in the question
    var kws = c.keywords || [];
    for (var k = 0; k < kws.length; k++) {
      if (kws[k].length >= 8 && kws[k].indexOf(' ') !== -1 && qNorm.indexOf(kws[k]) !== -1) score += 15;
    }
    if (score > top) top = score;
    scored.push({ s: score, c: c });
  }

  var floor = Math.max(EXCERPT_ABS_MIN, EXCERPT_REL_MIN * top);
  var keep = scored.filter(function (x) { return x.s >= floor; });
  if (!keep.length) return '';
  keep.sort(function (a, b) { return b.s - a.s; });
  var used = 0;
  var picked = [];
  for (var m = 0; m < keep.length && picked.length < EXCERPT_MAX_PICK; m++) {
    var cc = keep[m].c;
    if (used + cc.text.length > EXCERPT_BUDGET_CHARS) continue;
    used += cc.text.length;
    picked.push(cc);
  }
  if (!picked.length) return '';
  var parts = ['\n\n--- Verbatim excerpts from the AI Requirements Framework v' + (framework.version || '3.3') + ' (cite these IDs) ---'];
  for (var n = 0; n < picked.length; n++) {
    parts.push('\n[' + picked[n].id + '] ' + picked[n].title + '\n' + picked[n].text);
  }
  return parts.join('\n');
}

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

    // Framework corpus is best-effort: retrieval failure degrades to FAQ-only.
    let excerpts = '';
    try {
      const fwResp = await fetch(FRAMEWORK_URL, { cf: { cacheTtl: 300, cacheEverything: true } });
      if (fwResp.ok) excerpts = selectExcerpts(question, await fwResp.json());
    } catch (e) {
      excerpts = '';
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
          // num_ctx must clear instructions + FAQ + excerpts; Ollama's default
          // window would silently truncate the grounding. Excerpts go LAST in
          // the system block so the stable instructions+FAQ prefix stays
          // reusable in Ollama's KV prefix cache across questions.
          options: { temperature: 0.2, num_ctx: 8192, num_predict: 300 },
          messages: [
            { role: 'system', content: SYSTEM_INSTRUCTIONS + '\n\n' + faqText + excerpts },
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
