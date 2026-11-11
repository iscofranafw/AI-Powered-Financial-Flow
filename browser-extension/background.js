const SITE_ORIGIN = 'https://shazraz.my.id';
const SITE_PATH = '/financial-flow/';
const OLLAMA = 'http://127.0.0.1:11434';
const MODEL = 'llama3:8b';
const MAX_MESSAGES = 24;
const MAX_MESSAGE_CHARS = 100000;
const MAX_PAYLOAD_BYTES = 512 * 1024;
const MAX_FORMAT_BYTES = 64 * 1024;
const MAX_RESULT_CHARS = 12000;
const CHAT_TIMEOUT_MS = 240000;
const STATUS_TIMEOUT_MS = 5000;
let requestBusy = false;

function validSiteUrl(value) {
  try {
    const u = new URL(value);
    return u.protocol === 'https:' && u.origin === SITE_ORIGIN &&
      u.username === '' && u.password === '' && u.pathname.startsWith(SITE_PATH);
  } catch { return false; }
}
function reply(state, message, extra = {}) {
  return { ok: state === 'ready', state, message, ...extra };
}
async function pairedTabIds() {
  const value = await chrome.storage.session.get({ pairedTabIds: [] });
  return new Set(Array.isArray(value.pairedTabIds) ? value.pairedTabIds.filter(Number.isInteger) : []);
}
async function isPaired(tabId) {
  const tabs = await pairedTabIds();
  return tabs.has(tabId);
}
async function pairTab(tabId) {
  const tabs = await pairedTabIds();
  tabs.add(tabId);
  await chrome.storage.session.set({ pairedTabIds: [...tabs] });
}
async function unpairTab(tabId) {
  const tabs = await pairedTabIds();
  if (!tabs.delete(tabId)) return;
  await chrome.storage.session.set({ pairedTabIds: [...tabs] });
}
async function request(path, body, timeoutMs = CHAT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(OLLAMA + path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
      cache: 'no-store',
      credentials: 'omit',
      targetAddressSpace: 'loopback'
    });
  } finally { clearTimeout(timer); }
}
async function inspectOllama() {
  let response;
  try { response = await request('/api/tags', undefined, STATUS_TIMEOUT_MS); }
  catch (e) {
    const detail = e?.name === 'TypeError'
      ? 'Ollama belum terjangkau atau browser menolak CORS. Periksa Ollama dan izinkan origin extension yang tepat melalui OLLAMA_ORIGINS.'
      : 'Ollama tidak dapat dijangkau pada 127.0.0.1:11434. Pastikan berjalan dan hanya bind ke loopback.';
    return reply('ollama_unavailable', detail);
  }
  if (response.status === 403 || response.status === 401) return reply('origin_not_allowed', 'Ollama menolak origin extension. Atur OLLAMA_ORIGINS ke origin extension yang didokumentasikan, lalu restart Ollama.');
  if (!response.ok) return reply('error', `Ollama /api/tags gagal (HTTP ${response.status}).`);
  let data;
  try { data = await response.json(); } catch { return reply('error', 'Respons Ollama tags tidak valid.'); }
  const models = Array.isArray(data.models) ? data.models.map(m => String(m.name || m.model || '')).filter(Boolean).slice(0, 64) : [];
  if (!models.includes(MODEL)) return reply('model_missing', `Model ${MODEL} belum terpasang. Jalankan: ollama pull ${MODEL}`, { models });
  return reply('ready', `Terhubung; model ${MODEL} tersedia.`, { models });
}
function validMessages(messages) {
  if (!Array.isArray(messages) || messages.length < 1 || messages.length > MAX_MESSAGES) return false;
  for (const m of messages) {
    if (!m || typeof m !== 'object' || !['system', 'user', 'assistant'].includes(m.role) || typeof m.content !== 'string' || m.content.length > MAX_MESSAGE_CHARS) return false;
  }
  return true;
}
function safeFormat(format) {
  if (format === undefined || format === null) return { ok: true, value: undefined };
  if (format === 'json') return { ok: true, value: 'json' };
  if (typeof format !== 'object' || Array.isArray(format)) return { ok: false };
  try {
    const encoded = JSON.stringify(format);
    if (new TextEncoder().encode(encoded).length > MAX_FORMAT_BYTES) return { ok: false };
    const parsed = JSON.parse(encoded);
    return { ok: true, value: parsed };
  } catch { return { ok: false }; }
}
function safeOptions(options) {
  if (options === undefined) return { ok: true, value: {} };
  if (!options || typeof options !== 'object' || Array.isArray(options) || Object.keys(options).some(k => !['temperature', 'num_predict'].includes(k))) return { ok: false };
  const value = {};
  if (options.temperature !== undefined) {
    if (typeof options.temperature !== 'number' || !Number.isFinite(options.temperature) || options.temperature < 0 || options.temperature > 1) return { ok: false };
    value.temperature = options.temperature;
  }
  if (options.num_predict !== undefined) {
    if (!Number.isInteger(options.num_predict) || options.num_predict < 1 || options.num_predict > 2048) return { ok: false };
    value.num_predict = options.num_predict;
  }
  return { ok: true, value };
}
function validSender(sender) {
  return Number.isInteger(sender.tab?.id) && sender.frameId === 0 &&
    validSiteUrl(sender.url) && validSiteUrl(sender.tab?.url);
}
async function handle(message, sender) {
  const tabId = sender.tab?.id;
  if (!validSender(sender)) return reply('not_paired', 'Pesan hanya diizinkan dari tab top-frame Financial Flow yang tepat.');
  if (message?.type === 'ffai:pair') {
    await pairTab(tabId);
    return await inspectOllama();
  }
  if (!(await isPaired(tabId))) return reply('not_paired', 'Tab belum dipasangkan. Klik ikon extension lalu Hubungkan tab ini.');
  if (message?.type === 'ffai:status') return await inspectOllama();
  if (message?.type !== 'ffai:infer') return reply('error', 'Jenis pesan tidak didukung.');
  if (requestBusy) return reply('busy', 'Permintaan lokal lain sedang berjalan; coba lagi sebentar.');
  const payload = message.payload;
  if (!payload || typeof payload !== 'object' || !validMessages(payload.messages)) return reply('error', 'Format messages tidak valid atau melewati batas ukuran.');
  let payloadBytes;
  try { payloadBytes = new TextEncoder().encode(JSON.stringify(payload)).length; }
  catch { return reply('error', 'Payload permintaan tidak dapat dibaca.'); }
  if (payloadBytes > MAX_PAYLOAD_BYTES) return reply('error', 'Permintaan terlalu besar untuk bridge lokal.');
  const format = safeFormat(payload.format);
  if (!format.ok) return reply('error', 'Format keluaran model tidak valid atau terlalu besar.');
  const options = safeOptions(payload.options);
  if (!options.ok) return reply('error', 'Opsi model tidak diizinkan.');

  const ready = await inspectOllama();
  if (ready.state !== 'ready') return ready;

  requestBusy = true;
  try {
    const body = {
      model: MODEL,
      messages: payload.messages.map(m => ({ role: m.role, content: m.content })),
      stream: false,
      keep_alive: '10m',
      ...(format.value !== undefined ? { format: format.value } : {}),
      ...(Object.keys(options.value).length ? { options: options.value } : {})
    };
    const response = await request('/api/chat', body, CHAT_TIMEOUT_MS);
    if (response.status === 403 || response.status === 401) return reply('origin_not_allowed', 'Ollama menolak origin extension. Atur OLLAMA_ORIGINS dan restart Ollama.');
    if (!response.ok) return reply('error', `Ollama /api/chat gagal (HTTP ${response.status}).`);
    let data;
    try { data = await response.json(); } catch { return reply('error', 'Respons chat Ollama tidak valid.'); }
    const content = typeof data?.message?.content === 'string' ? data.message.content.slice(0, MAX_RESULT_CHARS) : '';
    if (!content) return reply('error', 'Ollama mengembalikan jawaban kosong atau format tidak valid.');
    return reply('ready', 'Jawaban lokal tersedia.', { content });
  } catch (e) {
    if (e?.name === 'TypeError') return reply('ollama_unavailable', 'Ollama belum terjangkau atau browser menolak CORS. Periksa OLLAMA_ORIGINS dan pastikan Ollama aktif.');
    if (e?.name === 'AbortError') return reply('timeout', 'Permintaan Ollama melewati batas waktu.');
    return reply('ollama_unavailable', 'Koneksi ke Ollama gagal.');
  } finally { requestBusy = false; }
}
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handle(message, sender).then(sendResponse).catch(() => sendResponse(reply('error', 'Terjadi kesalahan internal extension.')));
  return true;
});
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if ((changeInfo.url && !validSiteUrl(changeInfo.url)) || (tab.url && !validSiteUrl(tab.url))) unpairTab(tabId).catch(() => {});
});
chrome.tabs.onRemoved.addListener(tabId => { unpairTab(tabId).catch(() => {}); });
