(() => {
  const CHANNEL = 'financial-flow-local-ai-v1';
  const SOURCE = 'financial-flow-local-ai-component';
  const allowedTypes = new Set(['status', 'infer']);
  let paired = false;

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type !== 'ffai:pair') return false;
    chrome.runtime.sendMessage({ type: 'ffai:pair' }).then(result => {
      paired = !!result && result.state !== 'not_paired';
      sendResponse(result || { ok: false, state: 'error', message: 'Worker tidak memberi respons.' });
    }).catch(() => sendResponse({ ok: false, state: 'error', message: 'Tidak dapat menghubungi service worker.' }));
    return true;
  });

  window.addEventListener('message', async event => {
    if (event.origin !== location.origin || event.source === window || !event.source) return;
    const data = event.data;
    if (!data || data.source !== SOURCE || data.channel !== CHANNEL || !allowedTypes.has(data.type)) return;
    const requestId = data.requestId;
    if (typeof requestId !== 'string' || requestId.length < 1 || requestId.length > 128) return;
    if (!paired) {
      event.source.postMessage({ channel: CHANNEL, type: 'result', requestId, payload: { ok: false, state: 'not_paired', message: 'Tab belum dipasangkan dari popup extension.' } }, event.origin);
      return;
    }
    try {
      const result = await chrome.runtime.sendMessage({ type: data.type === 'status' ? 'ffai:status' : 'ffai:infer', payload: data.type === 'infer' ? data.payload : undefined });
      event.source.postMessage({ channel: CHANNEL, type: 'result', requestId, payload: result || { ok: false, state: 'error', message: 'Worker tidak memberi respons.' } }, event.origin);
    } catch {
      event.source.postMessage({ channel: CHANNEL, type: 'result', requestId, payload: { ok: false, state: 'extension_missing', message: 'Extension tidak dapat dihubungi.' } }, event.origin);
    }
  });
})();
