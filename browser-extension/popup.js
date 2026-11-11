const stateEl = document.getElementById('state');
const detailEl = document.getElementById('detail');
const pairButton = document.getElementById('pair');
let activeTabId = null;

function eligible(url) {
  try {
    const u = new URL(url);
    return u.protocol === 'https:' && u.hostname === 'shazraz.my.id' && u.port === '' &&
      u.username === '' && u.password === '' && u.pathname.startsWith('/financial-flow/');
  } catch { return false; }
}

chrome.tabs.query({ active: true, lastFocusedWindow: true }).then(tabs => {
  const tab = tabs[0];
  if (!tab || !eligible(tab.url)) {
    stateEl.textContent = 'origin_not_allowed';
    detailEl.textContent = 'Tab aktif tidak berada di https://shazraz.my.id/financial-flow/.';
    return;
  }
  activeTabId = tab.id;
  stateEl.textContent = 'Tab memenuhi syarat';
  detailEl.textContent = 'Klik Pair untuk menyetujui penggunaan Local AI pada tab ini.';
  pairButton.disabled = false;
}).catch(() => {
  stateEl.textContent = 'error';
  detailEl.textContent = 'Tidak dapat memeriksa tab aktif.';
});

pairButton.addEventListener('click', async () => {
  pairButton.disabled = true;
  stateEl.textContent = 'Memasangkan…';
  try {
    const tab = await chrome.tabs.get(activeTabId);
    if (!eligible(tab.url)) throw new Error('Tab sudah meninggalkan halaman yang diizinkan.');
    const result = await chrome.tabs.sendMessage(activeTabId, { type: 'ffai:pair' });
    stateEl.textContent = result?.state || 'error';
    detailEl.textContent = result?.message || 'Tidak ada respons dari content script.';
    if (result?.models?.length) detailEl.textContent += ` Model terdeteksi: ${result.models.join(', ')}.`;
  } catch (error) {
    stateEl.textContent = 'extension_missing';
    detailEl.textContent = error?.message || 'Content script tidak tersedia; muat ulang tab Financial Flow lalu coba lagi.';
  }
  pairButton.disabled = false;
});
