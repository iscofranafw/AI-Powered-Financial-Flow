# Financial Flow Local AI

## Setup untuk pengguna

Bridge ini memakai Ollama dan Llama 3 8B pada komputer yang membuka Financial Flow. Perlu Chrome atau Microsoft Edge desktop; extension tidak berjalan di browser ponsel.

1. Pasang Ollama dari [ollama.com/download](https://ollama.com/download), lalu jalankan di Terminal atau PowerShell:

   ```sh
   ollama pull llama3:8b
   ```

2. Atur environment Ollama dengan origin extension exact ini. Jangan gunakan wildcard:

   ```text
   OLLAMA_ORIGINS=chrome-extension://jomfepbhiphidnjdmjodcadeehjnjhcl
   ```

   Tutup lalu jalankan kembali Ollama setelah mengubah environment. Langkah penetapan environment berbeda menurut OS; lihat [panduan konfigurasi Ollama](https://docs.ollama.com/faq). Pastikan Ollama hanya mendengarkan di loopback komputer ini (`127.0.0.1:11434`); jangan gunakan `0.0.0.0` atau membuka port 11434 ke jaringan.

3. Unduh ZIP repositori `main` dari [GitHub](https://github.com/iscofranafw/AI-Powered-Financial-Flow/archive/refs/heads/main.zip) dan ekstrak.
4. Buka `chrome://extensions` (Chrome) atau `edge://extensions` (Edge), aktifkan **Developer mode**, pilih **Load unpacked**, lalu pilih folder `browser-extension` dari ZIP yang diekstrak. Pastikan ID extension yang tampil `jomfepbhiphidnjdmjodcadeehjnjhcl`.
5. Buka `https://shazraz.my.id/financial-flow/` dan masuk melalui Cloudflare Access. Klik ikon **Financial Flow Local AI**, lalu tombol **Pair** untuk memasangkan tab aktif.
6. Periksa panel **AI lokal · Llama 3 8B** pada sidebar. Status siap muncul bila extension, Ollama, dan model terdeteksi.

Jika browser menampilkan prompt Local Network Access, izinkan akses lokal untuk extension ini saja. Kebijakan local-network dapat berbeda menurut versi browser; lihat [panduan Chrome](https://developer.chrome.com/blog/local-network-access) dan [panduan Edge](https://learn.microsoft.com/en-us/deployedge/ms-edge-local-network-access). Jangan menonaktifkan keamanan browser secara global.

### Jika belum siap

- **Extension belum terdeteksi:** muat ulang tab staging setelah memasang extension.
- **Tab belum dipasangkan:** klik ikon extension dan pilih **Pair**.
- **Ollama belum terjangkau:** jalankan Ollama, lalu buka `http://127.0.0.1:11434/api/tags` pada komputer yang sama.
- **Model belum ada:** jalankan `ollama pull llama3:8b`.
- **Origin ditolak:** pastikan `OLLAMA_ORIGINS` berisi origin extension exact di atas, lalu restart Ollama.
- **Masih gagal:** jangan membuka port Ollama ke internet. Periksa izin Local Network Access browser dan konfigurasi Ollama.

## Alur data dan batas server

Prompt dari Financial Flow diteruskan lewat tab staging yang sudah dipasangkan ke extension. Inferensi dilakukan oleh Ollama di komputer pengguna, kemudian jawabannya kembali ke aplikasi untuk ditampilkan atau dipakai fitur yang diminta. Aplikasi staging tetap menjadi perantara prompt dan jawaban, tetapi tidak mengirimnya ke layanan model cloud; relay hanya menyimpan antrean sementara di memori dan tidak menulis prompt/hasil ke log. VPS staging tidak menjalankan Llama dan Ollama-nya tidak dibuka ke internet.

## Batas izin dan cara kerja extension

- Content script hanya cocok pada `https://shazraz.my.id/financial-flow/*`; tidak aktif pada homepage, `/kantor-ai`, atau situs lain.
- Host permission hanya `http://127.0.0.1:11434/*`. Service worker hanya menghubungi Ollama `/api/tags` dan `/api/chat` pada komputer pengguna, dengan model tetap `llama3:8b`.
- Pairing memerlukan tindakan eksplisit pada tab aktif, berlaku hanya pada sesi browser, dan dihapus saat tab ditutup atau keluar dari path staging.
- Extension tidak membaca atau mengirim cookie browser ke Ollama, tidak mengakses URL arbitrer, dan tidak menghubungi layanan pihak ketiga.

## Protokol extension (untuk pengembang)

The Streamlit component iframe posts to its same-origin top page using:

```js
window.parent.postMessage({
  source: 'financial-flow-local-ai-component',
  channel: 'financial-flow-local-ai-v1',
  type: 'status', // or 'infer'
  requestId: 'unique-id',
  payload: { messages: [{ role: 'user', content: '...' }] } // infer only
}, window.location.origin);
```

The content script accepts only same-origin messages with the expected source/channel and relays them to the extension. Responses have the shape `{ channel, type: 'result', requestId, payload: { ok, state, message, ... } }` and are posted to the originating frame at its origin. Only the allowed page origin/path and top frame can pair or submit work. The worker ignores model/URL values from the page, validates messages/options/format, and serializes inference requests with a four-minute timeout.
