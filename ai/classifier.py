import json
from ai.ollama_client import ask_json, LLMUnavailable
from ai.text_normalizer import normalize
from ai.rag_engine import retrieve_context

CHOICE_SCHEMA = {
    'type': 'object',
    'properties': {
        'kategori_id': {'type': 'string'},
        'nama_baru': {'type': 'string'},
        'alasan': {'type': 'string'}
    },
    'required': ['kategori_id', 'nama_baru', 'alasan']
}


def detect_type(text):
    s = normalize(text)
    income = ('omzet', 'penjualan', 'terima', 'pendapatan', 'masuk', 'setoran modal', 'tambahan modal')
    expense = ('beli', 'bayar', 'belanja', 'gaji', 'listrik', 'sewa', 'stok', 'biaya', 'gas', 'air', 'upah')
    a = sum(x in s for x in income)
    b = sum(x in s for x in expense)
    if a > b:
        return 'pemasukan'
    if b > a:
        return 'pengeluaran'
    try:
        reply = ask_json(
            'Tentukan jenis transaksi UMKM. Hanya jawab "pemasukan" atau "pengeluaran" dalam format JSON.',
            json.dumps({'transaksi': text}, ensure_ascii=False),
            {'type': 'object', 'properties': {'tipe': {'type': 'string', 'enum': ['pemasukan', 'pengeluaran']}}, 'required': ['tipe']}
        )
        if reply.get('tipe') in ('pemasukan', 'pengeluaran'):
            return reply['tipe']
    except Exception:
        pass
    raise ValueError('Jenis transaksi tidak dapat ditentukan otomatis. Silakan pilih jenis secara manual.')


def classify(tipe, deskripsi, cats, toko_id=None):
    relevant = [c for c in cats if c['tipe'] == tipe]
    allowed_categories = [{'id': c['id'], 'nama': c['nama'], 'deskripsi': c.get('deskripsi', '')}
                          for c in relevant if c['nama'].casefold() != 'others']
    other_cat = next((c for c in relevant if c['nama'].casefold() == 'others'), None)

    if not allowed_categories:
        return {
            'kategori_id': other_cat['id'] if other_cat else None,
            'usulan': {},
            'skor': None,
            'status_kategori': 'perlu_tinjauan',
            'message': 'Belum ada kategori aktif, dialihkan ke Others'
        }

    valid_ids = {c['id'] for c in allowed_categories}

    rag_hits = []
    if toko_id:
        try:
            rag_hits = retrieve_context(toko_id, tipe, deskripsi, top_k=5, min_score=0.35)
        except Exception:
            rag_hits = []

    best_rag_hit = rag_hits[0] if rag_hits else None

    system_prompt = (
        "Kamu adalah asisten akuntansi UMKM Indonesia. Tugasmu mengklasifikasikan deskripsi transaksi "
        "ke dalam salah satu ID kategori yang tersedia. Jika tidak ada kategori yang cocok sama sekali, "
        "kamu boleh mengusulkan nama kategori baru pada properti 'nama_baru' (dan kosongkan 'kategori_id'). "
        "Jawab hanya dalam JSON valid."
    )
    user_payload = {
        'tipe_transaksi': tipe,
        'deskripsi_transaksi': deskripsi,
        'daftar_kategori_tersedia': allowed_categories
    }
    if rag_hits:
        user_payload['bukti_historis_serupa'] = [
            {'contoh': h['text'], 'kategori': h['category_name'], 'kemiripan': h['score']}
            for h in rag_hits[:3]
        ]

    user_prompt = json.dumps(user_payload, ensure_ascii=False)

    try:
        reply = ask_json(system_prompt, user_prompt, CHOICE_SCHEMA)
        cid = reply.get('kategori_id', '')
        proposed = reply.get('nama_baru', '').strip()[:80]

        if cid in valid_ids:
            cat_name = next(c['nama'] for c in allowed_categories if c['id'] == cid)
            score = None
            if best_rag_hit and best_rag_hit.get('category_id') == cid:
                score = best_rag_hit['score']
            elif rag_hits:
                matched_hit = next((h for h in rag_hits if h.get('category_id') == cid), None)
                if matched_hit:
                    score = matched_hit['score']
            status = 'otomatis' if score is not None and score >= 0.75 else 'dikonfirmasi'
            return {
                'kategori_id': cid,
                'usulan': {},
                'skor': score,
                'status_kategori': status,
                'message': f"Llama 3 mengklasifikasikan ke kategori: {cat_name}"
            }

        if proposed:
            matched = next((c for c in relevant if c['nama'].casefold() == proposed.casefold()), None)
            if matched:
                return {
                    'kategori_id': matched['id'],
                    'usulan': {},
                    'skor': None,
                    'status_kategori': 'dikonfirmasi',
                    'message': f"Kategori '{matched['nama']}' sudah ada"
                }

            cap = {'pengeluaran': 10, 'pemasukan': 5}.get(tipe, 10)
            if len(relevant) >= cap:
                return {
                    'kategori_id': other_cat['id'] if other_cat else None,
                    'usulan': {},
                    'skor': None,
                    'status_kategori': 'perlu_tinjauan',
                    'message': f"Batas kategori {tipe} ({cap}) telah tercapai, disarankan masuk ke Others"
                }

            return {
                'kategori_id': None,
                'usulan': {'nama_baru': proposed},
                'skor': None,
                'status_kategori': 'perlu_tinjauan',
                'message': f"Llama 3 mengusulkan kategori baru: {proposed}"
            }

        return {
            'kategori_id': other_cat['id'] if other_cat else None,
            'usulan': {},
            'skor': None,
            'status_kategori': 'perlu_tinjauan',
            'message': 'Model tidak yakin, silakan tentukan kategori secara manual'
        }

    except LLMUnavailable as exc:
        # Fall back to high-confidence vector retrieval when local Ollama daemon is unreachable.
        if best_rag_hit and best_rag_hit['score'] >= 0.65 and best_rag_hit['category_id'] in valid_ids:
            return {
                'kategori_id': best_rag_hit['category_id'],
                'usulan': {},
                'skor': best_rag_hit['score'],
                'status_kategori': 'otomatis' if best_rag_hit['score'] >= 0.80 else 'perlu_tinjauan',
                'message': f"Pencarian semantik merekomendasikan: {best_rag_hit['category_name']} (skor: {best_rag_hit['score']:.2f})"
            }

        return {
            'kategori_id': other_cat['id'] if other_cat else None,
            'usulan': {},
            'skor': None,
            'status_kategori': 'perlu_tinjauan',
            'message': f"Ollama tidak tersedia ({exc}); silakan pilih kategori manual."
        }
    except Exception as exc:
        if best_rag_hit and best_rag_hit['score'] >= 0.70 and best_rag_hit['category_id'] in valid_ids:
            return {
                'kategori_id': best_rag_hit['category_id'],
                'usulan': {},
                'skor': best_rag_hit['score'],
                'status_kategori': 'perlu_tinjauan',
                'message': f"Pencarian semantik merekomendasikan: {best_rag_hit['category_name']} (skor: {best_rag_hit['score']:.2f})"
            }

        return {
            'kategori_id': other_cat['id'] if other_cat else None,
            'usulan': {},
            'skor': None,
            'status_kategori': 'perlu_tinjauan',
            'message': f"Gagal klasifikasi: {exc}"
        }
