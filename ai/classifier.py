import json
from ai.ollama_client import ask_json, LLMUnavailable
from ai.text_normalizer import normalize

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

def classify(tipe, deskripsi, cats):
    relevant = [c for c in cats if c['tipe'] == tipe]
    allowed_categories = [{'id': c['id'], 'nama': c['nama'], 'deskripsi': c.get('deskripsi', '')}
                          for c in relevant if c['nama'].casefold() != 'others']
    other_cat = next((c for c in relevant if c['nama'].casefold() == 'others'), None)

    if not allowed_categories:
        return {
            'kategori_id': other_cat['id'] if other_cat else None,
            'usulan': {},
            'message': 'Belum ada kategori aktif, dialihkan ke Others'
        }

    system_prompt = (
        "Kamu adalah asisten akuntansi UMKM Indonesia. Tugasmu mengklasifikasikan deskripsi transaksi "
        "ke dalam salah satu ID kategori yang tersedia. Jika tidak ada kategori yang cocok sama sekali, "
        "kamu boleh mengusulkan nama kategori baru pada properti 'nama_baru' (dan kosongkan 'kategori_id'). "
        "Jawab hanya dalam JSON valid."
    )
    user_prompt = json.dumps({
        'tipe_transaksi': tipe,
        'deskripsi_transaksi': deskripsi,
        'daftar_kategori_tersedia': allowed_categories
    }, ensure_ascii=False)

    try:
        reply = ask_json(system_prompt, user_prompt, CHOICE_SCHEMA)
        cid = reply.get('kategori_id', '')
        proposed = reply.get('nama_baru', '').strip()[:80]
        valid_ids = {c['id'] for c in allowed_categories}

        if cid in valid_ids:
            cat_name = next(c['nama'] for c in allowed_categories if c['id'] == cid)
            return {
                'kategori_id': cid,
                'usulan': {},
                'message': f"Llama 3 mengklasifikasikan ke kategori: {cat_name}"
            }

        if proposed:
            matched = next((c for c in relevant if c['nama'].casefold() == proposed.casefold()), None)
            if matched:
                return {
                    'kategori_id': matched['id'],
                    'usulan': {},
                    'message': f"Kategori '{matched['nama']}' sudah ada"
                }

            cap = {'pengeluaran': 10, 'pemasukan': 5}.get(tipe, 10)
            if len(relevant) >= cap:
                return {
                    'kategori_id': other_cat['id'] if other_cat else None,
                    'usulan': {},
                    'message': f"Batas kategori {tipe} ({cap}) telah tercapai, disarankan masuk ke Others"
                }

            return {
                'kategori_id': None,
                'usulan': {'nama_baru': proposed},
                'message': f"Llama 3 mengusulkan kategori baru: {proposed}"
            }

        return {
            'kategori_id': other_cat['id'] if other_cat else None,
            'usulan': {},
            'message': 'Model tidak yakin, silakan tentukan kategori secara manual'
        }

    except LLMUnavailable as exc:
        return {
            'kategori_id': other_cat['id'] if other_cat else None,
            'usulan': {},
            'message': f"Ollama tidak tersedia ({exc}); silakan pilih kategori manual."
        }
    except Exception as exc:
        return {
            'kategori_id': other_cat['id'] if other_cat else None,
            'usulan': {},
            'message': f"Gagal klasifikasi: {exc}"
        }
