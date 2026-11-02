import json
from ai.text_normalizer import normalize
from ai.ollama_client import ask_json, LLMUnavailable

def cluster_rows(rows, level='Sedang'):
    result = {}
    target_count = {'Ringkas': 3, 'Sedang': 5, 'Rinci': 7}.get(level, 4)

    for tp in ('pengeluaran', 'pemasukan'):
        subset = [(i, r) for i, r in enumerate(rows) if r.get('tipe') == tp]
        if not subset:
            continue

        unique_descriptions = list(dict.fromkeys([r['deskripsi'].strip() for _, r in subset]))
        sample = unique_descriptions[:30]

        max_k = 9 if tp == 'pengeluaran' else 4
        k = min(max_k, max(2, min(target_count, len(sample))))

        system_prompt = (
            f"Kelompokkan contoh transaksi {tp} UMKM berikut menjadi maksimal {k} nama kategori singkat (2-3 kata). "
            "Format balasan JSON: { 'kategori': ['Nama Kat 1', 'Nama Kat 2', ...] }"
        )
        user_prompt = json.dumps({'tipe': tp, 'contoh_transaksi': sample}, ensure_ascii=False)
        schema = {
            'type': 'object',
            'properties': {'kategori': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['kategori']
        }

        suggested_names = []
        try:
            reply = ask_json(system_prompt, user_prompt, schema)
            suggested_names = [s.strip() for s in reply.get('kategori', []) if s.strip()]
        except Exception:
            # Safe fallback: if LLM naming fails, fall back to default template categories below
            pass

        if not suggested_names:
            suggested_names = ['Operasional', 'Bahan Baku', 'Gaji'] if tp == 'pengeluaran' else ['Penjualan', 'Pendapatan Lain']

        for idx, (orig_idx, row) in enumerate(subset):
            assigned_group = idx % len(suggested_names)
            cat_name = suggested_names[assigned_group]
            result[orig_idx] = {
                'group': f'{tp}-{assigned_group}',
                'suggested_name': cat_name
            }

    return result
