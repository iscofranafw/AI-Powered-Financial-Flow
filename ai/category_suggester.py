import json
from ai.ollama_client import ask_json, LLMUnavailable

BASE_EXPENSE = ['Gaji', 'Operasional', 'Pemasaran', 'Others']
BASE_INCOME = ['Penjualan', 'Tambahan Modal', 'Pendapatan Lain', 'Others']
BUILTINS = {
    'grosir': (['Stok Sembako', 'Stok Makanan dan Minuman', 'Stok Kebutuhan Rumah Tangga', 'Kemasan'], ['Penjualan Eceran', 'Penjualan Partai']),
    'bakery': (['Bahan Baku', 'Kemasan'], ['Penjualan Toko', 'Pesanan', 'Penjualan Online']),
    'roti': (['Bahan Baku', 'Kemasan'], ['Penjualan Toko', 'Pesanan', 'Penjualan Online']),
    'warung makan': (['Bahan Baku', 'Kemasan', 'Transportasi'], ['Penjualan Makanan', 'Pesanan']),
    'bengkel': (['Stok Suku Cadang', 'Peralatan'], ['Jasa Servis', 'Penjualan Suku Cadang']),
}

def recommend(bidang):
    field = (bidang or '').lower()
    known = next((v for k, v in BUILTINS.items() if k in field), ([], []))
    out = {
        'pengeluaran': BASE_EXPENSE[:3] + known[0] + ['Others'],
        'pemasukan': known[1] + ['Penjualan', 'Tambahan Modal', 'Pendapatan Lain', 'Others']
    }
    try:
        data = ask_json(
            'Sarankan kategori UMKM khusus bidang usaha dalam Bahasa Indonesia. Nama pendek dan beda makna. JSON berisi pengeluaran dan pemasukan berupa list nama. Maksimal 4 per jenis.',
            json.dumps({'bidang_usaha': bidang}, ensure_ascii=False),
            {'type': 'object', 'properties': {'pengeluaran': {'type': 'array', 'items': {'type': 'string'}}, 'pemasukan': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['pengeluaran', 'pemasukan']}
        )
        for tp in out:
            for name in data.get(tp, [])[:4]:
                if isinstance(name, str) and 2 <= len(name.strip()) <= 60:
                    out[tp].insert(-1, name.strip())
    except (LLMUnavailable, ValueError, TypeError):
        pass

    for tp, cap in [('pengeluaran', 10), ('pemasukan', 5)]:
        seen = set()
        clean = []
        for n in out[tp]:
            if n.casefold() not in seen:
                seen.add(n.casefold())
                clean.append(n)
        if tp == 'pemasukan':
            mandatory = ['Tambahan Modal', 'Pendapatan Lain', 'Others']
            variable = [n for n in clean if n not in mandatory]
            clean = variable[:cap - len(mandatory)] + mandatory
        else:
            clean = [n for n in clean if n != 'Others'][:cap - 1] + ['Others']
        out[tp] = clean
    return out
