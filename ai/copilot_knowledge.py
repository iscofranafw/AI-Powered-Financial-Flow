"""Authoritative static knowledge base for application features, workflows, and business rules.

This module provides accurate, verified explanations of the AI-Powered Financial Flow
application capabilities for store owners and cashiers, derived directly from PRD.md
and the active codebase implementation.
"""

TOPIC_EXPLANATIONS = {
    'kategori': {
        'title': 'Kategori Pembukuan',
        'owner_guide': (
            "Fitur **Kategori Pembukuan** (Menu: *Kategori Pembukuan*):\n\n"
            "1. **Struktur Kategori:** Toko Anda dapat memiliki maksimal **7 kategori pemasukan** dan **7 kategori pengeluaran** (maksimal 14 kategori aktif per toko).\n"
            "2. **Jenis Kategori:**\n"
            "   - **Kustom:** Pos operasional harian yang dibuat sesuai bidang usaha toko Anda.\n"
            "   - **Sistem (Others):** Kategori sistem wajib sebagai penampung transaksi dengan tingkat keyakinan rendah atau belum terklasifikasi.\n"
            "   - **Non-Operasional (Tambahan Modal):** Khusus pemasukan setoran modal pemilik yang tidak dihitung sebagai omzet operasional toko.\n"
            "3. **Pengelolaan:**\n"
            "   - **Tambah Kategori:** Buat nama kategori dan deskripsi singkat.\n"
            "   - **Ubah Nama:** Perbarui nama atau deskripsi kategori kustom.\n"
            "   - **Hapus Kategori:** Hanya dapat dilakukan jika kategori belum pernah digunakan pada transaksi.\n"
            "   - **Gabungkan Kategori:** Pindahkan seluruh transaksi dari satu kategori ke kategori lain yang sejenis secara atomik.\n"
            "   - **Analisis Others:** Analisis pola transaksi pada kategori Others untuk mendapatkan rekomendasi kategori baru dari AI."
        ),
        'cashier_guide': (
            "Kategori pembukuan toko dikelola sepenuhnya oleh **Pemilik Toko**. "
            "Sebagai kasir, Anda memilih kategori yang tersedia saat mencatat transaksi."
        )
    },
    'persetujuan': {
        'title': 'Persetujuan Transaksi Kasir',
        'owner_guide': (
            "Fitur **Persetujuan Kasir** (Menu: *Persetujuan Kasir*):\n\n"
            "1. **Alur Kerja:** Seluruh transaksi yang dicatat oleh kasir otomatis berstatus **Menunggu Persetujuan** (*pending*).\n"
            "2. **Pemisahan Laporan:** Transaksi pending **tidak akan masuk** ke laporan keuangan resmi atau saldo kas sebelum disetujui.\n"
            "3. **Deteksi Anomali:** Sistem secara otomatis mendeteksi transaksi bernominal di atas 3x nilai tengah (*median*) transaksi wajar toko.\n"
            "4. **Tindakan Pemilik:**\n"
            "   - **Setujui Terpilih:** Menyetujui satu atau beberapa transaksi massal secara atomik.\n"
            "   - **Tolak Terpilih:** Menolak transaksi yang tidak sesuai."
        ),
        'cashier_guide': (
            "Setiap transaksi pemasukan atau pengeluaran yang Anda catat akan berstatus "
            "**Menunggu Persetujuan** dan akan ditinjau langsung oleh Pemilik Toko sebelum masuk ke laporan keuangan resmi."
        )
    },
    'impor': {
        'title': 'Impor Data Transaksi CSV',
        'owner_guide': (
            "Fitur **Impor Data CSV** (Menu: *Impor Data CSV*):\n\n"
            "1. **Unggah File:** Mendukung file CSV mutasi rekening bank atau rekap transaksi pembukuan lama.\n"
            "2. **Pratinjau & Pemetaan Kolom:** Memetakan kolom tanggal, keterangan, nominal, dan jenis transaksi (debit/kredit).\n"
            "3. **Klasifikasi Otomatis:** Sistem menjalankan klasifikasi batch cerdas menggunakan Ollama dan FAISS RAG untuk memetakan kategori secara otomatis.\n"
            "4. **Simpan Aman:** Transaksi divalidasi dan disimpan langsung dengan status disetujui untuk pemilik toko."
        ),
        'cashier_guide': (
            "Fitur Impor Data CSV adalah hak akses khusus **Pemilik Toko**."
        )
    },
    'laporan': {
        'title': 'Ringkasan Bisnis & Laporan Keuangan',
        'owner_guide': (
            "Fitur **Ringkasan Bisnis** (Menu: *Ringkasan Bisnis*):\n\n"
            "1. **Metrik Utama:** Menampilkan Total Pemasukan, Total Pengeluaran, Omzet Bersih, dan Arus Kas Operasional toko.\n"
            "2. **Penyaringan Waktu:** Filter berdasarkan rentang tanggal fleksibel (hari ini, minggu ini, bulan ini, atau kustom).\n"
            "3. **Visualisasi Data:** Grafik tren pemasukan vs pengeluaran serta diagram distribusi pengeluaran per kategori.\n"
            "4. **Ekspor Data:** Unduh laporan pembukuan ke dalam format **Excel (.xlsx)** atau **CSV (.csv)** ber-BOM UTF-8."
        ),
        'cashier_guide': (
            "Laporan rekapitulasi bisnis toko hanya dapat diakses oleh **Pemilik Toko** demi menjaga kerahasiaan data finansial."
        )
    },
    'kasir': {
        'title': 'Pengelolaan Akun Kasir',
        'owner_guide': (
            "Fitur **Akun Kasir** (Menu: *Akun Kasir*):\n\n"
            "1. **Pendaftaran Kasir:** Pemilik toko dapat mendaftarkan akun login untuk kasir atau staf toko.\n"
            "2. **Persyaratan:** Nama lengkap, alamat email aktif, dan kata sandi minimal 8 karakter.\n"
            "3. **Isolasi Toko:** Akun kasir yang dibuat terikat secara otomatis ke toko pemilik dan tidak dapat mengakses toko lain.\n"
            "4. **Kewenangan Kasir:** Kasir dapat mencatat transaksi (dengan status pending), melihat entri sendiri, dan menggunakan AI Chat."
        ),
        'cashier_guide': (
            "Akun kasir dikelola oleh Pemilik Toko. Anda dapat mencatat transaksi harian toko melalui formulir transaksi atau AI Financial Chat."
        )
    },
    'klasifikasi_ai': {
        'title': 'Klasifikasi AI & Semantic Search (FAISS RAG)',
        'owner_guide': (
            "Arsitektur **AI Transaksi & RAG**:\n\n"
            "1. **Normalisasi Teks:** Membersihkan input, mendeteksi mata uang Rupiah, dan memetakan tanggal Asia/Jakarta.\n"
            "2. **Pencarian Semantik (FAISS RAG):** Mengambil contoh transaksi dan kategori serupa dari riwayat toko Anda menggunakan embeddings *SentenceTransformers*.\n"
            "3. **Inferensi Model Lokal:** Menggunakan Llama 3 yang dijalankan secara lokal via **Ollama** dengan konteks ketat (*bounded context*).\n"
            "4. **Validasi Kategori Ketat:** AI dilarang membuat kategori baru sembarangan; hanya kategori resmi toko yang dapat dipilih.\n"
            "5. **Proteksi Others:** Jika skor keyakinan rendah (< threshold), transaksi otomatis dialokasikan ke kategori **Others** agar aman."
        ),
        'cashier_guide': (
            "AI secara otomatis membantu memprediksi kategori transaksi yang Anda catat "
            "menggunakan model lokal yang aman dan terlindungi."
        )
    },
    'diagnostik': {
        'title': 'Status Sistem & Diagnostik Infrastruktur',
        'owner_guide': (
            "Fitur **Status Sistem** (Menu: *Status Sistem*):\n\n"
            "1. **Pemeriksaan Database:** Memverifikasi koneksi aktif ke Supabase PostgreSQL.\n"
            "2. **Pemeriksaan Ollama:** Menguji ketersediaan server Ollama lokal dan model Llama 3.\n"
            "3. **Pemeriksaan FAISS Index:** Memeriksa kesiapan file indeks RAG toko untuk pemasukan dan pengeluaran.\n"
            "4. **Pipeline Embedding:** Memverifikasi model sentence-transformers lokal untuk ekstraksi vektor semantik."
        ),
        'cashier_guide': (
            "Menu Status Sistem menampilkan kesiapan server database dan AI lokal."
        )
    },
    'transaksi': {
        'title': 'Pencatatan Transaksi Keuangan',
        'owner_guide': (
            "Metode **Pencatatan Transaksi** di Aplikasi:\n\n"
            "1. **AI Financial Chat:** Catat transaksi secara santai dengan bahasa alami Indonesia (misal: *'Beli telur 5kg Rp120.000 kemarin'*).\n"
            "2. **Pencatatan Cepat:** Input teks singkat transaksi dengan pratinjau klasifikasi AI otomatis.\n"
            "3. **Menu Transaksi Manual:** Formulir lengkap dengan input tanggal, jenis transaksi, kategori, dan nominal.\n"
            "4. **Status:** Transaksi yang dicatat pemilik toko langsung disetujui (*approved*)."
        ),
        'cashier_guide': (
            "Metode **Pencatatan Transaksi Kasir**:\n\n"
            "1. **AI Financial Chat:** Ketik transaksi dengan bahasa sehari-hari, lalu konfirmasi draf transaksi.\n"
            "2. **Menu Transaksi Manual:** Pilih jenis pemasukan/pengeluaran dan isi nominal serta keterangan.\n"
            "3. **Status:** Seluruh catatan kasir berstatus **Menunggu Persetujuan** untuk ditinjau oleh Pemilik Toko."
        )
    }
}


def detect_feature_topic(text):
    s = text.lower()
    if any(k in s for k in ['kategori', 'pos biaya', 'pos anggaran', 'pos pemasukan', 'pos pengeluaran', 'pos belanja', 'others']):
        return 'kategori'
    if any(k in s for k in ['persetujuan', 'approval', 'approve', 'setujui', 'menyetujui', 'setuju', 'tolak', 'menolak', 'verifikasi kasir', 'transaksi pending']):
        return 'persetujuan'
    if any(k in s for k in ['impor', 'import', 'csv', 'upload', 'unggah']):
        return 'impor'
    if any(k in s for k in ['laporan', 'ringkasan bisnis', 'ekspor', 'excel', 'unduh data', 'download']):
        return 'laporan'
    if any(k in s for k in ['akun kasir', 'tambah kasir', 'buat kasir', 'daftar kasir', 'staf kasir']):
        return 'kasir'
    if any(k in s for k in ['klasifikasi', 'rag', 'faiss', 'ollama', 'llama', 'model ai', 'embedding']):
        return 'klasifikasi_ai'
    if any(k in s for k in ['status sistem', 'diagnostik', 'koneksi database', 'koneksi']):
        return 'diagnostik'
    if any(k in s for k in ['transaksi', 'catat transaksi', 'pencatatan cepat', 'input transaksi']):
        return 'transaksi'
    return 'umum'


def get_feature_explanation(topic, is_owner=True):
    if topic in TOPIC_EXPLANATIONS:
        entry = TOPIC_EXPLANATIONS[topic]
        guide = entry['owner_guide'] if is_owner else entry['cashier_guide']
        return f"### {entry['title']}\n\n{guide}"

    # General overview across main features
    return (
        "### Panduan Fitur Aplikasi AI-Powered Financial Flow\n\n"
        "Aplikasi ini menyediakan fitur pembukuan cerdas yang terintegrasi untuk UMKM:\n\n"
        "1. **Pencatatan Percakapan (AI Financial Chat):** Catat pemasukan dan pengeluaran secara alami dengan konfirmasi instan.\n"
        "2. **Pencatatan Cepat & AI:** Klasifikasi kategori otomatis menggunakan model lokal Llama 3 dan semantic search FAISS RAG.\n"
        "3. **Persetujuan Kasir:** Tinjauan dan verifikasi massal untuk transaksi yang dicatat staf kasir toko.\n"
        "4. **Kategori Pembukuan:** Kelola hingga 7 kategori pemasukan dan 7 kategori pengeluaran dengan perlindungan kategori sistem `Others`.\n"
        "5. **Ringkasan Bisnis & Laporan:** Pantau omzet, arus kas operasional, dan grafik keuangan serta ekspor ke Excel/CSV.\n"
        "6. **Impor CSV:** Masukkan rekap transaksi perbankan atau pembukuan lampau dengan klasifikasi otomatis.\n"
        "7. **Pengelolaan Kasir:** Pendaftaran akun staf kasir yang terisolasi aman per toko.\n\n"
        "Ketik pertanyaan lebih spesifik (misal: *'Bagaimana cara kerja persetujuan kasir?'* atau *'Bagaimana cara menambah kategori?'*) untuk panduan mendalam."
    )


def get_forecast_explanation(is_owner=True):
    if not is_owner:
        return (
            "Fitur peramalan dan proyeksi pengeluaran (*forecasting*) dirancang khusus "
            "untuk dashboard **Pemilik Toko**."
        )

    return (
        "### Forecasting Pengeluaran Toko (Prophet)\n\n"
        "Fitur peramalan pengeluaran dirancang untuk membantu pemilik UMKM mengantisipasi beban kas di masa depan:\n\n"
        "1. **Model Peramalan:** Menggunakan **Meta Prophet** untuk analisis deret waktu (*time-series analysis*) pada beban pengeluaran toko.\n"
        "2. **Fokus Beban Berulang:** Dirancang untuk mendeteksi dan memproyeksikan pengeluaran rutin operasional (seperti tagihan listrik, sewa tempat usaha, gaji staf, dan pasokan bahan pokok bulanan).\n"
        "3. **Prasyarat Riwayat Data:** Model membutuhkan riwayat transaksi historis yang berkesinambungan agar dapat dilatih dan diuji (*backtesting*) secara valid.\n"
        "4. **Standar Kinerja:** Target akurasi model adalah nilai **MAPE (Mean Absolute Percentage Error) di bawah 20%**.\n"
        "5. **Tampilan Dashboard:** Proyeksi akan disajikan pada dashboard Pemilik Toko mulai **3 hari sebelum perkiraan tanggal pengeluaran**.\n\n"
        "*Catatan Sistem:* Pembukuan Anda saat ini terus menghimpun riwayat transaksi valid agar estimasi peramalan deret waktu dapat dihitung secara akurat tanpa manipulasi angka."
    )
