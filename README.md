# AI-Powered Financial Recording & Cash Flow Assistant

**Sistem Pencatatan, Analisis, dan Otomasi Arus Kas UMKM Berbasis AI**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Supabase](https://img.shields.io/badge/Database-Supabase%20PostgreSQL-green.svg)](https://supabase.com/)
[![Ollama](https://img.shields.io/badge/AI-Ollama%20Llama%203%208B-orange.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

## 1. Gambaran Umum

**AI-Powered Financial Recording & Cash Flow Assistant** merupakan aplikasi web berbasis Python dan Streamlit yang dikembangkan untuk membantu pelaku Usaha Mikro, Kecil, dan Menengah (UMKM) dalam mencatat, mengelola, dan menganalisis arus kas usaha.

Aplikasi ini mengintegrasikan pengelolaan transaksi keuangan, sistem persetujuan transaksi, visualisasi laporan, serta kecerdasan buatan lokal menggunakan **Llama 3 8B melalui Ollama**.

Pengembangan aplikasi berfokus pada permasalahan umum dalam pengelolaan keuangan UMKM, seperti:

- Pencatatan pemasukan dan pengeluaran yang belum konsisten.
- Kesulitan mengelompokkan transaksi berdasarkan kategori usaha.
- Pencampuran tambahan modal dengan pendapatan operasional.
- Keterbatasan informasi untuk mengevaluasi kondisi arus kas.
- Kebutuhan pengawasan transaksi yang dicatat oleh kasir.

Sistem dirancang untuk mendukung berbagai jenis usaha, termasuk bakery, grosir, bengkel, dan warung makan.

### Status Pengembangan

**Status Pengembangan: Tahap Finalisasi untuk Pengajuan Proyek Capstone Dicoding**

Repository ini mendokumentasikan pengembangan lengkap aplikasi. Seluruh fitur utama sistem telah diimplementasikan, meliputi:
- Autentikasi pengguna dan pemisahan hak akses berbasis peran (Superadmin, Pemilik Toko, Kasir).
- Pencatatan dan pengelolaan transaksi keuangan (pemasukan, pengeluaran, serta pemisahan modal dari pendapatan operasional).
- Sistem persetujuan transaksi oleh Pemilik Toko (*owner approval workflow*).
- Impor transaksi massal via CSV dengan validasi integritas, deteksi anomali, dan proteksi idempotensi penulisan atomik.
- Klasifikasi transaksi berbasis kecerdasan buatan (*AI classification*) dengan ekstraksi nominal deterministik dan normalisasi teks.
- Embedding semantik (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensi) + FAISS `IndexFlatIP` + *similarity gate* (`score >= 0.80`, `margin >= 0.04`) untuk klasifikasi cepat transaksi, diikuti Llama 3 8B terkunci pada daftar kategori toko (`allowed_categories`) tanpa *few-shot* RAG pada prompt.
- Konfirmasi kategori awal toko saat pembuatan toko / unggah CSV sebelum klasifikasi berjalan (`allowed_categories` ketat; input di luar daftar ditolak ke `perlu_tinjauan`).
- RAG Panduan Usaha terisolasi per `jenis_usaha` (`ai/knowledge_rag.py`, `data/knowledge/`, `knowledge_store/`) untuk tanya jawab operasional pemilik toko.
- Peramalan pengeluaran kas usaha multi-model adaptif (`Prophet`, `Seasonal Naive`, `MA7`, `ETS`, `ARIMA`, `LinearRegression`) dengan seleksi otomatis model MAPE terendah dan horizon 3–90 hari.
- Asisten keuangan percakapan berbasis AI (*AI financial chat assistant*) dengan floating agent dan konfirmasi perubahan data.

Proyek saat ini berada pada tahap finalisasi dan pengujian komprehensif untuk pengajuan (*submission*) capstone project Dicoding. Aplikasi ini dirancang sebagai asisten operasional arus kas UMKM dan bukan pengganti sistem akuntansi formal untuk perpajakan maupun penggajian.

---

## 2. Fitur Aplikasi

### 2.1. Pengelolaan Transaksi Keuangan

- Pencatatan pemasukan dan pengeluaran usaha.
- Pengelompokan transaksi berdasarkan kategori.
- Pengelolaan riwayat transaksi.
- Penyaringan data berdasarkan periode.
- Pemisahan tambahan modal dari pemasukan operasional.
- Pengelolaan transaksi sesuai hak akses pengguna.

### 2.2. Klasifikasi Transaksi Berbasis AI

Aplikasi mengintegrasikan Llama 3 8B melalui Ollama untuk membantu memahami deskripsi transaksi menggunakan bahasa alami.

Kemampuan yang dikembangkan meliputi:

- Normalisasi teks transaksi dan penanganan variasi penulisan.
- Ekstraksi nominal menggunakan aturan deterministik.
- Identifikasi jenis transaksi berdasarkan konteks usaha.
- Rekomendasi kategori berdasarkan katalog kategori toko.
- Penanganan kesalahan respons model dan fallback aman.

Sebagai contoh, deskripsi:

`Beli telur 3 tray 250rb`

dapat diproses untuk mengenali nominal **Rp250.000** dan menentukan kategori pengeluaran yang relevan secara otomatis.

Klasifikasi diperkuat menggunakan **SentenceTransformers dan FAISS** sebagai *similarity gate* cepat sebelum fallback ke Llama 3 8B yang terkunci ke katalog kategori awal toko.

### Penyesuaian Model terhadap Data

Model bahasa yang dipakai adalah Llama 3 8B (melalui Ollama) tanpa fine-tuning. Untuk klasifikasi transaksi: (1) jalur pertama menghitung embedding 384 dimensi + FAISS *similarity gate* (`score >= 0.80` dan margin peringkat ke-2 `>= 0.04`) memakai indeks kategori & transaksi terkonfirmasi milik toko (`vector_store/<toko_id>/`); (2) jika belum lolos *similarity gate*, Llama 3 8B dipanggil **tanpa** contoh *few-shot* RAG dan hanya memilih dari daftar kategori resmi toko yang telah ditetapkan di awal (`allowed_categories`); (3) hasil di luar daftar atau berkeyakinan `< 70` masuk status `perlu_tinjauan`. Secara terpisah, fitur tanya jawab panduan bisnis (`Panduan Usaha`) memakai RAG Markdown + FAISS terisolasi per `jenis_usaha` di `knowledge_store/<jenis_usaha>/`. Catatan: setiap dataset pengembangan sintetis di repositori dilabeli eksplisit sebagai **data sintetis**, bukan data toko asli.

### 2.3. Dashboard dan Analisis Arus Kas

Dashboard menyediakan ringkasan kondisi keuangan usaha, meliputi:

- Total pemasukan.
- Total pengeluaran.
- Arus kas bersih.
- Tren transaksi berdasarkan periode.
- Distribusi pengeluaran berdasarkan kategori.
- Visualisasi data interaktif menggunakan Plotly.

Sistem membedakan arus kas operasional dari arus kas yang memperhitungkan tambahan modal agar pengguna dapat memahami sumber perubahan saldo secara lebih tepat.

### 2.4. Narasi Laporan Keuangan Berbasis AI

Integrasi Llama 3 digunakan untuk membantu menyusun penjelasan laporan keuangan dalam bahasa yang mudah dipahami. Nilai numerik dihitung secara deterministik oleh kode aplikasi atau database, sedangkan model bahasa menyusun narasi berdasarkan hasil perhitungan tersebut guna mencegah halusinasi angka.

### 2.5. Multi-Tenant dan Role-Based Access Control

Aplikasi dirancang untuk mendukung beberapa toko dengan pemisahan akses berbasis peran:

**Superadmin**
- Meninjau dan menyetujui pendaftaran pemilik toko baru.
- Meninjau pengajuan penghapusan transaksi toko.
- Mengelola administrasi aplikasi lintas toko.

**Pemilik Toko (Owner)**
- Mengelola transaksi dan katalog kategori toko.
- Meninjau dan menyetujui (*approval*) transaksi yang dicatat kasir.
- Menerbitkan tiket pendaftaran kasir.
- Mengakses dashboard, peramalan arus kas, dan laporan keuangan.

**Kasir (Cashier)**
- Mencatat transaksi kasir harian.
- Mengirim transaksi ke antrean persetujuan pemilik toko.
- Melihat riwayat transaksi yang dicatat oleh kasir bersangkutan.

Supabase PostgreSQL digunakan sebagai penyimpanan data dan menerapkan kebijakan **Row Level Security (RLS)** untuk menjamin isolasi data antartoko.

### 2.6. Impor Transaksi CSV

Fitur impor mendukung:

- Pembacaan transaksi dari file CSV.
- Pemeriksaan dan validasi struktur data sebelum penyimpanan.
- Peninjauan kesalahan baris per baris.
- Deteksi duplikasi dan idempotensi penulisan atomik via buku besar identitas impor (`import_transaction_identities`).
- Pengelompokan transaksi massal dalam satu transaksi database terpadu.

### 2.7. Pengelolaan Kategori Usaha

Sistem menyediakan pengaturan kategori berdasarkan kebutuhan usaha. Contoh bidang usaha yang didukung melalui data referensi:

- Bakery (Toko Roti)
- Bengkel Motor
- Grosir / Toko Kelontong
- Warung Makan

---

## 3. Arsitektur Sistem

Aplikasi menggunakan pendekatan modular dengan pemisahan antara antarmuka, logika bisnis, integrasi AI, dan akses database.

```text
                   Pengguna
                      |
                      v
              Streamlit Web App
                      |
          +-----------+-----------+
          |                       |
          v                       v
    Business Services        AI Services
          |                       |
          v                       v
   Supabase PostgreSQL      Ollama / Llama 3
          |                       |
          v                       v
   Auth, Transactions       Text Processing
   Categories, Reports      Classification
   Role-Based Access        Financial Narrative
```

### Komponen Utama

| Komponen | Teknologi | Fungsi |
|---|---|---|
| Bahasa pemrograman | Python 3.11+ | Logika aplikasi backend dan frontend |
| Antarmuka | Streamlit | Dashboard dan interaksi pengguna |
| Database | Supabase PostgreSQL | Penyimpanan, RLS, dan RPC mutasi atomik |
| Autentikasi | Supabase Auth | Manajemen sesi dan identitas pengguna |
| Model bahasa | Llama 3 8B | Pemrosesan bahasa alami dan penalaran klasifikasi |
| Inferensi lokal | Ollama | Server inferensi model bahasa lokal |
| Vector Engine | FAISS & SentenceTransformers | *Similarity gate* klasifikasi (`vector_store/`) dan RAG Panduan Usaha (`knowledge_store/`) |
| Peramalan Kas | Prophet, Statsmodels (ETS/ARIMA), Scikit-Learn (LinearRegression), MA7, Seasonal Naive | Peramalan pengeluaran multi-model adaptif (horizon 3–90 hari) |
| Pengolahan data | Pandas, NumPy | Transformasi data dan agregasi keuangan |
| Visualisasi | Plotly | Grafik dan diagram interaktif laporan |
| Ekspor Laporan | OpenPyXL | Pembuatan laporan spreadsheet Excel |
| Pengujian | Pytest | Rangkaian pengujian unit dan integrasi otomatis |

---

## 4. Struktur Direktori

```text
AI-Powered-Financial-Flow/
├── .streamlit/
│   └── config.toml
├── ai/
│   ├── category_discovery.py
│   ├── category_suggester.py
│   ├── classifier.py
│   ├── copilot_knowledge.py
│   ├── copilot_planner.py
│   ├── copilot_schema.py
│   ├── copilot_tools.py
│   ├── csv_cluster.py
│   ├── embedding_pipeline.py
│   ├── financial_chat.py
│   ├── hybrid_classifier.py
│   ├── ollama_client.py
│   ├── rag_engine.py          # Embedding + FAISS similarity gate (vector_store/)
│   ├── knowledge_rag.py       # RAG Panduan Usaha terisolasi per jenis_usaha (knowledge_store/)
│   ├── store_context.py
│   └── text_normalizer.py
├── config/
│   └── settings.py
├── data/
│   └── samples/
│       ├── bakery.csv
│       ├── bengkel_motor.csv
│       ├── grosir.csv
│       └── warung_makan.csv
├── database/
│   ├── client.py
│   ├── migrations/
│   │   ├── 001_initial.sql
│   │   ├── 002_bootstrap_superadmin.sql
│   │   ├── 003_stage2_finance.sql
│   │   ├── 004_fix_cashier_provisioning.sql
│   │   ├── 005_ai_forecast.sql
│   │   ├── 006_transaction_deletion_requests.sql
│   │   ├── 007_import_idempotency.sql
│   │   ├── 008_chat_proposal_idempotency.sql
│   │   ├── 009_audit_log.sql
│   │   ├── 010_atomic_audit_mutations.sql
│   │   ├── 011_nullable_kategori_for_ai_pending.sql
│   │   └── README.md
│   └── schema.sql
├── pages/
│   ├── ai_tools.py
│   ├── cashier.py
│   ├── chat_view.py
│   ├── finance.py
│   ├── import_view.py
│   ├── login.py
│   ├── owner.py
│   ├── report_view.py
│   └── superadmin.py
├── repositories/
│   ├── deletion_repository.py
│   ├── finance_repository.py
│   └── profile_repository.py
├── scripts/
│   ├── build_dist_zip.py
│   ├── build_dist_zip.sh
│   └── measure_accuracy_and_mape.py
├── services/
│   ├── admin_service.py
│   ├── ai_service.py
│   ├── auth_service.py
│   ├── cashier_service.py
│   ├── chat_service.py
│   ├── dataset_service.py
│   ├── deletion_service.py
│   ├── finance_service.py
│   ├── forecast_service.py
│   ├── import_service.py
│   ├── report_service.py
│   ├── smart_import_engine.py
│   └── validation_service.py
├── tests/
├── utils/
│   ├── floating_agent.py
│   └── ui_components.py
├── .env.example
├── .gitignore
├── app.py
├── LICENSE
├── pytest.ini
└── requirements.txt
```

---

## 5. Instalasi dan Konfigurasi

Ikuti langkah-langkah di bawah ini secara berurutan untuk menyiapkan aplikasi dari awal hingga dapat dijalankan pada perangkat lokal.

### 5.1. Prasyarat Sistem

Sebelum memulai instalasi, pastikan perangkat lokal Anda telah memenuhi prasyarat berikut:

1. **Python**: Versi `3.11.x` (direkomendasikan Python 3.11.9).
2. **Git**: Untuk mengkloning repositori kode sumber.
3. **Ollama**: Terpasang dan aktif di latar belakang untuk inferensi model bahasa lokal Llama 3 8B.
4. **Akun Supabase**: Proyek aktif di [Supabase](https://supabase.com) (PostgreSQL 15+).
5. **Koneksi Internet**: Diperlukan untuk mengunduh dependensi Python, model bahasa, dan mengakses Supabase API.

### 5.2. Clone Repository

Buka terminal atau PowerShell, lalu jalankan perintah berikut untuk mengkloning repositori:

```bash
git clone https://github.com/iscofranafw/AI-Powered-Financial-Flow.git
cd AI-Powered-Financial-Flow
```

### 5.3. Buat dan Aktifkan Virtual Environment

Gunakan virtual environment Python yang bersih untuk menghindari konflik dependensi.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS (Bash)**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.4. Instal Dependensi Python

Perbarui `pip` ke versi terbaru, kemudian instal seluruh paket dependensi yang terdaftar di `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5.5. Penyiapan Database Supabase

Aplikasi menggunakan skema database PostgreSQL terpadu yang telah menggabungkan seluruh tabel, indeks, prosedur RPC mutasi atomik, dan kebijakan keamanan tingkat baris (RLS).

1. Masuk ke [Supabase Dashboard](https://supabase.com/dashboard) dan buat proyek baru (*New Project*). Tentukan nama proyek, sandi database, dan region terdekat.
2. Tunggu proses pembuatan proyek selesai hingga status proyek aktif.
3. Navigasi ke menu **SQL Editor** pada bilah sisi kiri dashboard Supabase.
4. Buka berkas `database/schema.sql` dari repositori ini di editor teks Anda, salin seluruh isinya, dan tempelkan (*paste*) ke dalam **SQL Editor** Supabase.
5. Klik tombol **Run** untuk mengeksekusi skema database. Skema ini dirancang untuk dijalankan **satu kali saja (*run ONCE*)** pada proyek database kosong, dan akan membuat seluruh objek database berikut secara atomik:
   - Ekstensi `pgcrypto`.
   - 9 tabel utama dalam urutan yang aman dari konflik dependensi (*foreign-key safe order*): `toko`, `profiles`, `kategori`, `transaksi`, `cashier_provisioning`, `rag_index_version`, `transaction_deletion_requests`, `import_transaction_identities`, dan `audit_log`.
   - 12 indeks performa dan integritas data (termasuk indeks unik parsial untuk idempotensi chat dan penghapusan transaksi).
   - 12 fungsi stored procedure / RPC mutasi aman bertanda `SECURITY DEFINER` (seperti `mutate_finance_audited`, `import_transactions_atomic`, dan `approve_transactions_atomic`).
   - 3 trigger database (`on_auth_user_created`, `kategori_rag_bump`, `transaksi_rag_bump`).
   - Kebijakan Row Level Security (RLS) pada seluruh tabel beserta hak akses khusus `service_role`.
   - Blok bootstrap Superadmin idempoten.

> [!NOTE]
> Jika sebelumnya Anda telah menerapkan migrasi `001` hingga `011` secara berurutan pada proyek database yang ada, Anda **TIDAK PERLU** menjalankan `database/schema.sql` karena menghasilkan status akhir (*final state*) yang sama. Direktori `database/migrations/` tetap dipertahankan dalam repositori sebagai catatan riwayat evolusi skema database.

### 5.6. Konfigurasi Environment Variables (`.env`)

Salin file template `.env.example` menjadi file `.env` di direktori utama repositori:

**Windows (PowerShell)**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS (Bash)**

```bash
cp .env.example .env
```

Buka file `.env` menggunakan editor teks, lalu isi variabel konfigurasi sesuai kredensial proyek Supabase Anda:

```ini
# ==============================================================================
# AI-POWERED FINANCIAL FLOW - ENVIRONMENT CONFIGURATION
# ==============================================================================

# Kredensial Supabase (Dapatkan dari Project Settings -> API)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy_anon_public_key
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy_service_role_key

# Konfigurasi Layanan Inferensi Ollama Lokal
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3:8b

# Model Embedding Semantik (Opsional, otomatis fallback ke MiniLM jika kosong)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**Penjelasan Variabel Lingkungan:**

| Nama Variabel | Status | Keterangan |
|---|:---:|---|
| `SUPABASE_URL` | **Wajib** | URL endpoint REST API proyek Supabase Anda (ditemukan di menu *Project Settings* &rarr; *API* &rarr; *Project URL*). |
| `SUPABASE_ANON_KEY` | **Wajib** | Kunci API publik anonim untuk otentikasi sesi klien (*Project Settings* &rarr; *API* &rarr; *Project API keys* &rarr; `anon / public`). |
| `SUPABASE_SERVICE_ROLE_KEY` | **Wajib** | Kunci API rahasia `service_role` untuk eksekusi mutasi atomik berhak khusus di lapisan backend server (*Project Settings* &rarr; *API* &rarr; *Project API keys* &rarr; `service_role`). |
| `OLLAMA_HOST` | Opsional | Alamat URL layanan Ollama lokal (nilai default: `http://127.0.0.1:11434`). |
| `OLLAMA_MODEL` | Opsional | Nama model bahasa yang dijalankan melalui Ollama (nilai default: `llama3:8b`). |
| `EMBEDDING_MODEL` | Opsional | Nama model sentence-transformers untuk embedding similarity gate dan Panduan Usaha (nilai default: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`). |

> [!CAUTION]
> Jangan pernah membagikan file `.env` atau menyertakan `SUPABASE_SERVICE_ROLE_KEY` ke repository publik. File `.env` telah tercatat di `.gitignore` dan wajib dijaga kerahasiaannya.

### 5.7. Menyiapkan Layanan AI Lokal (Ollama)

1. Pastikan Ollama telah terpasang pada komputer Anda. Jika belum, unduh melalui [ollama.com](https://ollama.com).
2. Jalankan perintah berikut di terminal untuk mengunduh model **Llama 3 8B**:

```bash
ollama pull llama3:8b
```

3. Verifikasi bahwa model telah terpasang dan layanan Ollama aktif merespons:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

Pastikan tag `llama3:8b` tercantum pada daftar model yang dikembalikan.

### 5.8. Menjalankan Aplikasi

Jalankan server aplikasi Streamlit melalui terminal pada direktori root repositori:

```bash
streamlit run app.py
```

Setelah server berhasil diinisialisasi, buka browser web Anda dan navigasikan ke alamat:

```text
http://localhost:8501
```

### 5.9. Alur Login Pertama dan Pengaturan Tiga Peran Pengguna

Sistem menerapkan kontrol akses berbasis tiga peran pengguna dengan alur penyiapan awal sebagai berikut:

#### 1. Mendaftarkan Akun Pemilik Toko (Owner)
- Buka antarmuka aplikasi di browser pada halaman **Login**.
- Pilih tab **Daftar Akun Baru**.
- Isi data lengkap: Nama Pengguna, Email, Kata Sandi, Nama Toko, dan Bidang Usaha (pilih salah satu: *Bakery*, *Grosir*, *Bengkel Motor*, atau *Warung Makan*).
- Klik tombol **Daftar**. Akun pemilik toko baru akan otomatis dibuat dengan status awal `pending` (menunggu persetujuan Superadmin sebelum dapat mengelola transaksi toko).

#### 2. Aktivasi Akun Superadmin
- Untuk mengaktifkan akun Superadmin pertama kali pada database baru:
  1. Daftarkan sebuah akun melalui form pendaftaran aplikasi dengan alamat email yang ingin Anda gunakan sebagai admin utama.
  2. Buka menu **SQL Editor** pada dashboard Supabase.
  3. Jalankan blok kode SQL berikut dengan mengganti `'REPLACE_WITH_YOUR_ADMIN_EMAIL'` menggunakan alamat email akun yang baru saja Anda daftarkan:

```sql
DO $$
DECLARE
    v_id uuid;
    v_shop uuid;
BEGIN
    SELECT id, toko_id INTO v_id, v_shop
    FROM public.profiles
    WHERE email = 'REPLACE_WITH_YOUR_ADMIN_EMAIL' AND role = 'pemilik_toko';

    IF v_id IS NOT NULL THEN
        UPDATE public.profiles
        SET role = 'superadmin', toko_id = NULL, status = 'aktif'
        WHERE id = v_id;

        IF v_shop IS NOT NULL THEN
            DELETE FROM public.toko WHERE id = v_shop;
        END IF;

        UPDATE auth.users
        SET email_confirmed_at = coalesce(email_confirmed_at, now())
        WHERE id = v_id;
    ELSE
        RAISE NOTICE 'Superadmin candidate not found — pastikan akun sudah terdaftar terlebih dahulu.';
    END IF;
END $$;
```

  4. Setelah skrip berhasil dieksekusi, kembali ke aplikasi dan login menggunakan email tersebut. Anda akan langsung masuk ke halaman navigasi **Superadmin**.
  5. Pada halaman Superadmin, Anda dapat melihat daftar pemilik toko yang berstatus `pending` dan mengklik tombol **Setujui** untuk mengaktifkan toko.

#### 3. Mendaftarkan Akun Kasir (Cashier)
- Akun kasir **tidak dapat mendaftar mandiri** secara bebas guna mencegah pencatatan transaksi liar.
- Pemilik Toko yang akunnya telah aktif membuka menu **Manajemen Kasir** di sidebar, lalu membuat tiket pendaftaran kasir (menentukan nama dan email kasir).
- Kasir mendaftar menggunakan token tiket tersebut pada formulir pendaftaran kasir, sehingga akun kasir otomatis terhubung dengan toko yang sah dengan status langsung `aktif`.

### 5.10. Pemanfaatan Data Sampel (*Sample Datasets*)

Untuk mencoba fitur analisis arus kas, peramalan pengeluaran multi-model, dan klasifikasi AI tanpa harus mencatat transaksi manual satu per satu, repositori telah menyertakan dataset **sintetis** (bukan data toko asli) di direktori `data/samples/`:

- `data/samples/bakery.csv`: Dataset transaksi toko roti dan kue (pembelian bahan tepung, telur, mentega, penjualan harian, modal awal).
- `data/samples/bengkel_motor.csv`: Dataset transaksi bengkel motor (suku cadang, oli mesin, jasa servis, peralatan montir).
- `data/samples/grosir.csv`: Dataset transaksi toko grosir dan sembako (kulakan barang dagangan, operasional logistik, penerimaan tunai).
- `data/samples/warung_makan.csv`: Dataset transaksi usaha kuliner / warung makan (belanja pasar, bumbu, gas LPG, pendapatan harian).

**Cara Mengimpor Data Sampel:**
1. Login sebagai **Pemilik Toko** yang telah berstatus aktif.
2. Pilih menu **Impor Transaksi CSV** pada bilah sisi kiri.
3. Unggah salah satu file contoh dari direktori `data/samples/`.
4. Sistem akan menampilkan pratinjau data, memverifikasi kesesuaian kolom, mendeteksi anomali tanggal, dan menjalankan pencocokan kategori.
5. Klik **Simpan Transaksi** untuk memasukkan data secara massal ke dalam database.

---

## 6. Pengujian

Proyek dilengkapi rangkaian pengujian otomatis menggunakan framework **Pytest** untuk memverifikasi logika bisnis, validasi transaksi, otorisasi peran, penanganan idempotensi, dan algoritma AI.

Jalankan seluruh rangkaian pengujian dari direktori utama:

```bash
pytest tests/ -v
```

Atau menggunakan modul Python:

```bash
python -m pytest -v
```

### Cakupan Pengujian

Modul pengujian yang tersedia di direktori `tests/` mencakup:

- `test_auth.py`: Autentikasi Supabase dan manajemen sesi pengguna.
- `test_cashier.py` & `test_cashier_and_admin.py`: Hak akses kasir, tiket pendaftaran kasir, dan persetujuan pemilik.
- `test_finance_service.py` & `test_stage2_rules.py`: Logika pencatatan transaksi, penyaringan tanggal, dan pemisahan modal.
- `test_import_security.py` & `test_smart_import.py`: Validasi struktur CSV, batas baris, sanitasi payload, dan idempotensi impor.
- `test_ai_rules.py`, `test_ai_service.py`, & `test_classification.py`: Ekstraksi nominal, normalisasi teks, dan pipeline Llama 3.
- `test_rag.py` & `test_knowledge_rag.py`: Indeks FAISS *similarity gate* klasifikasi (`vector_store/`) dan RAG Panduan Usaha terisolasi per `jenis_usaha` (`knowledge_store/`).
- `test_copilot_forecasting.py` & `test_copilot_validation.py`: Engine peramalan multi-model (Prophet, Seasonal Naive, MA7, ETS, ARIMA, LinearRegression; horizon 3–90 hari) dan validasi input narasi keuangan.
- `test_deletion_approval.py`: Alur pengajuan persetujuan penghapusan transaksi toko oleh superadmin.
- `test_audit_remediation.py`: Verifikasi jejak audit atomik dan proteksi konkurensi.
- `test_floating_agent.py` & `test_chat_assistant.py`: Interaksi agen asisten keuangan AI dan eksekusi instruksi percakapan.

### Status Pengujian

Seluruh pengujian unit dan mock dapat dieksekusi secara lokal tanpa bergantung pada koneksi internet. Untuk pengujian integrasi live yang berinteraksi langsung dengan database Supabase atau Ollama lokal, pastikan variabel `.env` dan daemon Ollama telah disiapkan sesuai instruksi pada bagian 5.

---

## 7. Roadmap Pengembangan

Pengembangan dilakukan secara bertahap menggunakan pendekatan **feature-based development**.

Setiap fitur diimplementasikan, diuji, diperbaiki jika diperlukan, kemudian didokumentasikan melalui Git commit.

| Tahap | Fokus Pengembangan | Status |
|---|---|---|
| Baseline | Fondasi aplikasi, autentikasi, transaksi, dashboard, integrasi AI awal | Selesai |
| 1 | Dataset referensi dan preprocessing | Selesai |
| 2 | SentenceTransformers dan FAISS Vector Indexing | Selesai |
| 3 | Klasifikasi embedding + FAISS similarity gate + Llama 3 terkunci | Selesai |
| 4 | Konfirmasi kategori awal CSV & isolasi bidang usaha | Selesai |
| 5 | Klasifikasi transaksi massal terkunci `allowed_categories` | Selesai |
| 6 | RAG Panduan Usaha terisolasi per `jenis_usaha` (`knowledge_store/`) | Selesai |
| 7 | AI Financial Chat Assistant (Floating Agent + Panduan Usaha) | Selesai |
| 8 | Forecasting pengeluaran multi-model adaptif (Prophet, ETS, ARIMA, dll.) | Selesai |
| 9 | Evaluasi performa model AI dan validasi numerik | Selesai |
| 10 | Pengujian integrasi akhir, konsolidasi skema, dan dokumentasi | Selesai |

### Target Evaluasi AI

Sesuai tujuan proyek capstone, performa kecerdasan buatan dievaluasi berdasarkan kriteria kuantitatif berikut:

- **Akurasi klasifikasi transaksi minimal 85%.**
- **MAPE forecasting pengeluaran kas maksimal 20%.**
- Evaluasi klasifikasi *embedding + FAISS similarity gate* + Llama 3 terkunci **tanpa** *few-shot* RAG pada prompt.
- Pengujian transaksi di luar daftar kategori (`perlu_tinjauan` / `Others`) dan larangan pembuatan kategori liar.
- Pengukuran metrik MAE, RMSE, WAPE, dan MAPE pada seleksi otomatis multi-model forecasting (horizon 3–90 hari) untuk data toko asli maupun dataset sintetis berlabel jelas.

### Fitur AI Financial Chat Assistant

Fitur percakapan asisten keuangan cerdas memungkinkan pengguna:

- Mencatat pemasukan melalui bahasa alami secara interaktif.
- Mencatat pengeluaran melalui bahasa alami secara interaktif.
- Mengirim beberapa transaksi dalam satu percakapan.
- Memeriksa dan mengonfirmasi transaksi sebelum disimpan ke database.
- Menanyakan ringkasan arus kas toko.
- Menelusuri riwayat transaksi berdasarkan periode atau kategori.
- Meminta penjelasan kondisi keuangan berdasarkan data riil toko.

Fitur chat beroperasi di bawah aturan hak akses dan lapisan validasi yang sama dengan antarmuka aplikasi utama.

---

## 8. Keamanan dan Privasi

Pengembangan aplikasi menerapkan prinsip pertahanan berlapis (*defense-in-depth*):

**Environment Variables**
Kredensial disimpan melalui konfigurasi lingkungan (`.env`) dan tidak boleh dimasukkan ke source code atau repository publik.

**Role-Based Access Control**
Akses pengguna dibatasi secara ketat berdasarkan peran Superadmin, Pemilik Toko, dan Kasir.

**Row Level Security (RLS)**
Supabase PostgreSQL menerapkan aturan RLS pada seluruh tabel untuk memastikan data antartoko terisolasi secara aman. Operasi mutasi kritis dijalankan melalui RPC atomik dengan hak akses terkelola.

**Local AI Inference**
Pemrosesan model bahasa Llama 3 dilakukan secara lokal melalui daemon Ollama pada mesin pengguna, menjaga kerahasiaan deskripsi transaksi bisnis tanpa mengirim teks percakapan ke API pihak ketiga eksternal.

**Validasi Data dan Jejak Audit**
Seluruh input transaksi divalidasi sebelum disimpan. Perubahan transaksi penting dicatat dalam buku besar audit terenkapsulasi (`audit_log`).

**Distribusi Arsip & Sanitasi Data (Packaging Instructions)**

Untuk mencegah kebocoran data sensitif saat mendistribusikan aplikasi atau mengemas arsip proyek (misalnya format `.zip`), pastikan aset berikut **wajib dikecualikan**:
1. **Index Vektor Lokal (`vector_store/`):** Seluruh file `.faiss` dan metadata `_meta.json` berisi embedding teks transaksi nyata toko. File ini diabaikan oleh `.gitignore` dan akan dibuat ulang secara otomatis saat aplikasi dijalankan di lingkungan baru.
2. **Data Transaksi Riil (`data/Daftar_Transaksi_*.csv`):** File transaksi riil pengguna tidak boleh disertakan dalam paket distribusi. Folder `data/samples/` (seperti `bakery.csv`, `bengkel_motor.csv`, dll.) tetap dipertahankan karena merupakan dataset sintetis untuk keperluan pengujian dan demonstrasi.
3. **Kredensial dan Secrets:** File konfigurasi `.env`, `.env.*`, dan `.streamlit/secrets.toml`.

Perintah pembuatan arsip distribusi yang aman:

```bash
# Menggunakan skrip packaging otomatis (direkomendasikan, cross-platform):
python scripts/build_dist_zip.py financial-flow-dist.zip

# Atau menggunakan zip (Linux / macOS / Git Bash):
zip -r financial-flow-dist.zip . -x "vector_store/*" "data/Daftar_Transaksi_*.csv" ".env*" ".git/*" "__pycache__/*" ".pytest_cache/*"

# Atau menggunakan PowerShell (Windows):
$excludePatterns = @('[\\/]vector_store', 'Daftar_Transaksi_.*\.csv$', '^\.env', '[\\/]\.git', '[\\/]__pycache__', '[\\/]\.pytest_cache')
Get-ChildItem -Path . -Recurse | Where-Object {
    $item = $_
    -not ($excludePatterns | Where-Object { $item.FullName -match $_ -or $item.Name -match $_ })
} | Compress-Archive -DestinationPath financial-flow-dist.zip
```

---

## 9. Kontribusi

Kontribusi terhadap pengembangan proyek terbuka untuk komunitas.

Alur kontribusi yang disarankan:

1. Fork repository.
2. Buat branch untuk fitur atau perbaikan (`git checkout -b feature/nama-fitur`).
3. Implementasikan perubahan kode.
4. Jalankan pengujian otomatis (`pytest tests/ -v`).
5. Buat commit dengan pesan yang jelas dan deskriptif.
6. Ajukan Pull Request disertai penjelasan perubahan.

Kontribusi diharapkan menjaga kompatibilitas dengan arsitektur proyek dan tidak menyertakan kredensial, data transaksi pribadi, atau informasi sensitif lainnya.

---

## 10. Lisensi

Proyek **AI-Powered Financial Recording & Cash Flow Assistant** didistribusikan menggunakan **MIT License**.

Kode sumber dapat digunakan, disalin, dimodifikasi, digabungkan, dipublikasikan, dan didistribusikan, termasuk untuk penggunaan komersial, sesuai ketentuan lisensi MIT.

Lihat file [LICENSE](LICENSE) untuk ketentuan lisensi lengkap.

---

## 11. Catatan Pengembangan

Repository ini merupakan bagian dari pengembangan proyek capstone **Dicoding Bootcamp — DB15-I023**.

Dokumentasi dan implementasi diperbarui mengikuti standar kurikulum Dicoding, hasil pengujian integrasi, dan evaluasi performa model AI. Proyek dikembangkan sebagai perangkat lunak open-source yang dapat dipelajari, digunakan, dan dikembangkan lebih lanjut oleh komunitas UMKM Indonesia.

---

## 12. Sumber & Sitasi

- Model embedding: `paraphrase-multilingual-MiniLM-L12-v2` (SentenceTransformers) — https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Vector database: FAISS (Meta AI Research) — https://github.com/facebookresearch/faiss
- Peramalan: Prophet (Meta) — https://github.com/meta/prophet, Statsmodels (`ETS`, `ARIMA`) — https://www.statsmodels.org, dan Scikit-Learn (`LinearRegression`) — https://scikit-learn.org
- Model bahasa: Llama 3 8B (Meta), dijalankan secara lokal melalui Ollama — https://ollama.com dan https://llama.meta.com/llama3/
- Pustaka embedding: SentenceTransformers — https://www.sbert.net
- Database: Supabase (PostgreSQL) — https://supabase.com
- Antarmuka: Streamlit — https://streamlit.io
- Dataset: data contoh di `data/samples/` (bakery, grosir, bengkel motor, warung makan) bersifat sintetis dan dibuat untuk proyek ini, karena data transaksi asli toko tidak tersedia.
