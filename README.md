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

**Development Status: Work in Progress — Initial 50% Milestone**

Repository ini mendokumentasikan pengembangan bertahap proyek capstone. Persentase 50% merupakan estimasi progres implementasi, bukan hasil pengukuran otomatis terhadap seluruh persyaratan sistem.

Versi awal mencakup fondasi autentikasi, pengelolaan pengguna dan toko, pencatatan transaksi, pengaturan kategori, dashboard keuangan, impor data, serta integrasi awal Llama 3.

Pengembangan selanjutnya berfokus pada Retrieval-Augmented Generation (RAG), evaluasi klasifikasi transaksi, peramalan pengeluaran menggunakan Prophet, dan antarmuka percakapan AI untuk membantu pencatatan serta analisis keuangan.

Proyek ini bukan pengganti sistem akuntansi penuh dan tidak ditujukan untuk mengelola penggajian, perpajakan, atau penyusunan laporan akuntansi formal.

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

Kemampuan yang dikembangkan pada versi awal meliputi:

- Normalisasi teks transaksi.
- Ekstraksi nominal menggunakan aturan deterministik.
- Identifikasi jenis transaksi berdasarkan konteks.
- Rekomendasi kategori berdasarkan kategori yang tersedia.
- Penanganan kesalahan respons model.

Sebagai contoh, deskripsi:

`Beli telur 3 tray 250rb`

dapat diproses untuk mengenali nominal **Rp250.000** dan menentukan kategori pengeluaran yang relevan.

Pada tahap berikutnya, klasifikasi akan diperkuat menggunakan **SentenceTransformers dan FAISS** untuk mengambil contoh transaksi dari dataset referensi yang telah divalidasi.

Kategori hasil klasifikasi akan dibatasi pada kategori referensi yang diizinkan. Transaksi yang tidak sesuai dengan referensi akan diarahkan ke kategori `Others`.

### 2.3. Dashboard dan Analisis Arus Kas

Dashboard menyediakan ringkasan kondisi keuangan usaha, meliputi:

- Total pemasukan.
- Total pengeluaran.
- Arus kas bersih.
- Tren transaksi berdasarkan periode.
- Distribusi pengeluaran berdasarkan kategori.
- Visualisasi data menggunakan Plotly.

Sistem membedakan arus kas operasional dari arus kas yang memperhitungkan tambahan modal agar pengguna dapat memahami sumber perubahan saldo secara lebih tepat.

Arus kas bersih tidak selalu sama dengan laba akuntansi karena perhitungan laba dapat memerlukan informasi tambahan seperti harga pokok penjualan, persediaan, dan kewajiban usaha.

### 2.4. Narasi Laporan Keuangan Berbasis AI

Integrasi Llama 3 digunakan untuk membantu menyusun penjelasan laporan keuangan dalam bahasa yang mudah dipahami.

Nilai numerik harus dihitung oleh kode aplikasi atau database, sedangkan model bahasa digunakan untuk menyusun narasi berdasarkan hasil perhitungan tersebut.

Pendekatan ini bertujuan mengurangi risiko kesalahan angka yang dihasilkan oleh model bahasa.

### 2.5. Multi-Tenant dan Role-Based Access Control

Aplikasi dirancang untuk mendukung beberapa toko dengan pemisahan akses berdasarkan peran pengguna.

**Superadmin**

- Meninjau dan menyetujui pendaftaran pemilik toko.
- Mengelola administrasi aplikasi sesuai kewenangannya.

**Pemilik Toko (Owner)**

- Mengelola transaksi dan kategori.
- Meninjau transaksi yang dicatat kasir.
- Mengakses dashboard dan laporan keuangan.
- Mengelola operasional keuangan toko.

**Kasir (Cashier)**

- Mencatat transaksi sesuai kewenangan.
- Mengirim transaksi untuk proses persetujuan pemilik toko.
- Mengakses fungsi aplikasi yang diizinkan untuk peran kasir.

Supabase PostgreSQL digunakan sebagai penyimpanan data dan mendukung kebijakan **Row Level Security (RLS)** untuk membantu membatasi akses antar toko.

Keamanan isolasi data tetap memerlukan pengujian integrasi terhadap kebijakan dan konfigurasi database yang digunakan.

### 2.6. Impor Transaksi CSV

Fitur impor mendukung:

- Pembacaan transaksi dari file CSV.
- Pemeriksaan dan validasi data sebelum penyimpanan.
- Peninjauan kesalahan pada setiap baris.
- Pengelolaan impor transaksi secara massal.
- Integrasi dengan fungsi database untuk mendukung konsistensi penyimpanan.

Batas jumlah baris dan perilaku transaksi atomik mengikuti implementasi serta konfigurasi database yang digunakan.

### 2.7. Pengelolaan Kategori Usaha

Sistem menyediakan pengaturan kategori berdasarkan kebutuhan usaha.

Contoh bidang usaha yang didukung melalui data referensi awal:

- Bakery
- Grosir
- Bengkel motor
- Warung makan

Pengguna dapat menyesuaikan struktur kategori sesuai kebutuhan operasional toko.

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
| Bahasa pemrograman | Python 3.11+ | Logika aplikasi |
| Antarmuka | Streamlit | Dashboard dan interaksi pengguna |
| Database | Supabase PostgreSQL | Penyimpanan dan pengelolaan data |
| Autentikasi | Supabase Auth | Identitas dan akses pengguna |
| Model bahasa | Llama 3 8B | Pemrosesan bahasa alami |
| Inferensi lokal | Ollama | Menjalankan model bahasa |
| Pengolahan data | Pandas, NumPy | Transformasi dan analisis data |
| Visualisasi | Plotly | Grafik laporan keuangan |
| Ekspor Excel | OpenPyXL | Pembuatan laporan spreadsheet |
| Pengujian | Pytest | Pengujian otomatis |

### Pengembangan Arsitektur AI

Komponen berikut merupakan bagian dari roadmap dan tidak seluruhnya tersedia pada checkpoint awal:

- **SentenceTransformers:** menghasilkan embedding deskripsi transaksi.
- **FAISS:** melakukan pencarian kemiripan terhadap referensi transaksi.
- **Retrieval-Augmented Generation:** menyediakan konteks tambahan untuk klasifikasi dan tanya jawab.
- **Prophet:** melakukan forecasting pengeluaran.
- **AI Financial Chat Assistant:** menyediakan interaksi percakapan untuk pencatatan dan analisis keuangan.

---

## 4. Struktur Direktori

```text
AI-Powered-Financial-Flow/
├── .streamlit/
│   └── config.toml
├── ai/
│   ├── category_suggester.py
│   ├── classifier.py
│   ├── csv_cluster.py
│   ├── ollama_client.py
│   └── text_normalizer.py
├── config/
│   └── settings.py
├── data/
│   └── samples/
├── database/
│   ├── client.py
│   └── migrations/
├── pages/
│   ├── ai_tools.py
│   ├── cashier.py
│   ├── finance.py
│   ├── import_view.py
│   ├── login.py
│   ├── owner.py
│   ├── report_view.py
│   └── superadmin.py
├── repositories/
│   ├── finance_repository.py
│   └── profile_repository.py
├── services/
│   ├── admin_service.py
│   ├── ai_service.py
│   ├── auth_service.py
│   ├── cashier_service.py
│   ├── finance_service.py
│   ├── import_service.py
│   └── report_service.py
├── tests/
├── utils/
│   └── ui_components.py
├── .env.example
├── .gitignore
├── app.py
├── pytest.ini
├── README.md
├── LICENSE
└── requirements.txt
```

Struktur tersebut menggambarkan organisasi proyek pada checkpoint awal. Direktori dan modul tambahan akan diperkenalkan secara bertahap sesuai pengembangan fitur.

---

## 5. Instalasi dan Konfigurasi

### 5.1. Prasyarat

Pastikan perangkat telah memiliki:

1. Python 3.11 atau versi yang kompatibel dengan dependensi proyek.
2. Git.
3. Ollama.
4. Akun dan proyek Supabase.
5. Koneksi internet untuk instalasi dependensi dan akses Supabase.

Model bahasa dijalankan secara lokal melalui Ollama, sedangkan database Supabase memerlukan koneksi ke layanan yang dikonfigurasi.

### 5.2. Clone Repository

```bash
git clone https://github.com/USERNAME/AI-Powered-Financial-Flow.git
cd AI-Powered-Financial-Flow
```

Ganti `USERNAME` dengan pemilik repository GitHub setelah repository dibuat.

### 5.3. Buat Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.4. Instal Dependensi

```bash
python -m pip install -r requirements.txt
```

### 5.5. Konfigurasi Environment

Salin file `.env.example` menjadi `.env`.

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS**

```bash
cp .env.example .env
```

Lengkapi variabel yang diperlukan sesuai `config/settings.py` dan template `.env.example`.

Contoh konfigurasi:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3:8b
```

Nama variabel harus disesuaikan dengan konfigurasi yang benar-benar dibaca oleh aplikasi.

**Penting:**

- Jangan membagikan `.env`.
- Jangan memasukkan kredensial asli ke repository.
- Jangan menggunakan service role key pada kode yang berjalan di browser.
- Jangan menggunakan kredensial produksi untuk pengujian yang berpotensi mengubah data.

### 5.6. Siapkan Ollama

Unduh model:

```bash
ollama pull llama3:8b
```

Pastikan layanan Ollama berjalan sebelum menggunakan fitur AI.

### 5.7. Siapkan Database Supabase

Buka **SQL Editor** pada dashboard Supabase.

Periksa dan jalankan migrasi yang diperlukan dari direktori:

```text
database/migrations/
```

Jalankan sesuai urutan dependensi dan petunjuk pada masing-masing migrasi.

Gunakan proyek Supabase khusus pengembangan saat menyiapkan aplikasi untuk pertama kali. Periksa dampak setiap migrasi sebelum mengeksekusinya.

### 5.8. Jalankan Aplikasi

```bash
streamlit run app.py
```

Secara default, aplikasi Streamlit dapat diakses melalui:

```text
http://localhost:8501
```

---

## 6. Pengujian

Proyek menggunakan **Pytest** untuk membantu memverifikasi logika bisnis, validasi transaksi, autentikasi, pengaturan hak akses, dan pemrosesan AI.

Jalankan pengujian melalui terminal:

```bash
python -m pytest -v
```

Pengujian yang menggunakan mock dapat dijalankan tanpa koneksi langsung ke layanan eksternal, bergantung pada konfigurasi pengujian.

### Cakupan Pengujian

Modul pengujian yang tersedia mencakup:

- Administrasi pengguna.
- Autentikasi.
- Validasi transaksi.
- Aturan klasifikasi AI.
- Layanan AI.
- Hak akses kasir.
- Pengelolaan keuangan.
- Keamanan impor.
- Pembuatan laporan.
- Aturan bisnis aplikasi.

### Status Pengujian

Jumlah pengujian yang lulus harus mengikuti hasil eksekusi Pytest pada versi source code yang sedang digunakan.

Pengujian unit dengan mock tidak menggantikan pengujian integrasi langsung terhadap Supabase, Ollama, maupun kebijakan Row Level Security.

Sebelum merilis versi stabil, aplikasi perlu menjalani pengujian fungsional, integrasi, regresi, dan pemeriksaan keamanan yang relevan.

---

## 7. Roadmap Pengembangan

Pengembangan dilakukan secara bertahap menggunakan pendekatan **feature-based development**.

Setiap fitur diimplementasikan, diuji, diperbaiki jika diperlukan, kemudian didokumentasikan melalui Git commit.

| Tahap | Fokus Pengembangan | Status |
|---|---|---|
| Baseline | Fondasi aplikasi, autentikasi, transaksi, dashboard, integrasi AI awal | Implementasi awal tersedia; pengujian menyeluruh belum diverifikasi |
| 1 | Dataset referensi dan preprocessing | Direncanakan |
| 2 | SentenceTransformers dan FAISS | Direncanakan |
| 3 | Klasifikasi Llama 3 dengan RAG | Direncanakan |
| 4 | Deteksi kategori `Others` | Direncanakan |
| 5 | Klasifikasi transaksi massal berbasis RAG | Direncanakan |
| 6 | RAG untuk tanya jawab riwayat transaksi | Direncanakan |
| 7 | AI Financial Chat Assistant | Direncanakan |
| 8 | Forecasting pengeluaran menggunakan Prophet | Direncanakan |
| 9 | Evaluasi performa model AI | Direncanakan |
| 10 | Pengujian integrasi akhir dan dokumentasi | Direncanakan |

Beberapa fitur dasar telah memiliki implementasi awal dan dapat dikembangkan lebih lanjut tanpa membangun ulang aplikasi.

### Target Evaluasi AI

Sesuai tujuan proyek capstone, pengembangan AI diarahkan untuk memenuhi:

- **Akurasi klasifikasi transaksi minimal 85%.**
- **MAPE forecasting maksimal 20%.**
- Evaluasi klasifikasi dengan dan tanpa RAG.
- Pengujian transaksi di luar referensi menggunakan kategori `Others`.
- Pengukuran MAE dan MAPE untuk forecasting.

Target tersebut merupakan kriteria evaluasi dan belum dinyatakan tercapai sampai tersedia hasil pengujian yang dapat direproduksi.

### Rencana AI Financial Chat Assistant

Fitur percakapan direncanakan untuk memungkinkan pengguna:

- Mencatat pemasukan melalui bahasa alami.
- Mencatat pengeluaran melalui bahasa alami.
- Mengirim beberapa transaksi dalam satu percakapan.
- Memeriksa dan mengonfirmasi transaksi sebelum disimpan.
- Menanyakan ringkasan arus kas.
- Menelusuri riwayat transaksi.
- Meminta penjelasan kondisi keuangan berdasarkan data toko.

Fitur chat akan menggunakan layanan transaksi dan aturan hak akses yang sama dengan antarmuka aplikasi lainnya.

---

## 8. Keamanan dan Privasi

Pengembangan aplikasi memperhatikan beberapa aspek keamanan:

**Environment Variables**

Kredensial disimpan melalui konfigurasi lingkungan dan tidak boleh dimasukkan ke source code atau repository publik.

**Role-Based Access Control**

Akses pengguna dibatasi berdasarkan peran superadmin, pemilik toko, dan kasir.

**Row Level Security**

Supabase PostgreSQL mendukung penerapan RLS untuk membatasi akses data berdasarkan konteks pengguna dan toko.

Efektivitas aturan RLS harus dibuktikan melalui pengujian terhadap database yang dikonfigurasi.

**Local AI Inference**

Pemrosesan model Llama 3 dilakukan melalui Ollama lokal. Penyimpanan transaksi tetap menggunakan Supabase, sehingga keamanan data juga bergantung pada konfigurasi database, koneksi jaringan, dan kebijakan akses.

**Transaction Validation**

Input dan output model AI perlu divalidasi sebelum digunakan untuk membuat atau mengubah transaksi.

**Human Confirmation**

Untuk pengembangan fitur chat, tindakan penyimpanan, perubahan, dan penghapusan transaksi direncanakan memerlukan konfirmasi pengguna.

---

## 9. Kontribusi

Kontribusi terhadap pengembangan proyek terbuka untuk komunitas.

Alur kontribusi yang disarankan:

1. Fork repository.
2. Buat branch untuk fitur atau perbaikan.
3. Implementasikan perubahan.
4. Jalankan pengujian yang relevan.
5. Buat commit dengan pesan yang jelas.
6. Ajukan Pull Request disertai penjelasan perubahan.

Kontribusi diharapkan menjaga kompatibilitas dengan arsitektur proyek dan tidak menyertakan kredensial, data transaksi pribadi, atau informasi sensitif lainnya.

---

## 10. Lisensi

Proyek **AI-Powered Financial Recording & Cash Flow Assistant** didistribusikan menggunakan **MIT License**.

Kode sumber dapat digunakan, disalin, dimodifikasi, digabungkan, dipublikasikan, dan didistribusikan, termasuk untuk penggunaan komersial, sesuai ketentuan lisensi MIT.

Penggunaan ulang wajib mempertahankan pemberitahuan hak cipta dan teks lisensi sebagaimana ditentukan dalam MIT License.

Lihat file [LICENSE](LICENSE) untuk ketentuan lengkap.

---

## 11. Catatan Pengembangan

Repository ini merupakan bagian dari pengembangan proyek capstone **Dicoding Bootcamp — DB15-I023**.

Dokumentasi dan implementasi akan diperbarui secara bertahap mengikuti perkembangan fitur, hasil pengujian, dan evaluasi model.

Proyek dikembangkan sebagai perangkat lunak open-source yang dapat dipelajari, digunakan, dan dikembangkan lebih lanjut oleh komunitas.