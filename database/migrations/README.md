# Database Migrations Guide

Panduan urutan eksekusi, instruksi penerapan manual ke Supabase, dan langkah verifikasi migrasi skema database.

> [!IMPORTANT]
> Proyek ini tidak menggunakan automated migration runner bawaan di runtime aplikasi. Migrasi baru (`006` hingga `011`) **wajib dijalankan secara manual** pada Supabase SQL Editor atau melalui koneksi `psql` sesuai urutan dependensi di bawah.

---

## 1. Urutan Eksekusi Migrasi (Apply Order)

Terapkan seluruh migrasi secara berurutan dari `001` hingga `011`:

| No. | File Migrasi | Deskripsi Singkat |
|---|---|---|
| 001 | `001_initial.sql` | Skema dasar multi-tenant (`toko`, `profiles`, `kategori`, `transaksi`), RLS dasar, dan trigger `updated_at`. |
| 002 | `002_bootstrap_superadmin.sql` | Fungsi bootstrap aman untuk promosi akun superadmin pertama saat sistem diinisialisasi. |
| 003 | `003_stage2_finance.sql` | Bidang usaha multi-tenant, kuota kategori per jenis, dan pemisahan kategori non-operasional (`bukan_operasional`). |
| 004 | `004_fix_cashier_provisioning.sql` | Sinkronisasi trigger pembuatan akun kasir, penataan profil, dan perbaikan batasan RLS kasir. |
| 005 | `005_ai_forecast.sql` | Metadata klasifikasi AI (`status_kategori`, `usulan`, `skor`, `batch_id`), versi RAG, approval atomik, merge kategori, dan fungsi ringkasan superadmin. |
| 006 | `006_transaction_deletion_requests.sql` | Tabel alur persetujuan hapus transaksi kasir (`transaction_deletion_requests`), RPC approval/rejection superadmin/owner, dan kebijakan RLS. |
| 007 | `007_import_idempotency.sql` | Ledger identitas impor (`import_transaction_identities`), drop overload 3-arg legacy, dan RPC 4-arg `import_transactions_atomic` bergaransi idempotensi. |
| 008 | `008_chat_proposal_idempotency.sql` | Partial unique index (`transaksi_shop_proposal_unique`) pada `transaksi` untuk mencegah eksekusi ganda proposal chat AI oleh concurrent worker. |
| 009 | `009_audit_log.sql` | Tabel append-only audit trail (`audit_log`) untuk mutasi transaksi/owner, pembatasan akses eksklusif `service_role`, dan safeguard drop RPC 3-arg legacy. |
| 010 | `010_atomic_audit_mutations.sql` | RPC mutasi finansial atomik (`mutate_finance_audited`), pembungkus backward-compatible `approve_transactions_atomic`, dan pencatatan audit log terpadu. |
| 011 | `011_nullable_kategori_for_ai_pending.sql` | Kategori NULL hanya untuk antrean AI; impor tetap idempoten; retry terproteksi dan audit atomik. |

---

## 2. Cara Menerapkan Migrasi ke Supabase

### Metode A: Supabase Dashboard (SQL Editor) — Direkomendasikan
1. Masuk ke **Dashboard Supabase** proyek Anda.
2. Buka menu **SQL Editor** pada navigasi sebelah kiri.
3. Klik **New Query**.
4. Buka file migrasi (mulai dari `001_initial.sql` hingga `011_nullable_kategori_for_ai_pending.sql`), salin seluruh isinya, tempel ke SQL Editor, lalu klik tombol **Run** (atau `Ctrl+Enter`).
5. Ulangi secara berurutan untuk file berikutnya hingga `011_nullable_kategori_for_ai_pending.sql`.

### Metode B: Menggunakan PostgreSQL CLI (`psql`)
Jika memiliki connection string PostgreSQL langsung dari Supabase:

```bash
# Set connection string Supabase
export DB_URL="postgresql://postgres:[PASSWORD]@[HOST]:[PORT]/postgres"

# Eksekusi berurutan
psql "$DB_URL" -f database/migrations/001_initial.sql
psql "$DB_URL" -f database/migrations/002_bootstrap_superadmin.sql
psql "$DB_URL" -f database/migrations/003_stage2_finance.sql
psql "$DB_URL" -f database/migrations/004_fix_cashier_provisioning.sql
psql "$DB_URL" -f database/migrations/005_ai_forecast.sql
psql "$DB_URL" -f database/migrations/006_transaction_deletion_requests.sql
psql "$DB_URL" -f database/migrations/007_import_idempotency.sql
psql "$DB_URL" -f database/migrations/008_chat_proposal_idempotency.sql
psql "$DB_URL" -f database/migrations/009_audit_log.sql
psql "$DB_URL" -f database/migrations/010_atomic_audit_mutations.sql
psql "$DB_URL" -f database/migrations/011_nullable_kategori_for_ai_pending.sql
```

---

## 3. Langkah Verifikasi Penerapan (Verification Checklist)

Jalankan query pemeriksaan berikut di Supabase SQL Editor atau `psql` untuk memastikan seluruh objek database terpasang dengan benar:

### 1. Verifikasi RPC `import_transactions_atomic` (Hanya 1 Fungsi, 4 Argumen)
Pastikan fungsi overload 3-argumen lama telah terhapus, dan hanya fungsi 4-argumen terlindungi yang aktif:

```sql
SELECT 
    proname AS function_name, 
    pg_get_function_identity_arguments(oid) AS arguments
FROM pg_proc 
WHERE proname = 'import_transactions_atomic';
```
**Hasil yang diharapkan:** Tepat **1 baris**:
- `import_transactions_atomic` | `p_actor uuid, p_shop uuid, p_rows jsonb, p_batch uuid`

*(Jika menggunakan `psql`, jalankan `\df public.import_transactions_atomic`)*

### 2. Verifikasi Unique Index Idempotensi Proposal Chat AI (`008`)
```sql
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'transaksi' 
  AND indexname = 'transaksi_shop_proposal_unique';
```
**Hasil yang diharapkan:**
Indeks ditemukan dengan predikat:
`WHERE (nullif((usulan ->> 'proposal_id'::text), ''::text) IS NOT NULL)`

### 3. Verifikasi Tabel Audit Trail (`009`) dan Keamanan RLS
```sql
SELECT 
    relname AS table_name,
    relrowsecurity AS rls_enabled
FROM pg_class
WHERE relname = 'audit_log';
```
**Hasil yang diharapkan:** `table_name: audit_log`, `rls_enabled: true`.

Periksa permission:
```sql
SELECT 
    grantee, 
    privilege_type 
FROM information_schema.role_table_grants 
WHERE table_name = 'audit_log';
```
**Hasil yang diharapkan:** Hanya `service_role` (atau superuser postgres) yang memiliki `SELECT` dan `INSERT`. Role `anon` dan `authenticated` tidak memiliki akses.

### 4. Verifikasi Ledger Identitas Impor (`007`)
```sql
SELECT 
    table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name = 'import_transaction_identities';
```
**Hasil yang diharapkan:** `import_transaction_identities` ditemukan.

### 5. Verifikasi Alur Permintaan Hapus Transaksi (`006`)
```sql
SELECT 
    table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name = 'transaction_deletion_requests';
```
**Hasil yang diharapkan:** `transaction_deletion_requests` ditemukan.

### 6. Verifikasi RPC Mutasi Finansial Atomik (`010`)
```sql
SELECT 
    proname AS function_name, 
    pg_get_function_identity_arguments(oid) AS arguments
FROM pg_proc 
WHERE proname = 'mutate_finance_audited';
```
**Hasil yang diharapkan:** Tepat **1 baris**:
- `mutate_finance_audited` | `p_actor uuid, p_shop uuid, p_action text, p_ids uuid[], p_values jsonb`


### 011: verifikasi antrean AI

Migrasi idempoten: tidak mengubah baris lama dan mempertahankan FK kategori yang sama. FK PostgreSQL menerima NULL (MATCH SIMPLE). Penulisan baru tanpa kategori harus membawa `usulan.ai_pending=true`, `status_kategori=perlu_tinjauan`, dan `status_persetujuan=pending`. Definisi aktif `import_transactions_atomic` dari 007 diganti di 011, bukan mengedit 005/007. RPC mutasi 010 juga diganti di 011 untuk menjaga aturan kategori dan menolak approval sebelum antrean selesai. `retry_pending_ai_classification` memeriksa aktor, toko, versi baris, marker, dan status persetujuan dalam row lock; perubahan kategori, penghapusan marker, dan audit berada dalam transaksi yang sama. Terapkan 011 sebelum menjalankan kode antrean baru. Penerapan dua kali dan uji NULL dengan/tanpa marker sudah diuji pada PostgreSQL sementara; penerapan Supabase aplikasi belum dilakukan karena sesi ini tidak memiliki koneksi SQL terautentikasi.
