-- Tahap 2: jalankan sekali SETELAH migrasi 001 dan 002. Tidak menghapus data yang sudah ada.
create table public.kategori (
 id uuid primary key default gen_random_uuid(),
 toko_id uuid not null references public.toko(id) on delete cascade,
 nama text not null check(length(trim(nama))>0),
 tipe text not null check(tipe in ('pemasukan','pengeluaran')),
 deskripsi text not null default '',
 sistem boolean not null default false,
 bukan_operasional boolean not null default false,
 created_at timestamptz not null default now(),
 unique(toko_id,tipe,nama),
 unique(id,toko_id,tipe),
 constraint capital_only_income check(not bukan_operasional or tipe='pemasukan')
);
create table public.transaksi (
 id uuid primary key default gen_random_uuid(),
 toko_id uuid not null references public.toko(id) on delete cascade,
 tanggal date not null,
 deskripsi text not null check(length(trim(deskripsi))>0),
 nominal numeric(18,2) not null check(nominal>0),
 tipe text not null check(tipe in ('pemasukan','pengeluaran')),
 kategori_id uuid not null,
 status_persetujuan text not null check(status_persetujuan in ('pending','disetujui','ditolak')),
 dicatat_oleh uuid not null references public.profiles(id),
 disetujui_oleh uuid references public.profiles(id),
 disetujui_pada timestamptz,
 created_at timestamptz not null default now(),
 updated_at timestamptz not null default now(),
 foreign key(kategori_id,toko_id,tipe) references public.kategori(id,toko_id,tipe) on update restrict on delete restrict
);
create index transaksi_shop_date on public.transaksi(toko_id,tanggal desc);
create index transaksi_shop_status on public.transaksi(toko_id,status_persetujuan);
create index kategori_shop_type on public.kategori(toko_id,tipe);
alter table public.kategori enable row level security;
alter table public.transaksi enable row level security;
-- Kebijakan HANYA BACA untuk klien terautentikasi. Perubahan data HANYA melalui layanan server tepercaya.
create policy kategori_read on public.kategori for select to authenticated using (
 exists(select 1 from public.profiles p where p.id=(select auth.uid()) and p.toko_id=kategori.toko_id and p.status='aktif')
);
create policy transaksi_read on public.transaksi for select to authenticated using (
 exists(select 1 from public.profiles p where p.id=(select auth.uid()) and p.toko_id=transaksi.toko_id and p.status='aktif' and (p.role='pemilik_toko' or (p.role='kasir' and transaksi.dicatat_oleh=p.id)))
);
-- Pemeriksaan sisi server membatasi setiap operasi penulisan data ke toko_id milik aktor yang telah terverifikasi.
