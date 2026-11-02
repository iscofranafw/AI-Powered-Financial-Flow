-- ======================================================================
-- AI-Powered Financial Flow (BizFlow) - Consolidated Database Schema
-- ======================================================================
-- This file defines the complete, unified database schema representing
-- the final state produced by migrations 001 through 011.
--
-- Instructions:
-- - Run ONCE on an empty database (Supabase SQL Editor) for new setups.
-- - If you already applied migrations 001-011, do NOT run this file -
--   it produces the same final state.
--
-- Order of objects:
-- 1. Extensions
-- 2. Tables & Constraints (Foreign-Key Safe Order)
-- 3. Indexes
-- 4. Stored Procedures, Functions, & Triggers (Final State)
-- 5. Row Level Security (RLS) & Policies
-- 6. Seed Data & Superadmin Bootstrap (Safe & Idempotent)
-- ======================================================================

BEGIN;

-- ======================================================================
-- 1. EXTENSIONS
-- ======================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ======================================================================
-- 2. TABLES & CONSTRAINTS (Foreign-Key Safe Order)
-- ======================================================================

-- 2.1 Toko (from 001_initial.sql)
CREATE TABLE IF NOT EXISTS public.toko (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    nama text NOT NULL CHECK (length(trim(nama)) > 0),
    bidang_usaha text NOT NULL CHECK (length(trim(bidang_usaha)) > 0),
    created_at timestamptz NOT NULL DEFAULT now()
);

-- 2.2 Profiles (from 001_initial.sql)
CREATE TABLE IF NOT EXISTS public.profiles (
    id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email text NOT NULL,
    nama text NOT NULL,
    role text NOT NULL CHECK (role IN ('superadmin', 'pemilik_toko', 'kasir')),
    toko_id uuid REFERENCES public.toko(id),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'aktif', 'nonaktif', 'ditolak')),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT shop_role CHECK (
        (role = 'superadmin' AND toko_id IS NULL) OR
        (role <> 'superadmin' AND toko_id IS NOT NULL)
    )
);

-- 2.3 Kategori (from 003_stage2_finance.sql, updated with kata_kunci from 005_ai_forecast.sql)
CREATE TABLE IF NOT EXISTS public.kategori (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    toko_id uuid NOT NULL REFERENCES public.toko(id) ON DELETE CASCADE,
    nama text NOT NULL CHECK (length(trim(nama)) > 0),
    tipe text NOT NULL CHECK (tipe IN ('pemasukan', 'pengeluaran')),
    deskripsi text NOT NULL DEFAULT '',
    sistem boolean NOT NULL DEFAULT false,
    bukan_operasional boolean NOT NULL DEFAULT false,
    kata_kunci text[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (toko_id, tipe, nama),
    UNIQUE (id, toko_id, tipe),
    CONSTRAINT capital_only_income CHECK (NOT bukan_operasional OR tipe = 'pemasukan')
);

-- 2.4 Transaksi (from 003_stage2_finance.sql, 005_ai_forecast.sql, 011_nullable_kategori_for_ai_pending.sql)
-- Final state: kategori_id is nullable for AI-pending queue rows; contains AI classification metadata columns.
CREATE TABLE IF NOT EXISTS public.transaksi (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    toko_id uuid NOT NULL REFERENCES public.toko(id) ON DELETE CASCADE,
    tanggal date NOT NULL,
    deskripsi text NOT NULL CHECK (length(trim(deskripsi)) > 0),
    nominal numeric(18,2) NOT NULL CHECK (nominal > 0),
    tipe text NOT NULL CHECK (tipe IN ('pemasukan', 'pengeluaran')),
    kategori_id uuid,
    status_persetujuan text NOT NULL CHECK (status_persetujuan IN ('pending', 'disetujui', 'ditolak')),
    dicatat_oleh uuid NOT NULL REFERENCES public.profiles(id),
    disetujui_oleh uuid REFERENCES public.profiles(id),
    disetujui_pada timestamptz,
    status_kategori text NOT NULL DEFAULT 'dikonfirmasi' CHECK (status_kategori IN ('otomatis', 'perlu_tinjauan', 'dikonfirmasi')),
    usulan jsonb NOT NULL DEFAULT '{}'::jsonb,
    skor numeric(5,4),
    batch_id uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (kategori_id, toko_id, tipe) REFERENCES public.kategori(id, toko_id, tipe) ON UPDATE RESTRICT ON DELETE RESTRICT,
    CONSTRAINT status_kategori_valid CHECK (status_kategori IN ('otomatis', 'perlu_tinjauan', 'dikonfirmasi')),
    CONSTRAINT transaksi_ai_pending_category CHECK (
        kategori_id IS NOT NULL OR (
            coalesce(usulan->>'ai_pending', 'false') = 'true'
            AND status_kategori = 'perlu_tinjauan'
            AND status_persetujuan = 'pending'
        )
    )
);

-- 2.5 Cashier Provisioning (from 004_fix_cashier_provisioning.sql)
CREATE TABLE IF NOT EXISTS public.cashier_provisioning (
    token_hash text PRIMARY KEY,
    email text NOT NULL,
    nama text NOT NULL,
    toko_id uuid NOT NULL REFERENCES public.toko(id),
    created_by uuid NOT NULL REFERENCES public.profiles(id),
    expires_at timestamptz NOT NULL,
    consumed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- 2.6 RAG Index Version (from 005_ai_forecast.sql)
CREATE TABLE IF NOT EXISTS public.rag_index_version (
    toko_id uuid NOT NULL REFERENCES public.toko(id) ON DELETE CASCADE,
    tipe text NOT NULL CHECK (tipe IN ('pengeluaran', 'pemasukan')),
    version bigint NOT NULL DEFAULT 1,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (toko_id, tipe)
);

-- 2.7 Transaction Deletion Requests (from 006_transaction_deletion_requests.sql)
CREATE TABLE IF NOT EXISTS public.transaction_deletion_requests (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    store_id uuid NOT NULL REFERENCES public.toko(id) ON DELETE CASCADE,
    requested_by uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    requested_at timestamptz NOT NULL DEFAULT now(),
    status text NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'cancelled', 'failed', 'completed')) DEFAULT 'pending',
    reviewed_by uuid REFERENCES public.profiles(id),
    reviewed_at timestamptz,
    executed_at timestamptz,
    transaction_count_requested integer NOT NULL DEFAULT 0 CHECK (transaction_count_requested >= 0),
    transaction_count_deleted integer NOT NULL DEFAULT 0 CHECK (transaction_count_deleted >= 0),
    rejection_reason text,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- 2.8 Import Transaction Identities (from 007_import_idempotency.sql)
CREATE TABLE IF NOT EXISTS public.import_transaction_identities (
    toko_id uuid NOT NULL,
    identity_key text NOT NULL,
    PRIMARY KEY (toko_id, identity_key)
);

-- 2.9 Audit Log (from 009_audit_log.sql)
CREATE TABLE IF NOT EXISTS public.audit_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id uuid NOT NULL,
    action text NOT NULL CHECK (action IN (
        'update_transaction', 'delete_transaction', 'approve_transactions', 'change_owner_status')),
    target_table text NOT NULL CHECK (target_table IN ('transaksi', 'profiles')),
    target_id uuid NOT NULL,
    toko_id uuid,
    before_json jsonb,
    after_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (before_json IS NOT NULL OR after_json IS NOT NULL)
);

COMMENT ON TABLE public.audit_log IS
 'Independent append-only server audit records; usulan.audit_history is secondary. Service mutations and audit records are executed atomically in database RPCs (e.g. mutate_finance_audited).';

-- ======================================================================
-- 3. INDEXES
-- ======================================================================

-- From 001_initial.sql
CREATE INDEX IF NOT EXISTS profiles_toko_idx ON public.profiles(toko_id);

-- From 003_stage2_finance.sql
CREATE INDEX IF NOT EXISTS transaksi_shop_date ON public.transaksi(toko_id, tanggal DESC);
CREATE INDEX IF NOT EXISTS transaksi_shop_status ON public.transaksi(toko_id, status_persetujuan);
CREATE INDEX IF NOT EXISTS kategori_shop_type ON public.kategori(toko_id, tipe);

-- From 006_transaction_deletion_requests.sql
CREATE INDEX IF NOT EXISTS idx_del_req_store_status ON public.transaction_deletion_requests(store_id, status);
CREATE INDEX IF NOT EXISTS idx_del_req_status ON public.transaction_deletion_requests(status);
CREATE INDEX IF NOT EXISTS idx_del_req_requested_at ON public.transaction_deletion_requests(requested_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_pending_deletion_per_store
    ON public.transaction_deletion_requests(store_id)
    WHERE status = 'pending';

-- From 007_import_idempotency.sql
CREATE INDEX IF NOT EXISTS transaksi_import_source_lookup
    ON public.transaksi(toko_id, lower(btrim(usulan->>'source_id')));

-- From 008_chat_proposal_idempotency.sql
CREATE UNIQUE INDEX IF NOT EXISTS transaksi_shop_proposal_unique
    ON public.transaksi(toko_id, (usulan->>'proposal_id'))
    WHERE nullif(usulan->>'proposal_id', '') IS NOT NULL;

-- From 009_audit_log.sql
CREATE INDEX IF NOT EXISTS audit_log_shop_created ON public.audit_log(toko_id, created_at DESC);
CREATE INDEX IF NOT EXISTS audit_log_target ON public.audit_log(target_table, target_id, created_at DESC);

-- ======================================================================
-- 4. FUNCTIONS, RPCS, & TRIGGERS (Final State)
-- ======================================================================

-- 4.1 create_profile_on_signup (final from 004_fix_cashier_provisioning.sql)
CREATE OR REPLACE FUNCTION public.create_profile_on_signup()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
DECLARE
 v_shop uuid;
 v_name text;
 v_ticket public.cashier_provisioning%ROWTYPE;
 v_token text;
BEGIN
 v_name := coalesce(nullif(trim(new.raw_user_meta_data->>'nama'),''),'Pengguna');
 v_token := nullif(new.raw_user_meta_data->>'bizflow_cashier_ticket','');
 IF v_token IS NOT NULL THEN
   -- Rahasia, sekali pakai, terikat pada email dan pemilik toko yang sah; pendaftaran publik tidak dapat memalsukannya.
   SELECT * INTO v_ticket FROM public.cashier_provisioning
   WHERE token_hash = pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(v_token,'UTF8')),'hex')
     AND lower(email) = lower(new.email)
     AND expires_at > now() AND consumed_at IS NULL FOR UPDATE;
   IF NOT FOUND THEN RAISE EXCEPTION 'Tiket kasir tidak sah atau telah kedaluwarsa'; END IF;
   IF NOT EXISTS (SELECT 1 FROM public.profiles p
       WHERE p.id=v_ticket.created_by AND p.role='pemilik_toko'
         AND p.status='aktif' AND p.toko_id=v_ticket.toko_id) THEN
       RAISE EXCEPTION 'Pemilik toko kasir tidak aktif';
   END IF;
   INSERT INTO public.profiles(id,email,nama,role,toko_id,status)
     VALUES (new.id,new.email,v_ticket.nama,'kasir',v_ticket.toko_id,'aktif');
   UPDATE public.cashier_provisioning SET consumed_at=now()
     WHERE token_hash=v_ticket.token_hash;
 ELSE
   -- Pendaftaran publik HANYA dapat membuat akun pemilik toko berstatus pending, terlepas dari metadata yang dikirim.
   IF coalesce(trim(new.raw_user_meta_data->>'nama_toko'),'')='' OR
      coalesce(trim(new.raw_user_meta_data->>'bidang_usaha'),'')='' THEN
       RAISE EXCEPTION 'Nama toko dan bidang usaha wajib diisi';
   END IF;
   INSERT INTO public.toko(nama,bidang_usaha)
      VALUES(trim(new.raw_user_meta_data->>'nama_toko'),trim(new.raw_user_meta_data->>'bidang_usaha'))
      RETURNING id INTO v_shop;
   INSERT INTO public.profiles(id,email,nama,role,toko_id,status)
      VALUES(new.id,new.email,v_name,'pemilik_toko',v_shop,'pending');
 END IF;
 RETURN new;
END; $$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.create_profile_on_signup();

-- 4.2 bump_rag_index_version (from 005_ai_forecast.sql)
CREATE OR REPLACE FUNCTION public.bump_rag_index_version()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
DECLARE v_shop uuid; v_type text;
BEGIN
 IF TG_OP='DELETE' THEN
   v_shop := OLD.toko_id; v_type := OLD.tipe;
 ELSE
   v_shop := NEW.toko_id; v_type := NEW.tipe;
 END IF;
 INSERT INTO public.rag_index_version(toko_id,tipe,version,updated_at)
   VALUES(v_shop,v_type,1,now())
 ON CONFLICT (toko_id,tipe) DO UPDATE
   SET version = public.rag_index_version.version + 1, updated_at=now();
 IF TG_OP='UPDATE' AND (NEW.toko_id<>OLD.toko_id OR NEW.tipe<>OLD.tipe) THEN
   INSERT INTO public.rag_index_version(toko_id,tipe,version,updated_at)
     VALUES(OLD.toko_id,OLD.tipe,1,now())
     ON CONFLICT (toko_id,tipe) DO UPDATE SET version=public.rag_index_version.version+1,updated_at=now();
 END IF;
 IF TG_OP='DELETE' THEN RETURN OLD; END IF;
 RETURN NEW;
END; $$;

DROP TRIGGER IF EXISTS kategori_rag_bump ON public.kategori;
CREATE TRIGGER kategori_rag_bump
    AFTER INSERT OR UPDATE OR DELETE ON public.kategori
    FOR EACH ROW
    EXECUTE FUNCTION public.bump_rag_index_version();

DROP TRIGGER IF EXISTS transaksi_rag_bump ON public.transaksi;
CREATE TRIGGER transaksi_rag_bump
    AFTER INSERT OR UPDATE OR DELETE ON public.transaksi
    FOR EACH ROW
    EXECUTE FUNCTION public.bump_rag_index_version();

REVOKE ALL ON FUNCTION public.bump_rag_index_version() FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.bump_rag_index_version() TO service_role;

-- 4.3 admin_shop_summary (final 7-column version from 005_ai_forecast.sql)
CREATE OR REPLACE FUNCTION public.admin_shop_summary()
RETURNS TABLE(toko_id uuid, nama_toko text, bidang_usaha text, jumlah_pengguna bigint,
 jumlah_transaksi bigint, jumlah_kategori bigint, transaksi_terakhir date)
LANGUAGE sql SECURITY DEFINER SET search_path='' AS $$
 SELECT t.id, t.nama, t.bidang_usaha,
  (SELECT count(*) FROM public.profiles p WHERE p.toko_id=t.id)::bigint,
  (SELECT count(*) FROM public.transaksi tr WHERE tr.toko_id=t.id)::bigint,
  (SELECT count(*) FROM public.kategori k WHERE k.toko_id=t.id)::bigint,
  (SELECT max(tanggal) FROM public.transaksi tr WHERE tr.toko_id=t.id)
 FROM public.toko t;
$$;

REVOKE ALL ON FUNCTION public.admin_shop_summary() FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.admin_shop_summary() TO service_role;

-- 4.4 merge_category_atomic (from 005_ai_forecast.sql)
CREATE OR REPLACE FUNCTION public.merge_category_atomic(
  p_actor uuid, p_shop uuid, p_source uuid, p_target uuid
) RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path='' AS $$
DECLARE src public.kategori%ROWTYPE; dst public.kategori%ROWTYPE; n integer;
BEGIN
 IF NOT EXISTS(SELECT 1 FROM public.profiles WHERE id=p_actor AND toko_id=p_shop AND role='pemilik_toko' AND status='aktif')
   THEN RAISE EXCEPTION 'Tidak diizinkan'; END IF;
 SELECT * INTO src FROM public.kategori WHERE toko_id=p_shop AND id=p_source FOR UPDATE;
 SELECT * INTO dst FROM public.kategori WHERE toko_id=p_shop AND id=p_target FOR UPDATE;
 IF src.id IS NULL OR dst.id IS NULL OR src.id=dst.id OR src.tipe<>dst.tipe OR
    src.sistem OR src.bukan_operasional OR dst.bukan_operasional THEN
    RAISE EXCEPTION 'Kategori tidak dapat digabung'; END IF;
 UPDATE public.transaksi SET kategori_id=p_target,updated_at=now()
   WHERE toko_id=p_shop AND kategori_id=p_source;
 GET DIAGNOSTICS n=ROW_COUNT;
 DELETE FROM public.kategori WHERE toko_id=p_shop AND id=p_source;
 RETURN n;
END $$;

REVOKE ALL ON FUNCTION public.merge_category_atomic(uuid, uuid, uuid, uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.merge_category_atomic(uuid, uuid, uuid, uuid) TO service_role;

-- 4.5 request_store_transaction_deletion (from 006_transaction_deletion_requests.sql)
CREATE OR REPLACE FUNCTION public.request_store_transaction_deletion(p_actor uuid, p_shop uuid)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
DECLARE
    v_role text;
    v_active_shop uuid;
    v_count integer;
    v_new_req public.transaction_deletion_requests%ROWTYPE;
BEGIN
    SELECT role, toko_id INTO v_role, v_active_shop 
    FROM public.profiles 
    WHERE id = p_actor AND status = 'aktif';

    IF v_role IS NULL OR v_role <> 'pemilik_toko' OR v_active_shop <> p_shop THEN
        RAISE EXCEPTION 'Hanya pemilik toko aktif yang berhak mengajukan penghapusan transaksi tokonya';
    END IF;

    -- Cegah pengajuan ganda jika masih ada pengajuan yang berstatus pending
    IF EXISTS (SELECT 1 FROM public.transaction_deletion_requests WHERE store_id = p_shop AND status = 'pending') THEN
        RAISE EXCEPTION 'Sudah ada pengajuan penghapusan transaksi yang sedang menunggu persetujuan Superadmin';
    END IF;

    -- Hitung jumlah transaksi saat ini
    SELECT count(*)::integer INTO v_count FROM public.transaksi WHERE toko_id = p_shop;

    INSERT INTO public.transaction_deletion_requests (
        store_id, requested_by, status, transaction_count_requested
    ) VALUES (
        p_shop, p_actor, 'pending', v_count
    ) RETURNING * INTO v_new_req;

    RETURN to_jsonb(v_new_req);
END; $$;

REVOKE ALL ON FUNCTION public.request_store_transaction_deletion(uuid, uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.request_store_transaction_deletion(uuid, uuid) TO service_role;

-- 4.6 cancel_store_transaction_deletion (from 006_transaction_deletion_requests.sql)
CREATE OR REPLACE FUNCTION public.cancel_store_transaction_deletion(p_actor uuid, p_request_id uuid)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
DECLARE
    v_role text;
    v_active_shop uuid;
    v_req public.transaction_deletion_requests%ROWTYPE;
BEGIN
    SELECT role, toko_id INTO v_role, v_active_shop 
    FROM public.profiles 
    WHERE id = p_actor AND status = 'aktif';

    SELECT * INTO v_req FROM public.transaction_deletion_requests 
    WHERE id = p_request_id FOR UPDATE;

    IF v_req.id IS NULL THEN
        RAISE EXCEPTION 'Pengajuan tidak ditemukan';
    END IF;

    IF v_role <> 'pemilik_toko' OR v_active_shop <> v_req.store_id THEN
        RAISE EXCEPTION 'Tidak berhak membatalkan pengajuan toko lain';
    END IF;

    IF v_req.status <> 'pending' THEN
        RAISE EXCEPTION 'Hanya pengajuan dengan status pending yang dapat dibatalkan';
    END IF;

    UPDATE public.transaction_deletion_requests
    SET status = 'cancelled', updated_at = now()
    WHERE id = p_request_id
    RETURNING * INTO v_req;

    RETURN to_jsonb(v_req);
END; $$;

REVOKE ALL ON FUNCTION public.cancel_store_transaction_deletion(uuid, uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.cancel_store_transaction_deletion(uuid, uuid) TO service_role;

-- 4.7 reject_store_transaction_deletion (from 006_transaction_deletion_requests.sql)
CREATE OR REPLACE FUNCTION public.reject_store_transaction_deletion(p_admin uuid, p_request_id uuid, p_reason text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
DECLARE
    v_role text;
    v_req public.transaction_deletion_requests%ROWTYPE;
BEGIN
    SELECT role INTO v_role FROM public.profiles WHERE id = p_admin AND status = 'aktif';
    IF v_role IS NULL OR v_role <> 'superadmin' THEN
        RAISE EXCEPTION 'Hanya Superadmin yang berhak menolak pengajuan';
    END IF;

    SELECT * INTO v_req FROM public.transaction_deletion_requests 
    WHERE id = p_request_id FOR UPDATE;

    IF v_req.id IS NULL THEN
        RAISE EXCEPTION 'Pengajuan tidak ditemukan';
    END IF;

    IF v_req.status <> 'pending' THEN
        RAISE EXCEPTION 'Pengajuan tidak dalam status pending';
    END IF;

    UPDATE public.transaction_deletion_requests
    SET status = 'rejected', reviewed_by = p_admin, reviewed_at = now(),
        rejection_reason = coalesce(nullif(trim(p_reason), ''), 'Ditolak oleh Superadmin'),
        updated_at = now()
    WHERE id = p_request_id
    RETURNING * INTO v_req;

    RETURN to_jsonb(v_req);
END; $$;

REVOKE ALL ON FUNCTION public.reject_store_transaction_deletion(uuid, uuid, text) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.reject_store_transaction_deletion(uuid, uuid, text) TO service_role;

-- 4.8 execute_store_transaction_deletion (from 006_transaction_deletion_requests.sql)
CREATE OR REPLACE FUNCTION public.execute_store_transaction_deletion(p_admin uuid, p_request_id uuid)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
DECLARE
    v_role text;
    v_req public.transaction_deletion_requests%ROWTYPE;
    v_deleted_count integer := 0;
BEGIN
    SELECT role INTO v_role FROM public.profiles WHERE id = p_admin AND status = 'aktif';
    IF v_role IS NULL OR v_role <> 'superadmin' THEN
        RAISE EXCEPTION 'Hanya Superadmin yang berhak menyetujui dan mengeksekusi penghapusan massal';
    END IF;

    SELECT * INTO v_req FROM public.transaction_deletion_requests 
    WHERE id = p_request_id FOR UPDATE;

    IF v_req.id IS NULL THEN
        RAISE EXCEPTION 'Pengajuan tidak ditemukan';
    END IF;

    IF v_req.status <> 'pending' THEN
        RAISE EXCEPTION 'Pengajuan tidak dalam status pending (status saat ini: %)', v_req.status;
    END IF;

    -- Eksekusi penghapusan hanya untuk toko yang terkait
    DELETE FROM public.transaksi WHERE toko_id = v_req.store_id;
    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;

    UPDATE public.transaction_deletion_requests
    SET status = 'completed',
        reviewed_by = p_admin,
        reviewed_at = now(),
        executed_at = now(),
        transaction_count_deleted = v_deleted_count,
        updated_at = now()
    WHERE id = p_request_id
    RETURNING * INTO v_req;

    RETURN jsonb_build_object(
        'success', true,
        'request_id', v_req.id,
        'store_id', v_req.store_id,
        'deleted_count', v_deleted_count,
        'status', 'completed'
    );
EXCEPTION WHEN OTHERS THEN
    UPDATE public.transaction_deletion_requests
    SET status = 'failed',
        error_message = SQLERRM,
        updated_at = now()
    WHERE id = p_request_id;
    RAISE;
END; $$;

REVOKE ALL ON FUNCTION public.execute_store_transaction_deletion(uuid, uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.execute_store_transaction_deletion(uuid, uuid) TO service_role;

-- 4.9 import_transactions_atomic (final from 011_nullable_kategori_for_ai_pending.sql)
-- Drops legacy 3-argument signature if present to ensure 4-argument version is authoritative.
DROP FUNCTION IF EXISTS public.import_transactions_atomic(uuid, uuid, jsonb);

CREATE OR REPLACE FUNCTION public.import_transactions_atomic(
 p_actor uuid, p_shop uuid, p_rows jsonb, p_batch uuid
) RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path='' AS $$
DECLARE entry jsonb; n int:=0; v_role text; v_cat public.kategori%ROWTYPE; v_type text;
 v_meta jsonb; v_source text; v_import text; v_fp text; v_identity text;
 v_occurrence int; v_occurrences jsonb := '{}'::jsonb; v_claimed int;
BEGIN
 SELECT role INTO v_role FROM public.profiles WHERE id=p_actor AND toko_id=p_shop AND status='aktif';
 IF v_role IS NULL OR v_role NOT IN ('pemilik_toko','kasir') THEN RAISE EXCEPTION 'Tidak diizinkan'; END IF;
 IF jsonb_typeof(p_rows)<>'array' OR jsonb_array_length(p_rows)=0 OR jsonb_array_length(p_rows)>1000
   THEN RAISE EXCEPTION 'Batch harus terdiri dari 1-1000 transaksi'; END IF;
 IF p_batch IS NULL THEN RAISE EXCEPTION 'Batch ID wajib diisi'; END IF;
 -- Serialize imports per shop, including compatibility checks against pre-007 rows.
 -- The ledger primary key also protects concurrent identity claims.
 PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_shop::text, 7));
 FOR entry IN SELECT value FROM jsonb_array_elements(p_rows) LOOP
   v_type := entry->>'tipe';
   v_meta := coalesce(nullif(entry->'usulan', 'null'::jsonb), '{}'::jsonb);
   SELECT * INTO v_cat FROM public.kategori WHERE id=nullif(entry->>'kategori_id','')::uuid
       AND toko_id=p_shop AND tipe=v_type;
   IF v_cat.id IS NULL AND NOT (nullif(entry->>'kategori_id','') IS NULL AND coalesce(v_meta->>'ai_pending','false')='true')
   THEN RAISE EXCEPTION 'Kategori tidak cocok pada baris %',n+1; END IF;
   v_meta := coalesce(nullif(entry->'usulan', 'null'::jsonb), '{}'::jsonb);
   v_source := nullif(lower(btrim(coalesce(nullif(entry->>'source_id',''), v_meta->>'source_id', ''))), '');
   v_fp := (entry->>'tanggal')::date::text || '|' ||
     lower(btrim(regexp_replace(entry->>'deskripsi', '\s+', ' ', 'g'))) || '|' ||
     round((entry->>'nominal')::numeric, 2)::text || '|' || lower(entry->>'tipe');
   v_import := coalesce(nullif(v_meta->>'import_id', ''), p_batch::text);
   v_occurrence := coalesce((v_occurrences->>v_fp)::int, 0) + 1;
   v_occurrences := jsonb_set(v_occurrences, ARRAY[v_fp], to_jsonb(v_occurrence));
   v_occurrence := coalesce((v_meta->>'import_occurrence')::int, v_occurrence);
   IF v_occurrence < 1 THEN RAISE EXCEPTION 'Nomor kemunculan impor tidak valid'; END IF;
   v_identity := encode(sha256(convert_to(
     CASE WHEN v_source IS NOT NULL THEN jsonb_build_array('source', v_source)
     ELSE jsonb_build_array('import', v_import, v_fp, v_occurrence) END::text,
     'UTF8')), 'hex');
   IF v_source IS NOT NULL AND EXISTS (
     SELECT 1 FROM public.transaksi WHERE toko_id=p_shop
       AND lower(btrim(usulan->>'source_id'))=v_source
   ) THEN CONTINUE; END IF;
   INSERT INTO public.import_transaction_identities(toko_id, identity_key)
     VALUES(p_shop, v_identity) ON CONFLICT DO NOTHING;
   GET DIAGNOSTICS v_claimed = ROW_COUNT;
   IF v_claimed = 0 THEN CONTINUE; END IF;
   v_meta := v_meta || jsonb_build_object('import_id', v_import,
     'fingerprint', v_fp, 'import_occurrence', v_occurrence);
   IF v_source IS NOT NULL THEN
     v_meta := v_meta || jsonb_build_object('source_id', v_source);
   END IF;
   INSERT INTO public.transaksi(toko_id,tanggal,deskripsi,nominal,tipe,kategori_id,
     dicatat_oleh,status_persetujuan,disetujui_oleh,disetujui_pada,status_kategori,skor,usulan,batch_id)
     VALUES(p_shop,(entry->>'tanggal')::date,entry->>'deskripsi',(entry->>'nominal')::numeric,
     v_type,v_cat.id,p_actor,CASE WHEN v_role='pemilik_toko' AND coalesce(v_meta->>'ai_pending','false')<>'true' THEN 'disetujui' ELSE 'pending' END,
     CASE WHEN v_role='pemilik_toko' AND coalesce(v_meta->>'ai_pending','false')<>'true' THEN p_actor ELSE NULL END,
     CASE WHEN v_role='pemilik_toko' AND coalesce(v_meta->>'ai_pending','false')<>'true' THEN now() ELSE NULL END,
     CASE WHEN v_meta->>'ai_pending'='true' THEN 'perlu_tinjauan' ELSE coalesce(entry->>'status_kategori','dikonfirmasi') END,nullif(entry->>'skor','')::numeric,
     v_meta,p_batch);
   n := n+1;
 END LOOP;
 RETURN n;
END $$;

REVOKE ALL ON FUNCTION public.import_transactions_atomic(uuid, uuid, jsonb, uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.import_transactions_atomic(uuid, uuid, jsonb, uuid) TO service_role;

-- 4.10 mutate_finance_audited (final from 011_nullable_kategori_for_ai_pending.sql)
CREATE OR REPLACE FUNCTION public.mutate_finance_audited(
 p_actor uuid, p_shop uuid, p_action text, p_ids uuid[], p_values jsonb DEFAULT '{}'::jsonb
) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path='' AS $$
DECLARE
 actor public.profiles%ROWTYPE; owner_before public.profiles%ROWTYPE;
 owner_after public.profiles%ROWTYPE; old_row public.transaksi%ROWTYPE;
 new_row public.transaksi%ROWTYPE; meta jsonb; history jsonb;
 changed boolean; reset_approval boolean; n integer := 0; result jsonb := '[]'::jsonb;
BEGIN
 SELECT * INTO actor FROM public.profiles WHERE id=p_actor AND status='aktif' FOR SHARE;
 IF actor.id IS NULL THEN RAISE EXCEPTION 'Tidak diizinkan'; END IF;
 IF p_action IS NULL OR p_action NOT IN ('update_transaction','delete_transaction','approve_transactions','change_owner_status')
 THEN RAISE EXCEPTION 'Operasi tidak valid'; END IF;
 IF p_ids IS NULL OR cardinality(p_ids)=0 OR cardinality(p_ids) <> (SELECT count(DISTINCT x) FROM unnest(p_ids) u(x))
 THEN RAISE EXCEPTION 'ID tidak valid'; END IF;
 IF p_action <> 'approve_transactions' AND cardinality(p_ids) <> 1 THEN RAISE EXCEPTION 'Satu ID diperlukan'; END IF;
 IF p_action='change_owner_status' THEN
   IF actor.role <> 'superadmin' OR p_actor=p_ids[1] THEN RAISE EXCEPTION 'Hanya Superadmin dapat mengubah pemilik lain'; END IF;
   IF (p_values->>'status') IS NULL OR p_values->>'status' NOT IN ('aktif','ditolak','nonaktif') THEN RAISE EXCEPTION 'Status tidak valid'; END IF;
   SELECT * INTO owner_before FROM public.profiles WHERE id=p_ids[1] FOR UPDATE;
   IF owner_before.id IS NULL OR owner_before.role <> 'pemilik_toko' OR owner_before.toko_id IS DISTINCT FROM p_shop
   THEN RAISE EXCEPTION 'Pemilik toko tidak ditemukan'; END IF;
   UPDATE public.profiles SET status=p_values->>'status' WHERE id=owner_before.id RETURNING * INTO owner_after;
   INSERT INTO public.audit_log(actor_id,action,target_table,target_id,toko_id,before_json,after_json)
   VALUES(p_actor,p_action,'profiles',owner_before.id,p_shop,to_jsonb(owner_before),to_jsonb(owner_after));
   RETURN jsonb_build_array(to_jsonb(owner_after));
 END IF;
 IF actor.toko_id IS DISTINCT FROM p_shop OR actor.role NOT IN ('pemilik_toko','kasir')
    OR (p_action <> 'update_transaction' AND actor.role <> 'pemilik_toko')
 THEN RAISE EXCEPTION 'Tidak diizinkan'; END IF;
 IF p_action='approve_transactions' AND ((p_values->>'decision') IS NULL OR p_values->>'decision' NOT IN ('disetujui','ditolak'))
 THEN RAISE EXCEPTION 'Keputusan tidak valid'; END IF;
 PERFORM 1 FROM public.transaksi WHERE toko_id=p_shop AND id=ANY(p_ids) ORDER BY id FOR UPDATE;
 IF (SELECT count(*) FROM public.transaksi WHERE toko_id=p_shop AND id=ANY(p_ids)) <> cardinality(p_ids)
 THEN RAISE EXCEPTION 'Transaksi bukan milik toko / tidak ditemukan'; END IF;
 FOR old_row IN SELECT * FROM public.transaksi WHERE toko_id=p_shop AND id=ANY(p_ids) ORDER BY id LOOP
   IF p_action='delete_transaction' THEN
     DELETE FROM public.transaksi WHERE id=old_row.id;
     INSERT INTO public.audit_log(actor_id,action,target_table,target_id,toko_id,before_json,after_json)
     VALUES(p_actor,p_action,'transaksi',old_row.id,p_shop,to_jsonb(old_row),NULL);
   ELSE
     IF p_action='approve_transactions' THEN
       IF old_row.status_persetujuan <> 'pending' OR old_row.kategori_id IS NULL OR old_row.usulan->>'ai_pending'='true' THEN RAISE EXCEPTION 'Transaksi bukan pending'; END IF;
       UPDATE public.transaksi SET status_persetujuan=p_values->>'decision',
         disetujui_oleh=p_actor,disetujui_pada=now(),updated_at=now(),
         status_kategori=CASE WHEN p_values->>'decision'='disetujui' AND status_kategori='perlu_tinjauan' THEN 'dikonfirmasi' ELSE status_kategori END
       WHERE id=old_row.id RETURNING * INTO new_row;
     ELSE
       IF actor.role='kasir' AND (old_row.dicatat_oleh<>p_actor OR old_row.status_persetujuan NOT IN ('pending','ditolak'))
       THEN RAISE EXCEPTION 'Kasir tidak dapat mengubah transaksi ini'; END IF;
       meta := CASE WHEN jsonb_typeof(old_row.usulan)='object' THEN old_row.usulan ELSE '{}'::jsonb END;
       IF NOT EXISTS(SELECT 1 FROM public.kategori WHERE id=nullif(p_values->>'kategori_id','')::uuid AND toko_id=p_shop AND tipe=p_values->>'tipe')
          AND NOT (nullif(p_values->>'kategori_id','') IS NULL AND coalesce(meta->>'ai_pending','false')='true')
       THEN RAISE EXCEPTION 'Kategori tidak sesuai jenis transaksi/toko'; END IF;
       IF coalesce(length(btrim(p_values->>'deskripsi')),0) NOT BETWEEN 1 AND 500
          OR (p_values->>'tanggal') IS NULL OR (p_values->>'nominal') IS NULL
          OR (p_values->>'nominal')::numeric <= 0 THEN RAISE EXCEPTION 'Data transaksi tidak valid'; END IF;
       changed := ROW(old_row.tanggal,old_row.deskripsi,old_row.nominal,old_row.tipe,old_row.kategori_id)
          IS DISTINCT FROM ROW((p_values->>'tanggal')::date,p_values->>'deskripsi',(p_values->>'nominal')::numeric,p_values->>'tipe',nullif(p_values->>'kategori_id','')::uuid);
       reset_approval := actor.role='kasir' OR old_row.status_persetujuan='ditolak' OR (old_row.status_persetujuan='disetujui' AND changed);
       meta := CASE WHEN jsonb_typeof(old_row.usulan)='object' THEN old_row.usulan ELSE '{}'::jsonb END;
       IF nullif(p_values->>'kategori_id','') IS NOT NULL THEN meta := meta - 'ai_pending' - 'ai_error'; END IF;
       history := CASE WHEN jsonb_typeof(meta->'audit_history')='array' THEN meta->'audit_history' ELSE '[]'::jsonb END;
       meta := meta || jsonb_build_object('audit_history', history || jsonb_build_array(jsonb_build_object(
         'action','edit','editor_id',p_actor,'editor_role',actor.role,'timestamp',now(),
         'previous',jsonb_build_object('tanggal',old_row.tanggal,'deskripsi',old_row.deskripsi,'nominal',old_row.nominal::text,
           'tipe',old_row.tipe,'kategori_id',old_row.kategori_id,'status_persetujuan',old_row.status_persetujuan))));
       UPDATE public.transaksi SET tanggal=(p_values->>'tanggal')::date,deskripsi=p_values->>'deskripsi',
         nominal=(p_values->>'nominal')::numeric,tipe=p_values->>'tipe',kategori_id=nullif(p_values->>'kategori_id','')::uuid,
         status_kategori=CASE WHEN meta->>'ai_pending'='true' THEN 'perlu_tinjauan' ELSE 'dikonfirmasi' END,updated_at=now(),usulan=meta,
         status_persetujuan=CASE WHEN reset_approval THEN 'pending' ELSE old_row.status_persetujuan END,
         disetujui_oleh=CASE WHEN reset_approval THEN NULL ELSE old_row.disetujui_oleh END,
         disetujui_pada=CASE WHEN reset_approval THEN NULL ELSE old_row.disetujui_pada END
       WHERE id=old_row.id RETURNING * INTO new_row;
     END IF;
     INSERT INTO public.audit_log(actor_id,action,target_table,target_id,toko_id,before_json,after_json)
     VALUES(p_actor,p_action,'transaksi',old_row.id,p_shop,to_jsonb(old_row),to_jsonb(new_row));
     result := result || jsonb_build_array(to_jsonb(new_row));
   END IF;
   n := n+1;
 END LOOP;
 IF p_action='update_transaction' THEN RETURN result; END IF;
 RETURN to_jsonb(n);
END $$;

REVOKE ALL ON FUNCTION public.mutate_finance_audited(uuid, uuid, text, uuid[], jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.mutate_finance_audited(uuid, uuid, text, uuid[], jsonb) TO service_role;

-- 4.11 approve_transactions_atomic (final from 011_nullable_kategori_for_ai_pending.sql)
CREATE OR REPLACE FUNCTION public.approve_transactions_atomic(p_actor uuid, p_shop uuid, p_ids uuid[], p_decision text)
RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path='' AS $$
BEGIN
 RETURN public.mutate_finance_audited(p_actor, p_shop, 'approve_transactions', p_ids, jsonb_build_object('decision', p_decision))::integer;
END $$;

REVOKE ALL ON FUNCTION public.approve_transactions_atomic(uuid, uuid, uuid[], text) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.approve_transactions_atomic(uuid, uuid, uuid[], text) TO service_role;

-- 4.12 retry_pending_ai_classification (from 011_nullable_kategori_for_ai_pending.sql)
CREATE OR REPLACE FUNCTION public.retry_pending_ai_classification(
 p_actor uuid, p_shop uuid, p_id uuid, p_category uuid, p_updated_at timestamptz
) RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path='' AS $$
DECLARE actor public.profiles%ROWTYPE; old_row public.transaksi%ROWTYPE; new_row public.transaksi%ROWTYPE;
BEGIN
 SELECT * INTO actor FROM public.profiles WHERE id=p_actor AND toko_id=p_shop AND status='aktif' FOR SHARE;
 IF actor.id IS NULL OR actor.role NOT IN ('pemilik_toko','kasir') THEN RAISE EXCEPTION 'Tidak diizinkan'; END IF;
 SELECT * INTO old_row FROM public.transaksi WHERE id=p_id AND toko_id=p_shop FOR UPDATE;
 IF old_row.id IS NULL OR old_row.status_persetujuan='disetujui'
    OR coalesce(old_row.usulan->>'ai_pending','false')<>'true'
    OR old_row.updated_at IS DISTINCT FROM p_updated_at
    OR (actor.role='kasir' AND old_row.dicatat_oleh<>p_actor) THEN RETURN false; END IF;
 IF NOT EXISTS(SELECT 1 FROM public.kategori WHERE id=p_category AND toko_id=p_shop AND tipe=old_row.tipe)
 THEN RAISE EXCEPTION 'Kategori tidak cocok'; END IF;
 UPDATE public.transaksi SET kategori_id=p_category,status_kategori='dikonfirmasi',skor=NULL,
   usulan=(usulan-'ai_pending'-'ai_error') || jsonb_build_object('source','llm'),updated_at=now()
 WHERE id=p_id RETURNING * INTO new_row;
 INSERT INTO public.audit_log(actor_id,action,target_table,target_id,toko_id,before_json,after_json)
 VALUES(p_actor,'update_transaction','transaksi',p_id,p_shop,to_jsonb(old_row),to_jsonb(new_row));
 RETURN true;
END $$;

REVOKE ALL ON FUNCTION public.retry_pending_ai_classification(uuid, uuid, uuid, uuid, timestamptz) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.retry_pending_ai_classification(uuid, uuid, uuid, uuid, timestamptz) TO service_role;

-- ======================================================================
-- 5. ROW LEVEL SECURITY (RLS) & POLICIES
-- ======================================================================

-- 5.1 Toko
ALTER TABLE public.toko ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS shop_active_member ON public.toko;
CREATE POLICY shop_active_member ON public.toko FOR SELECT TO authenticated
    USING (EXISTS (SELECT 1 FROM public.profiles p WHERE p.id = (SELECT auth.uid()) AND p.toko_id = toko.id AND p.status = 'aktif'));

-- 5.2 Profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS profile_self ON public.profiles;
CREATE POLICY profile_self ON public.profiles FOR SELECT TO authenticated
    USING (id = (SELECT auth.uid()) AND status = 'aktif');

-- 5.3 Kategori
ALTER TABLE public.kategori ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS kategori_read ON public.kategori;
CREATE POLICY kategori_read ON public.kategori FOR SELECT TO authenticated
    USING (EXISTS (SELECT 1 FROM public.profiles p WHERE p.id = (SELECT auth.uid()) AND p.toko_id = kategori.toko_id AND p.status = 'aktif'));

-- 5.4 Transaksi
ALTER TABLE public.transaksi ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS transaksi_read ON public.transaksi;
CREATE POLICY transaksi_read ON public.transaksi FOR SELECT TO authenticated
    USING (EXISTS (
        SELECT 1 FROM public.profiles p
        WHERE p.id = (SELECT auth.uid())
          AND p.toko_id = transaksi.toko_id
          AND p.status = 'aktif'
          AND (p.role = 'pemilik_toko' OR (p.role = 'kasir' AND transaksi.dicatat_oleh = p.id))
    ));

-- 5.5 Cashier Provisioning
ALTER TABLE public.cashier_provisioning ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.cashier_provisioning FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.cashier_provisioning TO service_role;

-- 5.6 RAG Index Version
ALTER TABLE public.rag_index_version ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.rag_index_version FROM PUBLIC, anon, authenticated;
GRANT ALL ON public.rag_index_version TO service_role;

-- 5.7 Transaction Deletion Requests
ALTER TABLE public.transaction_deletion_requests ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.transaction_deletion_requests FROM PUBLIC, anon, authenticated;
GRANT ALL ON public.transaction_deletion_requests TO service_role;

DROP POLICY IF EXISTS superadmin_read_all_deletion_requests ON public.transaction_deletion_requests;
CREATE POLICY superadmin_read_all_deletion_requests ON public.transaction_deletion_requests
    FOR SELECT TO authenticated
    USING (EXISTS (SELECT 1 FROM public.profiles p WHERE p.id = (SELECT auth.uid()) AND p.role = 'superadmin' AND p.status = 'aktif'));

DROP POLICY IF EXISTS owner_read_own_deletion_requests ON public.transaction_deletion_requests;
CREATE POLICY owner_read_own_deletion_requests ON public.transaction_deletion_requests
    FOR SELECT TO authenticated
    USING (EXISTS (SELECT 1 FROM public.profiles p WHERE p.id = (SELECT auth.uid()) AND p.role = 'pemilik_toko' AND p.toko_id = transaction_deletion_requests.store_id AND p.status = 'aktif'));

-- 5.8 Import Transaction Identities
ALTER TABLE public.import_transaction_identities ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.import_transaction_identities FROM PUBLIC, anon, authenticated;
GRANT ALL ON public.import_transaction_identities TO service_role;

-- 5.9 Audit Log
ALTER TABLE public.audit_log ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.audit_log FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT, INSERT ON public.audit_log TO service_role;

DROP POLICY IF EXISTS audit_log_service_insert ON public.audit_log;
CREATE POLICY audit_log_service_insert ON public.audit_log
    FOR INSERT TO service_role WITH CHECK (true);

DROP POLICY IF EXISTS audit_log_service_read ON public.audit_log;
CREATE POLICY audit_log_service_read ON public.audit_log
    FOR SELECT TO service_role USING (true);

-- ======================================================================
-- 6. SEED DATA & SUPERADMIN BOOTSTRAP (from 002_bootstrap_superadmin.sql)
-- ======================================================================
-- To promote an initial user to superadmin:
-- 1. Register an account in the application (this automatically creates an initial shop).
-- 2. Replace 'REPLACE_WITH_YOUR_ADMIN_EMAIL' below with your registered email and run the block.
-- On a clean/empty database where no users exist yet, this block safely does nothing.
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
        RAISE NOTICE 'Superadmin candidate not found — create the admin user first (sign up with the admin email), then re-run only this DO block.';
    END IF;
END $$;

COMMIT;
