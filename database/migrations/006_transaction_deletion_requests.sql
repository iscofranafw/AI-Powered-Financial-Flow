-- Migrasi 006: Pengajuan Penghapusan Seluruh Transaksi Toko dengan Persetujuan Superadmin
-- Fitur ini mensyaratkan alur persetujuan ketat sebelum seluruh transaksi toko dihapus secara permanen.

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

-- Indeks untuk pencarian cepat berdasarkan toko dan status
CREATE INDEX IF NOT EXISTS idx_del_req_store_status ON public.transaction_deletion_requests(store_id, status);
CREATE INDEX IF NOT EXISTS idx_del_req_status ON public.transaction_deletion_requests(status);
CREATE INDEX IF NOT EXISTS idx_del_req_requested_at ON public.transaction_deletion_requests(requested_at DESC);

-- Batasan unik: Hanya boleh ada 1 pengajuan berstatus 'pending' per toko dalam satu waktu
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_pending_deletion_per_store 
ON public.transaction_deletion_requests(store_id) 
WHERE status = 'pending';

-- Keamanan Tingkat Baris (Row Level Security)
ALTER TABLE public.transaction_deletion_requests ENABLE ROW LEVEL SECURITY;

-- Cabut akses publik, anonim, dan terautentikasi langsung; akses hanya lewat kebijakan dan service_role
REVOKE ALL ON public.transaction_deletion_requests FROM PUBLIC, anon, authenticated;
GRANT ALL ON public.transaction_deletion_requests TO service_role;

-- Kebijakan baca untuk Superadmin (melihat seluruh pengajuan dari semua toko)
CREATE POLICY superadmin_read_all_deletion_requests ON public.transaction_deletion_requests
FOR SELECT TO authenticated USING (
    EXISTS (SELECT 1 FROM public.profiles p WHERE p.id = (SELECT auth.uid()) AND p.role = 'superadmin' AND p.status = 'aktif')
);

-- Kebijakan baca untuk Pemilik Toko (hanya melihat pengajuan milik tokonya sendiri)
CREATE POLICY owner_read_own_deletion_requests ON public.transaction_deletion_requests
FOR SELECT TO authenticated USING (
    EXISTS (SELECT 1 FROM public.profiles p WHERE p.id = (SELECT auth.uid()) AND p.role = 'pemilik_toko' AND p.toko_id = transaction_deletion_requests.store_id AND p.status = 'aktif')
);

-- Fungsi Atomik: Pemilik Toko mengajukan penghapusan seluruh transaksi tokonya
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

-- Fungsi Atomik: Pemilik Toko membatalkan pengajuan pending miliknya
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

-- Fungsi Atomik: Superadmin menolak pengajuan
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

-- Fungsi Atomik: Superadmin menyetujui dan mengeksekusi penghapusan seluruh transaksi toko
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

REVOKE ALL ON FUNCTION public.request_store_transaction_deletion(uuid, uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.request_store_transaction_deletion(uuid, uuid) TO service_role;

REVOKE ALL ON FUNCTION public.cancel_store_transaction_deletion(uuid, uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.cancel_store_transaction_deletion(uuid, uuid) TO service_role;

REVOKE ALL ON FUNCTION public.reject_store_transaction_deletion(uuid, uuid, text) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.reject_store_transaction_deletion(uuid, uuid, text) TO service_role;

REVOKE ALL ON FUNCTION public.execute_store_transaction_deletion(uuid, uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.execute_store_transaction_deletion(uuid, uuid) TO service_role;
