BEGIN;

-- No FK: this retry record must survive the deletion of its shop and profiles.
CREATE TABLE IF NOT EXISTS public.shop_deletion_jobs (
    owner_id uuid PRIMARY KEY,
    shop_id uuid NOT NULL UNIQUE,
    owner_name text NOT NULL,
    email text NOT NULL,
    shop_name text NOT NULL,
    auth_user_ids uuid[] NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
REVOKE ALL ON TABLE public.shop_deletion_jobs FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.shop_deletion_jobs TO service_role;

CREATE OR REPLACE FUNCTION public.delete_inactive_shop_atomic(
    p_actor uuid, p_owner uuid, p_shop uuid, p_email text
) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
DECLARE
    actor public.profiles%ROWTYPE;
    owner_row public.profiles%ROWTYPE;
    shop public.toko%ROWTYPE;
    job public.shop_deletion_jobs%ROWTYPE;
    user_ids uuid[];
BEGIN
    SELECT * INTO actor FROM public.profiles WHERE id=p_actor AND status='aktif' FOR SHARE;
    IF actor.id IS NULL OR actor.role <> 'superadmin' OR p_actor=p_owner THEN
        RAISE EXCEPTION 'Hanya Superadmin dapat menghapus toko pemilik lain';
    END IF;
    SELECT * INTO owner_row FROM public.profiles WHERE id=p_owner FOR UPDATE;
    SELECT * INTO job FROM public.shop_deletion_jobs WHERE owner_id=p_owner FOR UPDATE;
    IF job.owner_id IS NOT NULL THEN
        IF job.shop_id IS DISTINCT FROM p_shop OR job.email IS DISTINCT FROM p_email THEN
            RAISE EXCEPTION 'Identitas penghapusan tidak cocok';
        END IF;
        RETURN to_jsonb(job);
    END IF;
    IF owner_row.id IS NULL OR owner_row.role <> 'pemilik_toko' OR owner_row.status <> 'nonaktif'
       OR owner_row.toko_id IS DISTINCT FROM p_shop OR owner_row.email IS DISTINCT FROM p_email THEN
        RAISE EXCEPTION 'Hanya pemilik Nonaktif dengan identitas toko yang sesuai dapat dihapus';
    END IF;
    SELECT * INTO shop FROM public.toko WHERE id=p_shop FOR UPDATE;
    IF shop.id IS NULL THEN RAISE EXCEPTION 'Toko tidak ditemukan'; END IF;
    PERFORM 1 FROM public.profiles WHERE toko_id=p_shop ORDER BY id FOR UPDATE;
    IF EXISTS (SELECT 1 FROM public.profiles WHERE toko_id=p_shop AND role='pemilik_toko' AND id<>p_owner) THEN
        RAISE EXCEPTION 'Toko memiliki pemilik lain dan tidak dapat dihapus';
    END IF;
    SELECT array_agg(id ORDER BY id) INTO user_ids FROM public.profiles WHERE toko_id=p_shop;
    INSERT INTO public.shop_deletion_jobs(owner_id,shop_id,owner_name,email,shop_name,auth_user_ids)
    VALUES(p_owner,p_shop,owner_row.nama,owner_row.email,shop.nama,user_ids) RETURNING * INTO job;

    DELETE FROM public.cashier_provisioning WHERE toko_id=p_shop;
    DELETE FROM public.transaction_deletion_requests WHERE store_id=p_shop;
    DELETE FROM public.transaksi WHERE toko_id=p_shop;
    DELETE FROM public.kategori WHERE toko_id=p_shop;
    DELETE FROM public.rag_index_version WHERE toko_id=p_shop;
    DELETE FROM public.import_transaction_identities WHERE toko_id=p_shop;
    DELETE FROM public.profiles WHERE toko_id=p_shop;
    DELETE FROM public.toko WHERE id=p_shop;
    -- Existing append-only administrative audit history is deliberately retained.
    RETURN to_jsonb(job);
END $$;
REVOKE ALL ON FUNCTION public.delete_inactive_shop_atomic(uuid,uuid,uuid,text) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.delete_inactive_shop_atomic(uuid,uuid,uuid,text) TO service_role;

COMMIT;
