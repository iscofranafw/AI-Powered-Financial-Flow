-- Migration 005: extended category, classification metadata, RAG versioning and atomic writes.
ALTER TABLE public.kategori ADD COLUMN IF NOT EXISTS kata_kunci text[] NOT NULL DEFAULT '{}';
ALTER TABLE public.transaksi ADD COLUMN IF NOT EXISTS status_kategori text NOT NULL DEFAULT 'dikonfirmasi';
ALTER TABLE public.transaksi ADD COLUMN IF NOT EXISTS usulan jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE public.transaksi ADD COLUMN IF NOT EXISTS skor numeric(5,4);
ALTER TABLE public.transaksi ADD COLUMN IF NOT EXISTS batch_id uuid;
DO $$ BEGIN
 ALTER TABLE public.transaksi ADD CONSTRAINT status_kategori_valid
   CHECK (status_kategori IN ('otomatis','perlu_tinjauan','dikonfirmasi'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
CREATE TABLE IF NOT EXISTS public.rag_index_version (
 toko_id uuid NOT NULL REFERENCES public.toko(id) ON DELETE CASCADE,
 tipe text NOT NULL CHECK(tipe IN ('pengeluaran','pemasukan')),
 version bigint NOT NULL DEFAULT 1,
 updated_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(toko_id,tipe)
);
ALTER TABLE public.rag_index_version ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.rag_index_version FROM PUBLIC, anon, authenticated;
GRANT ALL ON public.rag_index_version TO service_role;
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
CREATE TRIGGER kategori_rag_bump AFTER INSERT OR UPDATE OR DELETE ON public.kategori
FOR EACH ROW EXECUTE FUNCTION public.bump_rag_index_version();
DROP TRIGGER IF EXISTS transaksi_rag_bump ON public.transaksi;
CREATE TRIGGER transaksi_rag_bump AFTER INSERT OR UPDATE OR DELETE ON public.transaksi
FOR EACH ROW EXECUTE FUNCTION public.bump_rag_index_version();

-- Atomic multi-row approval; only callable by server service role, checks role and shop.
CREATE OR REPLACE FUNCTION public.approve_transactions_atomic(
  p_actor uuid,p_shop uuid,p_ids uuid[],p_decision text
) RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path='' AS $$
DECLARE n integer;
BEGIN
 IF NOT EXISTS(SELECT 1 FROM public.profiles
   WHERE id=p_actor AND toko_id=p_shop AND role='pemilik_toko' AND status='aktif')
   THEN RAISE EXCEPTION 'Tidak diizinkan'; END IF;
 IF p_decision NOT IN ('disetujui','ditolak') OR cardinality(p_ids)=0
   OR cardinality(p_ids)<> (SELECT count(DISTINCT x) FROM unnest(p_ids) AS u(x))
   THEN RAISE EXCEPTION 'Keputusan/ID tidak valid'; END IF;
 PERFORM 1 FROM public.transaksi WHERE toko_id=p_shop AND id=ANY(p_ids) FOR UPDATE;
 IF (SELECT count(*) FROM public.transaksi WHERE toko_id=p_shop AND id=ANY(p_ids) AND status_persetujuan='pending')
      <> cardinality(p_ids) THEN RAISE EXCEPTION 'Sebagian transaksi bukan pending / bukan milik toko'; END IF;
 UPDATE public.transaksi SET status_persetujuan=p_decision,
   disetujui_oleh=p_actor,disetujui_pada=now(),updated_at=now(),
   status_kategori=CASE WHEN p_decision='disetujui' AND status_kategori='perlu_tinjauan'
                         THEN 'dikonfirmasi' ELSE status_kategori END
 WHERE toko_id=p_shop AND id=ANY(p_ids);
 GET DIAGNOSTICS n=ROW_COUNT;
 RETURN n;
END $$;
REVOKE ALL ON FUNCTION public.approve_transactions_atomic(uuid,uuid,uuid[],text) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.approve_transactions_atomic(uuid,uuid,uuid[],text) TO service_role;

-- Atomic category merge: entire reassignment + deletion runs in one transaction.
CREATE OR REPLACE FUNCTION public.merge_category_atomic(
  p_actor uuid,p_shop uuid,p_source uuid,p_target uuid
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
REVOKE ALL ON FUNCTION public.merge_category_atomic(uuid,uuid,uuid,uuid) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.merge_category_atomic(uuid,uuid,uuid,uuid) TO service_role;

-- Atomic batch import of previewed rows, all-or-nothing.
CREATE OR REPLACE FUNCTION public.import_transactions_atomic(
 p_actor uuid,p_shop uuid,p_rows jsonb,p_batch uuid
) RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path='' AS $$
DECLARE entry jsonb; n int:=0; v_role text; v_cat public.kategori%ROWTYPE; v_type text;
BEGIN
 SELECT role INTO v_role FROM public.profiles WHERE id=p_actor AND toko_id=p_shop AND status='aktif';
 IF v_role IS NULL OR v_role NOT IN ('pemilik_toko','kasir') THEN RAISE EXCEPTION 'Tidak diizinkan'; END IF;
 IF jsonb_typeof(p_rows)<>'array' OR jsonb_array_length(p_rows)=0 OR jsonb_array_length(p_rows)>1000
   THEN RAISE EXCEPTION 'Batch harus terdiri dari 1-1000 transaksi'; END IF;
 FOR entry IN SELECT value FROM jsonb_array_elements(p_rows) LOOP
   v_type := entry->>'tipe';
   SELECT * INTO v_cat FROM public.kategori WHERE id=(entry->>'kategori_id')::uuid
       AND toko_id=p_shop AND tipe=v_type;
   IF v_cat.id IS NULL THEN RAISE EXCEPTION 'Kategori tidak cocok pada baris %',n+1; END IF;
   INSERT INTO public.transaksi(toko_id,tanggal,deskripsi,nominal,tipe,kategori_id,
     dicatat_oleh,status_persetujuan,disetujui_oleh,disetujui_pada,status_kategori,skor,usulan,batch_id)
     VALUES(p_shop,(entry->>'tanggal')::date,entry->>'deskripsi',(entry->>'nominal')::numeric,
     v_type,v_cat.id,p_actor,CASE WHEN v_role='pemilik_toko' THEN 'disetujui' ELSE 'pending' END,
     CASE WHEN v_role='pemilik_toko' THEN p_actor ELSE NULL END,
     CASE WHEN v_role='pemilik_toko' THEN now() ELSE NULL END,
     coalesce(entry->>'status_kategori','dikonfirmasi'),nullif(entry->>'skor','')::numeric,
     coalesce(entry->'usulan','{}'::jsonb),p_batch);
   n := n+1;
 END LOOP;
 RETURN n;
END $$;
REVOKE ALL ON FUNCTION public.import_transactions_atomic(uuid,uuid,jsonb,uuid) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.import_transactions_atomic(uuid,uuid,jsonb,uuid) TO service_role;

-- Superadmin sees counts and latest date only; no transaction content.
DROP FUNCTION IF EXISTS public.admin_shop_summary();
CREATE OR REPLACE FUNCTION public.admin_shop_summary()
RETURNS TABLE(toko_id uuid,nama_toko text,bidang_usaha text,jumlah_pengguna bigint,
 jumlah_transaksi bigint,jumlah_kategori bigint,transaksi_terakhir date)
LANGUAGE sql SECURITY DEFINER SET search_path='' AS $$
 SELECT t.id,t.nama,t.bidang_usaha,
  (SELECT count(*) FROM public.profiles p WHERE p.toko_id=t.id)::bigint,
  (SELECT count(*) FROM public.transaksi tr WHERE tr.toko_id=t.id)::bigint,
  (SELECT count(*) FROM public.kategori k WHERE k.toko_id=t.id)::bigint,
  (SELECT max(tanggal) FROM public.transaksi tr WHERE tr.toko_id=t.id)
 FROM public.toko t;
$$;
REVOKE ALL ON FUNCTION public.admin_shop_summary() FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.admin_shop_summary() TO service_role;
