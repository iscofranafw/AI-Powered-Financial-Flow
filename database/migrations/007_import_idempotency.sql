-- P0-03: durable import identities, deliberately separate from transaction values.
-- Apply after 006 and before deploying the service changes. Existing transactions
-- without source IDs have no recoverable import identity; they are not backfilled.
-- Direct RPC callers without row metadata must reuse p_batch on retry. Services
-- supply a dataset key plus fingerprint occurrence, stable across chunk sizes.
-- Ledger entries survive transaction deletion so retries cannot resurrect imports.
BEGIN;
CREATE TABLE IF NOT EXISTS public.import_transaction_identities (
 toko_id uuid NOT NULL,
 identity_key text NOT NULL,
 PRIMARY KEY(toko_id, identity_key)
);
ALTER TABLE public.import_transaction_identities ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.import_transaction_identities FROM PUBLIC, anon, authenticated;
CREATE INDEX IF NOT EXISTS transaksi_import_source_lookup
 ON public.transaksi(toko_id, lower(btrim(usulan->>'source_id')));

-- Drop legacy 3-argument overload from 005_ai_forecast.sql if present,
-- ensuring only the protected 4-argument version remains.
DROP FUNCTION IF EXISTS public.import_transactions_atomic(uuid, uuid, jsonb);

CREATE OR REPLACE FUNCTION public.import_transactions_atomic(
 p_actor uuid,p_shop uuid,p_rows jsonb,p_batch uuid
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
   SELECT * INTO v_cat FROM public.kategori WHERE id=(entry->>'kategori_id')::uuid
       AND toko_id=p_shop AND tipe=v_type;
   IF v_cat.id IS NULL THEN RAISE EXCEPTION 'Kategori tidak cocok pada baris %',n+1; END IF;
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
     v_type,v_cat.id,p_actor,CASE WHEN v_role='pemilik_toko' THEN 'disetujui' ELSE 'pending' END,
     CASE WHEN v_role='pemilik_toko' THEN p_actor ELSE NULL END,
     CASE WHEN v_role='pemilik_toko' THEN now() ELSE NULL END,
     coalesce(entry->>'status_kategori','dikonfirmasi'),nullif(entry->>'skor','')::numeric,
     v_meta,p_batch);
   n := n+1;
 END LOOP;
 RETURN n;
END $$;
REVOKE ALL ON FUNCTION public.import_transactions_atomic(uuid,uuid,jsonb,uuid) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.import_transactions_atomic(uuid,uuid,jsonb,uuid) TO service_role;


COMMIT;
