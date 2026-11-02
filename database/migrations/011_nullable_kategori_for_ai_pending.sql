-- LLM queue support. NULL category FK values are allowed by PostgreSQL's
-- existing composite FK (MATCH SIMPLE); keep that FK unchanged.
BEGIN;
ALTER TABLE public.transaksi ALTER COLUMN kategori_id DROP NOT NULL;
-- Enforce the marker for new writes, without modifying or scanning existing rows.
DO $$ BEGIN
 IF NOT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='public.transaksi'::regclass AND conname='transaksi_ai_pending_category') THEN
 ALTER TABLE public.transaksi ADD CONSTRAINT transaksi_ai_pending_category CHECK (
   kategori_id IS NOT NULL OR (coalesce(usulan->>'ai_pending','false')='true'
     AND status_kategori='perlu_tinjauan' AND status_persetujuan='pending')) NOT VALID;
 END IF;
END $$;
-- Replace the active 4-argument import RPC (005 was superseded by 007).
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
REVOKE ALL ON FUNCTION public.import_transactions_atomic(uuid,uuid,jsonb,uuid) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.import_transactions_atomic(uuid,uuid,jsonb,uuid) TO service_role;


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
REVOKE ALL ON FUNCTION public.mutate_finance_audited(uuid,uuid,text,uuid[],jsonb) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.mutate_finance_audited(uuid,uuid,text,uuid[],jsonb) TO service_role;

-- Preserve the existing RPC contract for other callers, with the same audit guarantee.
CREATE OR REPLACE FUNCTION public.approve_transactions_atomic(p_actor uuid,p_shop uuid,p_ids uuid[],p_decision text)
RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path='' AS $$
BEGIN
 RETURN public.mutate_finance_audited(p_actor,p_shop,'approve_transactions',p_ids,jsonb_build_object('decision',p_decision))::integer;
END $$;
REVOKE ALL ON FUNCTION public.approve_transactions_atomic(uuid,uuid,uuid[],text) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.approve_transactions_atomic(uuid,uuid,uuid[],text) TO service_role;
-- Retry is a compare-and-set mutation: never touch approved, manually edited,
-- already retried, or another store's rows. Audit and marker removal are atomic.
CREATE OR REPLACE FUNCTION public.retry_pending_ai_classification(
 p_actor uuid,p_shop uuid,p_id uuid,p_category uuid,p_updated_at timestamptz
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
REVOKE ALL ON FUNCTION public.retry_pending_ai_classification(uuid,uuid,uuid,uuid,timestamptz) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.retry_pending_ai_classification(uuid,uuid,uuid,uuid,timestamptz) TO service_role;
COMMIT;
