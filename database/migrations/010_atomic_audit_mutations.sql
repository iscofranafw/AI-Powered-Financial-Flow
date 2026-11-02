BEGIN;
-- All snapshots are taken under row locks, not supplied by the Python caller.
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
       IF old_row.status_persetujuan <> 'pending' THEN RAISE EXCEPTION 'Transaksi bukan pending'; END IF;
       UPDATE public.transaksi SET status_persetujuan=p_values->>'decision',
         disetujui_oleh=p_actor,disetujui_pada=now(),updated_at=now(),
         status_kategori=CASE WHEN p_values->>'decision'='disetujui' AND status_kategori='perlu_tinjauan' THEN 'dikonfirmasi' ELSE status_kategori END
       WHERE id=old_row.id RETURNING * INTO new_row;
     ELSE
       IF actor.role='kasir' AND (old_row.dicatat_oleh<>p_actor OR old_row.status_persetujuan NOT IN ('pending','ditolak'))
       THEN RAISE EXCEPTION 'Kasir tidak dapat mengubah transaksi ini'; END IF;
       IF NOT EXISTS(SELECT 1 FROM public.kategori WHERE id=(p_values->>'kategori_id')::uuid AND toko_id=p_shop AND tipe=p_values->>'tipe')
       THEN RAISE EXCEPTION 'Kategori tidak sesuai jenis transaksi/toko'; END IF;
       IF coalesce(length(btrim(p_values->>'deskripsi')),0) NOT BETWEEN 1 AND 500
          OR (p_values->>'tanggal') IS NULL OR (p_values->>'nominal') IS NULL
          OR (p_values->>'nominal')::numeric <= 0 THEN RAISE EXCEPTION 'Data transaksi tidak valid'; END IF;
       changed := ROW(old_row.tanggal,old_row.deskripsi,old_row.nominal,old_row.tipe,old_row.kategori_id)
          IS DISTINCT FROM ROW((p_values->>'tanggal')::date,p_values->>'deskripsi',(p_values->>'nominal')::numeric,p_values->>'tipe',(p_values->>'kategori_id')::uuid);
       reset_approval := actor.role='kasir' OR old_row.status_persetujuan='ditolak' OR (old_row.status_persetujuan='disetujui' AND changed);
       meta := CASE WHEN jsonb_typeof(old_row.usulan)='object' THEN old_row.usulan ELSE '{}'::jsonb END;
       history := CASE WHEN jsonb_typeof(meta->'audit_history')='array' THEN meta->'audit_history' ELSE '[]'::jsonb END;
       meta := meta || jsonb_build_object('audit_history', history || jsonb_build_array(jsonb_build_object(
         'action','edit','editor_id',p_actor,'editor_role',actor.role,'timestamp',now(),
         'previous',jsonb_build_object('tanggal',old_row.tanggal,'deskripsi',old_row.deskripsi,'nominal',old_row.nominal::text,
           'tipe',old_row.tipe,'kategori_id',old_row.kategori_id,'status_persetujuan',old_row.status_persetujuan))));
       UPDATE public.transaksi SET tanggal=(p_values->>'tanggal')::date,deskripsi=p_values->>'deskripsi',
         nominal=(p_values->>'nominal')::numeric,tipe=p_values->>'tipe',kategori_id=(p_values->>'kategori_id')::uuid,
         status_kategori='dikonfirmasi',updated_at=now(),usulan=meta,
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
COMMIT;
