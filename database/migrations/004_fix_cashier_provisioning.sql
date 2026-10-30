-- Migration 004: repair Stage-1 cashier creation without trusting user-controlled role metadata.
-- Apply AFTER 001 & 003. Safe to repeat creation of functions; this migration is intended to run once.
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
ALTER TABLE public.cashier_provisioning ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.cashier_provisioning FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.cashier_provisioning TO service_role;

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
   -- Secret, single-use, scoped to an email and approved owner; public signups can't forge this.
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
   -- Public registration can ONLY create pending owner accounts, regardless of metadata.
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
-- Uses built-in pg_catalog.sha256 (PostgreSQL 15+), not extension schema assumptions.
