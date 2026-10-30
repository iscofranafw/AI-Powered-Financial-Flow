-- AFTER registering your own account in the app, run in SQL Editor.
-- Replace the email. Verify it is YOUR account before running.
-- The initial signup creates an unused shop; this script deletes it.
do $$
declare v_id uuid; v_shop uuid;
begin
 select id,toko_id into v_id,v_shop from public.profiles
 where email='REPLACE_WITH_YOUR_ADMIN_EMAIL' and role='pemilik_toko';
 if v_id is null then raise exception 'Admin candidate not found'; end if;
 update public.profiles set role='superadmin',toko_id=null,status='aktif' where id=v_id;
 delete from public.toko where id=v_shop;
 update auth.users set email_confirmed_at=coalesce(email_confirmed_at,now()) where id=v_id;
end $$;
