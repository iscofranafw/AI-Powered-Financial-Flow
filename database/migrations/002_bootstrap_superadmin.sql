-- Jalankan di SQL Editor SETELAH mendaftarkan akun Anda sendiri di aplikasi.
-- Ganti alamat email berikut. Pastikan akun tersebut adalah akun Anda sebelum menjalankan skrip.
-- Pendaftaran awal otomatis membuat toko yang tidak terpakai; skrip ini akan menghapusnya.
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
