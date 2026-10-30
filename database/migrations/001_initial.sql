-- Run once in Supabase SQL Editor. WARNING: new project recommended.
create extension if not exists pgcrypto;
create table public.toko (
 id uuid primary key default gen_random_uuid(),
 nama text not null check(length(trim(nama))>0),
 bidang_usaha text not null check(length(trim(bidang_usaha))>0),
 created_at timestamptz not null default now()
);
create table public.profiles (
 id uuid primary key references auth.users(id) on delete cascade,
 email text not null,
 nama text not null,
 role text not null check(role in ('superadmin','pemilik_toko','kasir')),
 toko_id uuid references public.toko(id),
 status text not null default 'pending' check(status in ('pending','aktif','nonaktif','ditolak')),
 created_at timestamptz not null default now(),
 constraint shop_role check ((role='superadmin' and toko_id is null) or (role<>'superadmin' and toko_id is not null))
);
create index profiles_toko_idx on public.profiles(toko_id);

-- SECURITY DEFINER trigger: app_metadata is writable only through trusted Admin API.
create or replace function public.create_profile_on_signup()
returns trigger language plpgsql security definer set search_path = '' as $$
declare
 v_shop uuid;
 v_role text;
 v_name text;
begin
 v_role := new.raw_app_meta_data->>'bizflow_role';
 v_name := coalesce(nullif(trim(new.raw_user_meta_data->>'nama'),''),'Pengguna');
 if v_role = 'kasir' then
   v_shop := (new.raw_app_meta_data->>'bizflow_toko_id')::uuid;
   if not exists(select 1 from public.profiles where toko_id=v_shop and role='pemilik_toko' and status='aktif') then
     raise exception 'Toko kasir tidak valid atau belum aktif';
   end if;
   insert into public.profiles(id,email,nama,role,toko_id,status)
   values(new.id,new.email,v_name,'kasir',v_shop,'aktif');
 else
   -- Public signup is ALWAYS owner, regardless of user_metadata.
   if coalesce(trim(new.raw_user_meta_data->>'nama_toko'),'')='' or coalesce(trim(new.raw_user_meta_data->>'bidang_usaha'),'')='' then
     raise exception 'Nama toko dan bidang usaha wajib diisi';
   end if;
   insert into public.toko(nama,bidang_usaha)
   values(new.raw_user_meta_data->>'nama_toko',new.raw_user_meta_data->>'bidang_usaha') returning id into v_shop;
   insert into public.profiles(id,email,nama,role,toko_id,status)
   values(new.id,new.email,v_name,'pemilik_toko',v_shop,'pending');
 end if;
 return new;
end; $$;
create trigger on_auth_user_created after insert on auth.users
for each row execute function public.create_profile_on_signup();

alter table public.toko enable row level security;
alter table public.profiles enable row level security;
-- Only own active profile visible to client. Server uses service role for administrative reads.
create policy profile_self on public.profiles for select to authenticated
 using(id=(select auth.uid()) and status='aktif');
create policy shop_active_member on public.toko for select to authenticated
 using(exists(select 1 from public.profiles p where p.id=(select auth.uid()) and p.toko_id=toko.id and p.status='aktif'));
-- No direct client insert/update/delete grants through RLS. Server operations validate role.

-- Server-only aggregate function: never grant execute to anon/authenticated.
create or replace function public.admin_shop_summary()
returns table(toko_id uuid, nama_toko text, bidang_usaha text, jumlah_pengguna bigint)
language sql security definer set search_path = '' as $$
 select t.id,t.nama,t.bidang_usaha,count(p.id)::bigint
 from public.toko t left join public.profiles p on p.toko_id=t.id
 group by t.id,t.nama,t.bidang_usaha;
$$;
revoke all on function public.admin_shop_summary() from public, anon, authenticated;
grant execute on function public.admin_shop_summary() to service_role;
