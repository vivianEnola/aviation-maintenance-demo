create extension if not exists pgcrypto;

insert into storage.buckets (id, name, public)
values ('mmsstv-images', 'mmsstv-images', false)
on conflict (id) do update set public = excluded.public;

create table if not exists public.image_queue (
    id uuid primary key default gen_random_uuid(),
    device_id text not null,
    storage_path text not null unique,
    original_name text not null,
    sha256 text not null,
    status text not null default 'pending'
        check (status in ('pending', 'processed', 'failed')),
    result jsonb,
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (device_id, sha256)
);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists image_queue_set_updated_at on public.image_queue;
create trigger image_queue_set_updated_at
before update on public.image_queue
for each row execute function public.set_updated_at();

alter table public.image_queue enable row level security;

-- 本项目的 Streamlit 服务端与本地上传器使用 service_role key。
-- service_role 会绕过 RLS；不要把该密钥提交到 Git 或发送到浏览器代码。

