-- NorthStar production memory layer.
-- Apply in Supabase SQL Editor or with:
-- supabase db push --include-all

create extension if not exists pgcrypto;
create extension if not exists vector;

create type public.source_kind as enum ('cheat_sheet', 'notes', 'outline', 'image_reference', 'youtube', 'web', 'prompt_pack', 'other');
create type public.asset_kind as enum ('video', 'image', 'audio', 'manim_code', 'caption', 'share_package', 'blender_model', 'latex');
create type public.run_status as enum ('draft', 'planned', 'coding', 'rendering', 'complete', 'failed', 'archived');
create type public.publish_platform as enum ('youtube', 'twitch', 'x', 'download', 'embed');

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  display_name text,
  plan_key text not null default 'free',
  trial_video_credits integer not null default 3,
  paid_video_credits integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.projects (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.profiles(id) on delete cascade,
  title text not null,
  description text,
  default_aspect_ratio text not null default '9:16',
  default_model text not null default 'gpt-5',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.source_documents (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.profiles(id) on delete cascade,
  project_id uuid references public.projects(id) on delete set null,
  kind public.source_kind not null default 'other',
  title text not null,
  original_filename text,
  storage_path text,
  mime_type text,
  bytes integer,
  summary text,
  extracted_text text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table public.source_chunks (
  id uuid primary key default gen_random_uuid(),
  source_id uuid not null references public.source_documents(id) on delete cascade,
  owner_id uuid not null references public.profiles(id) on delete cascade,
  chunk_index integer not null,
  content text not null,
  token_count integer,
  embedding vector(1536),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (source_id, chunk_index)
);

create table public.memories (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.profiles(id) on delete cascade,
  project_id uuid references public.projects(id) on delete cascade,
  title text not null,
  content text not null,
  tags text[] not null default '{}',
  category text not null default 'general',
  source_id uuid references public.source_documents(id) on delete set null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.animation_runs (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.profiles(id) on delete cascade,
  project_id uuid references public.projects(id) on delete set null,
  local_job_id text,
  title text not null default 'Untitled animation',
  prompt text,
  plan jsonb,
  status public.run_status not null default 'draft',
  aspect_ratio text not null default '9:16',
  duration_seconds numeric(8, 2),
  text_model text,
  image_model text,
  thumbnail_path text,
  video_path text,
  share_slug text unique,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.assets (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.profiles(id) on delete cascade,
  project_id uuid references public.projects(id) on delete set null,
  run_id uuid references public.animation_runs(id) on delete cascade,
  kind public.asset_kind not null,
  title text,
  storage_path text not null,
  mime_type text,
  bytes integer,
  width integer,
  height integer,
  duration_seconds numeric(8, 2),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table public.share_links (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.profiles(id) on delete cascade,
  run_id uuid not null references public.animation_runs(id) on delete cascade,
  slug text not null unique,
  title text not null,
  description text,
  social_copy text,
  is_public boolean not null default false,
  view_count bigint not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.publish_events (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.profiles(id) on delete cascade,
  run_id uuid not null references public.animation_runs(id) on delete cascade,
  platform public.publish_platform not null,
  status text not null default 'queued',
  external_url text,
  error text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.user_provider_configs (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.profiles(id) on delete cascade,
  provider text not null,
  label text not null,
  default_model text,
  vault_secret_id text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (owner_id, provider, label)
);

create or replace function public.touch_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger profiles_touch_updated_at
  before update on public.profiles
  for each row execute function public.touch_updated_at();
create trigger projects_touch_updated_at
  before update on public.projects
  for each row execute function public.touch_updated_at();
create trigger memories_touch_updated_at
  before update on public.memories
  for each row execute function public.touch_updated_at();
create trigger animation_runs_touch_updated_at
  before update on public.animation_runs
  for each row execute function public.touch_updated_at();
create trigger share_links_touch_updated_at
  before update on public.share_links
  for each row execute function public.touch_updated_at();
create trigger publish_events_touch_updated_at
  before update on public.publish_events
  for each row execute function public.touch_updated_at();
create trigger user_provider_configs_touch_updated_at
  before update on public.user_provider_configs
  for each row execute function public.touch_updated_at();

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, email, display_name)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'name', split_part(new.email, '@', 1))
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

create index projects_owner_updated_idx on public.projects(owner_id, updated_at desc);
create index source_documents_owner_kind_idx on public.source_documents(owner_id, kind, created_at desc);
create index source_chunks_owner_source_idx on public.source_chunks(owner_id, source_id, chunk_index);
create index memories_owner_category_idx on public.memories(owner_id, category, updated_at desc);
create index memories_tags_idx on public.memories using gin(tags);
create index animation_runs_owner_status_idx on public.animation_runs(owner_id, status, created_at desc);
create index animation_runs_owner_local_job_idx on public.animation_runs(owner_id, local_job_id);
create index assets_owner_run_idx on public.assets(owner_id, run_id, kind);
create index share_links_owner_run_idx on public.share_links(owner_id, run_id);
create index publish_events_owner_run_idx on public.publish_events(owner_id, run_id, created_at desc);

alter table public.profiles enable row level security;
alter table public.projects enable row level security;
alter table public.source_documents enable row level security;
alter table public.source_chunks enable row level security;
alter table public.memories enable row level security;
alter table public.animation_runs enable row level security;
alter table public.assets enable row level security;
alter table public.share_links enable row level security;
alter table public.publish_events enable row level security;
alter table public.user_provider_configs enable row level security;

create policy "profiles are self-readable" on public.profiles
  for select using (auth.uid() = id);
create policy "profiles are self-updatable" on public.profiles
  for update using (auth.uid() = id) with check (auth.uid() = id);

create policy "owners manage projects" on public.projects
  for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy "owners manage source documents" on public.source_documents
  for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy "owners manage source chunks" on public.source_chunks
  for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy "owners manage memories" on public.memories
  for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy "owners manage animation runs" on public.animation_runs
  for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy "owners manage assets" on public.assets
  for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy "owners manage share links" on public.share_links
  for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy "public share links are readable" on public.share_links
  for select using (is_public = true or auth.uid() = owner_id);
create policy "owners manage publish events" on public.publish_events
  for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);
create policy "owners manage provider configs" on public.user_provider_configs
  for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values
  ('northstar-sources', 'northstar-sources', false, 52428800, null),
  ('northstar-assets', 'northstar-assets', false, 104857600, null),
  ('northstar-public-renders', 'northstar-public-renders', true, 524288000, null)
on conflict (id) do nothing;

create policy "owners read source storage" on storage.objects
  for select using (bucket_id = 'northstar-sources' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "owners insert source storage" on storage.objects
  for insert with check (bucket_id = 'northstar-sources' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "owners update source storage" on storage.objects
  for update using (bucket_id = 'northstar-sources' and (storage.foldername(name))[1] = auth.uid()::text)
  with check (bucket_id = 'northstar-sources' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "owners delete source storage" on storage.objects
  for delete using (bucket_id = 'northstar-sources' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "owners read asset storage" on storage.objects
  for select using (bucket_id = 'northstar-assets' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "owners insert asset storage" on storage.objects
  for insert with check (bucket_id = 'northstar-assets' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "owners update asset storage" on storage.objects
  for update using (bucket_id = 'northstar-assets' and (storage.foldername(name))[1] = auth.uid()::text)
  with check (bucket_id = 'northstar-assets' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "owners delete asset storage" on storage.objects
  for delete using (bucket_id = 'northstar-assets' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "public reads public renders" on storage.objects
  for select using (bucket_id = 'northstar-public-renders');
create policy "owners insert public renders" on storage.objects
  for insert with check (bucket_id = 'northstar-public-renders' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "owners update public renders" on storage.objects
  for update using (bucket_id = 'northstar-public-renders' and (storage.foldername(name))[1] = auth.uid()::text)
  with check (bucket_id = 'northstar-public-renders' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "owners delete public renders" on storage.objects
  for delete using (bucket_id = 'northstar-public-renders' and (storage.foldername(name))[1] = auth.uid()::text);
