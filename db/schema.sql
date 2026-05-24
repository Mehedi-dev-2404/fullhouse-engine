-- ============================================================
-- Fullhouse Hackathon — Postgres Schema
-- Run this in Supabase SQL editor or via psql.
-- Uses Supabase Auth for user accounts (auth.users table exists already).
-- ============================================================

-- Extensions
create extension if not exists "pgcrypto";

-- ============================================================
-- 1. USERS (extends Supabase auth.users)
-- ============================================================

create table public.users (
  id            uuid primary key references auth.users(id) on delete cascade,
  email         text not null,
  display_name  text not null,
  avatar_key    text not null default 'robot_1',   -- portal avatar selection
  hat_key       text not null default 'none',       -- cosmetic hat
  created_at    timestamptz not null default now()
);

-- Auto-create profile on signup
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer as $$
begin
  insert into public.users (id, email, display_name)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data->>'display_name', split_part(new.email, '@', 1))
  );
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- ============================================================
-- 2. BOTS
-- ============================================================

create type bot_status as enum ('pending', 'validating', 'ready', 'error', 'disqualified');

create table public.bots (
  id            uuid primary key default gen_random_uuid(),
  user_id       uuid not null references public.users(id) on delete cascade,
  bot_name      text not null,
  storage_path  text not null,         -- S3/Supabase Storage path to bot.py
  version       int  not null default 1,
  status        bot_status not null default 'pending',
  error_message text,                  -- populated if status = 'error'
  submitted_at  timestamptz not null default now(),

  constraint bots_name_len check (char_length(bot_name) between 2 and 32)
);

-- One active bot per user per tournament phase
create unique index bots_user_active_idx on public.bots (user_id)
  where status = 'ready';

-- ============================================================
-- 3. TOURNAMENTS
-- ============================================================

create type tournament_phase as enum (
  'registration', 'day1', 'patch_window', 'day2', 'finale', 'complete'
);

create table public.tournaments (
  id            uuid primary key default gen_random_uuid(),
  name          text not null,
  phase         tournament_phase not null default 'registration',
  current_round int  not null default 0,
  total_rounds  int  not null default 3,   -- swiss rounds before finale
  n_finalists   int  not null default 64,
  starts_at     timestamptz not null,
  created_at    timestamptz not null default now()
);

-- ============================================================
-- 4. MATCHES
-- ============================================================

create type match_status as enum ('queued', 'running', 'complete', 'failed');

create table public.matches (
  id            uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id),
  round         int  not null,
  table_index   int  not null,
  status        match_status not null default 'queued',
  n_hands       int  not null default 400,
  started_at    timestamptz,
  completed_at  timestamptz,
  error_message text,

  constraint matches_round_table unique (tournament_id, round, table_index)
);

create index matches_tournament_idx on public.matches (tournament_id, round, status);

-- ============================================================
-- 5. MATCH_BOTS (who played at each table, results)
-- ============================================================

create table public.match_bots (
  match_id    uuid not null references public.matches(id) on delete cascade,
  bot_id      uuid not null references public.bots(id),
  seat        int  not null,           -- 0-5
  final_stack int,                     -- null until match complete
  chip_delta  int,                     -- final_stack - starting_stack

  primary key (match_id, bot_id)
);

create index match_bots_bot_idx on public.match_bots (bot_id);

-- ============================================================
-- 6. HANDS (full hand history for replay)
-- ============================================================

create table public.hands (
  id            uuid primary key default gen_random_uuid(),
  match_id      uuid not null references public.matches(id) on delete cascade,
  hand_num      int  not null,
  street        text not null,          -- street where hand ended
  pot           int  not null,
  community_cards text[] not null default '{}',
  action_log    jsonb not null default '[]',
  revealed_cards jsonb not null default '{}',  -- {bot_id: [card, card]} at showdown
  played_at     timestamptz not null default now(),

  constraint hands_match_num unique (match_id, hand_num)
);

-- Partial index — only index hands from completed matches for replay queries
create index hands_match_idx on public.hands (match_id, hand_num);

-- ============================================================
-- 7. HAND_WINNERS
-- ============================================================

create table public.hand_winners (
  hand_id   uuid not null references public.hands(id) on delete cascade,
  bot_id    uuid not null references public.bots(id),
  amount    int  not null,

  primary key (hand_id, bot_id)
);

-- ============================================================
-- 8. LEADERBOARD (materialised, updated after each match)
-- ============================================================

create table public.leaderboard (
  tournament_id    uuid not null references public.tournaments(id) on delete cascade,
  bot_id           uuid not null references public.bots(id) on delete cascade,
  rank             int  not null default 0,
  cumulative_delta int  not null default 0,
  matches_played   int  not null default 0,
  updated_at       timestamptz not null default now(),

  primary key (tournament_id, bot_id)
);

create index leaderboard_rank_idx on public.leaderboard (tournament_id, rank);

-- ============================================================
-- 9. LEADERBOARD UPDATE FUNCTION
-- Called by the worker API after each match completes.
-- ============================================================

create or replace function public.record_match_result(
  p_match_id    uuid,
  p_tournament_id uuid,
  p_results     jsonb   -- [{bot_id, chip_delta}]
)
returns void language plpgsql as $$
declare
  r jsonb;
begin
  -- 1. Mark match complete
  update public.matches
  set status = 'complete', completed_at = now()
  where id = p_match_id;

  -- 2. Write per-bot results into match_bots
  for r in select * from jsonb_array_elements(p_results) loop
    update public.match_bots
    set
      chip_delta  = (r->>'chip_delta')::int,
      final_stack = 10000 + (r->>'chip_delta')::int
    where match_id = p_match_id
      and bot_id   = (r->>'bot_id')::uuid;
  end loop;

  -- 3. Upsert leaderboard
  for r in select * from jsonb_array_elements(p_results) loop
    insert into public.leaderboard (tournament_id, bot_id, cumulative_delta, matches_played, updated_at)
    values (
      p_tournament_id,
      (r->>'bot_id')::uuid,
      (r->>'chip_delta')::int,
      1,
      now()
    )
    on conflict (tournament_id, bot_id) do update set
      cumulative_delta = leaderboard.cumulative_delta + (r->>'chip_delta')::int,
      matches_played   = leaderboard.matches_played + 1,
      updated_at       = now();
  end loop;

  -- 4. Recompute ranks (dense_rank over cumulative_delta desc)
  update public.leaderboard l
  set rank = sub.new_rank
  from (
    select bot_id,
           dense_rank() over (order by cumulative_delta desc) as new_rank
    from public.leaderboard
    where tournament_id = p_tournament_id
  ) sub
  where l.tournament_id = p_tournament_id
    and l.bot_id = sub.bot_id;
end;
$$;

-- ============================================================
-- 10. ROW LEVEL SECURITY
-- ============================================================

alter table public.users        enable row level security;
alter table public.bots         enable row level security;
alter table public.tournaments  enable row level security;
alter table public.matches      enable row level security;
alter table public.match_bots   enable row level security;
alter table public.hands        enable row level security;
alter table public.hand_winners enable row level security;
alter table public.leaderboard  enable row level security;

-- Users: can only see/edit their own profile
create policy "users_self" on public.users
  using (id = auth.uid());

-- Bots: owner can manage, everyone can read ready bots
create policy "bots_owner_manage" on public.bots
  using (user_id = auth.uid())
  with check (user_id = auth.uid());

create policy "bots_public_read" on public.bots
  for select using (status = 'ready');

-- Tournaments, matches, leaderboard: public read
create policy "tournaments_read" on public.tournaments
  for select using (true);

create policy "matches_read" on public.matches
  for select using (true);

create policy "match_bots_read" on public.match_bots
  for select using (true);

create policy "hands_read" on public.hands
  for select using (true);

create policy "hand_winners_read" on public.hand_winners
  for select using (true);

create policy "leaderboard_read" on public.leaderboard
  for select using (true);

-- ============================================================
-- 11. REALTIME (enable for leaderboard + match status)
-- ============================================================

alter publication supabase_realtime add table public.leaderboard;
alter publication supabase_realtime add table public.matches;
