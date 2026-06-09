-- Run this in Supabase SQL Editor.
--
-- The application talks to Supabase using the SERVICE ROLE key (server-side
-- only), which bypasses RLS by design. The anon key is NOT used by the
-- application. Therefore the table itself does not need public RLS policies.
--
-- If you ever decide to expose the table to a client (Streamlit, browser,
-- mobile app), DO NOT add a permissive policy. Instead:
--   1. Issue a per-user JWT (Supabase Auth)
--   2. Add a strict policy keyed on auth.uid() = users.id or similar
--   3. Keep the service-role key on the server only

CREATE TABLE IF NOT EXISTS users (
  id BIGSERIAL PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  plan TEXT DEFAULT 'free',
  analyses_used INT DEFAULT 0,
  analyses_limit INT DEFAULT 3,
  created_at TIMESTPTZ DEFAULT NOW()
);

-- RLS is left DISABLED on purpose. The app authenticates to Supabase with
-- the service-role key, so RLS would not be enforced anyway. Leaving it
-- disabled but the key server-side is the simplest correct setup.
--
-- DO NOT enable RLS with a permissive policy like "FOR ALL USING (true)".
-- That is equivalent to no RLS but adds the false impression of safety,
-- and a future client that mistakenly uses the anon key would still get
-- full read/write.
--
-- If you need to enable RLS in the future, drop the "Allow all" policy
-- and replace with strict per-user policies (see comment block above).
ALTER TABLE users DISABLE ROW LEVEL SECURITY;
