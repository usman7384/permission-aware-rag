CREATE TABLE IF NOT EXISTS documents (
  doc_id       UUID PRIMARY KEY,
  path         TEXT NOT NULL UNIQUE,
  folder       TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  version      INTEGER NOT NULL,
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  deleted_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS parents (
  parent_id UUID PRIMARY KEY,
  doc_id    UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  version   INTEGER NOT NULL,
  ordinal   INTEGER NOT NULL,
  heading   TEXT,
  context   TEXT,
  text      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS parents_doc_version ON parents(doc_id, version);

CREATE TABLE IF NOT EXISTS tuples (
  subject  TEXT NOT NULL,
  relation TEXT NOT NULL CHECK (relation IN ('member', 'viewer', 'parent')),
  object   TEXT NOT NULL,
  PRIMARY KEY (subject, relation, object)
);
CREATE INDEX IF NOT EXISTS tuples_object_relation ON tuples(object, relation);

CREATE TABLE IF NOT EXISTS api_keys (
  key_hash TEXT PRIMARY KEY,          -- sha256 hex of the key
  user_id  TEXT NOT NULL,             -- e.g. 'user:ana'
  is_admin BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS turns (
  turn_id    BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL,
  user_id    TEXT NOT NULL,
  role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  text       TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, turn_id);

CREATE TABLE IF NOT EXISTS requests (
  request_id UUID PRIMARY KEY,
  user_id    TEXT NOT NULL,
  session_id TEXT NOT NULL,
  question   TEXT NOT NULL,
  rewritten  TEXT,
  cache_tier TEXT NOT NULL CHECK (cache_tier IN ('none', 'exact', 'semantic')),
  abstained  BOOLEAN NOT NULL DEFAULT false,
  degraded   TEXT[] NOT NULL DEFAULT '{}',
  doc_ids    UUID[] NOT NULL DEFAULT '{}',
  timings    JSONB NOT NULL,
  answer     TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
  request_id UUID NOT NULL REFERENCES requests(request_id),
  rating     SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
  comment    TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_runs (
  run_id     UUID PRIMARY KEY,
  name       TEXT NOT NULL,
  config     JSONB NOT NULL,
  metrics    JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
