

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;


CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT NOT NULL UNIQUE,
    display_name  TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE roles (
    id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name  TEXT NOT NULL UNIQUE          -- 'engineer', 'hr_partner', 'finance_analyst', 'legal_counsel', 'exec'
);

CREATE TABLE user_roles (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, role_id)
);

CREATE TABLE role_department_access (
    role_id     UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    department  TEXT NOT NULL,
    PRIMARY KEY (role_id, department)
);

-- ------------------------------------------------------------
-- 2. DOCUMENTS & CHUNKS
--    (chunks no longer carry embeddings — see section 3)
-- ------------------------------------------------------------

CREATE TABLE documents (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source            TEXT NOT NULL DEFAULT 'gitlab-handbook',
    source_path       TEXT NOT NULL,          -- 'content/handbook/finance/expenses.md'
    title             TEXT,
    department        TEXT NOT NULL,          -- derived from path: 'finance'
    doc_type          TEXT,
    content_hash      TEXT NOT NULL,          -- sha256 → change detection
    raw_uri           TEXT,                   -- object-storage pointer to raw file
    source_updated_at TIMESTAMPTZ,
    indexed_at        TIMESTAMPTZ,
    status            TEXT NOT NULL DEFAULT 'active'
                      CHECK (status IN ('active', 'tombstoned')),
    UNIQUE (source, source_path)
);

CREATE INDEX idx_documents_department ON documents (department);
CREATE INDEX idx_documents_status     ON documents (status);

CREATE TABLE chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT  NOT NULL,
    content      TEXT NOT NULL,
    heading_path TEXT,                        -- 'Finance > Expenses > Travel'
    token_count  INT,
    department   TEXT NOT NULL,               -- denormalized for filtering
    status       TEXT NOT NULL DEFAULT 'active'
                 CHECK (status IN ('active', 'tombstoned')),
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX idx_chunks_tsv        ON chunks USING gin (tsv);
CREATE INDEX idx_chunks_department ON chunks (department);
CREATE INDEX idx_chunks_document   ON chunks (document_id);



CREATE TABLE embedding_models (
    model_name  TEXT PRIMARY KEY,             
    dimensions  INT  NOT NULL,
    table_name  TEXT NOT NULL,                -- 'chunk_embeddings_bge_1024'
    is_active   BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Enforce at most one active model
CREATE UNIQUE INDEX idx_one_active_model
    ON embedding_models (is_active) WHERE is_active;

CREATE TABLE chunk_embeddings_bge_1024 (
    chunk_id   UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    department TEXT NOT NULL,                 -- copied from chunk at insert
    embedding  vector(1024) NOT NULL
);


CREATE INDEX idx_emb_bge_company      ON chunk_embeddings_bge_1024 USING hnsw (embedding vector_cosine_ops) WHERE department = 'company';
CREATE INDEX idx_emb_bge_engineering  ON chunk_embeddings_bge_1024 USING hnsw (embedding vector_cosine_ops) WHERE department = 'engineering';
CREATE INDEX idx_emb_bge_security     ON chunk_embeddings_bge_1024 USING hnsw (embedding vector_cosine_ops) WHERE department = 'security';
CREATE INDEX idx_emb_bge_people_group ON chunk_embeddings_bge_1024 USING hnsw (embedding vector_cosine_ops) WHERE department = 'people-group';
CREATE INDEX idx_emb_bge_finance      ON chunk_embeddings_bge_1024 USING hnsw (embedding vector_cosine_ops) WHERE department = 'finance';
CREATE INDEX idx_emb_bge_legal        ON chunk_embeddings_bge_1024 USING hnsw (embedding vector_cosine_ops) WHERE department = 'legal';
CREATE INDEX idx_emb_bge_sales        ON chunk_embeddings_bge_1024 USING hnsw (embedding vector_cosine_ops) WHERE department = 'sales';
CREATE INDEX idx_emb_bge_marketing    ON chunk_embeddings_bge_1024 USING hnsw (embedding vector_cosine_ops) WHERE department = 'marketing';
CREATE INDEX idx_emb_bge_product      ON chunk_embeddings_bge_1024 USING hnsw (embedding vector_cosine_ops) WHERE department = 'product';

INSERT INTO embedding_models (model_name, dimensions, table_name, is_active)
VALUES ('bge-large-en-v1.5', 1024, 'chunk_embeddings_bge_1024', TRUE);

-- ------------------------------------------------------------
-- 4. INGESTION BOOKKEEPING  (unchanged from v1)
-- ------------------------------------------------------------

CREATE TABLE ingestion_runs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    git_commit   TEXT,
    status       TEXT NOT NULL DEFAULT 'running'
                 CHECK (status IN ('running', 'succeeded', 'failed')),
    stats        JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE ingestion_errors (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID NOT NULL REFERENCES ingestion_runs(id) ON DELETE CASCADE,
    source_path TEXT NOT NULL,
    error       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- 5. CONVERSATIONS & MEMORY  (unchanged from v1)
-- ------------------------------------------------------------

CREATE TABLE conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id),
    title           TEXT,
    running_summary TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_messages_conversation ON messages (conversation_id, created_at);

-- ------------------------------------------------------------
-- 6. TRACES  (unchanged from v1)
-- ------------------------------------------------------------

CREATE TABLE query_traces (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id       UUID REFERENCES messages(id) ON DELETE SET NULL,
    user_id          UUID REFERENCES users(id),
    raw_query        TEXT NOT NULL,
    rewritten_query  TEXT,
    needs_retrieval  BOOLEAN,
    retrieval        JSONB NOT NULL DEFAULT '[]'::jsonb,
    prompt_version   TEXT,
    model            TEXT,
    embedding_model  TEXT,                    -- which generation served this query
    input_tokens     INT,
    output_tokens    INT,
    latency_ms       JSONB,                   -- {"rewrite":120,"retrieve":45,"rerank":180,"llm":2400}
    cache_status     TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_traces_created ON query_traces (created_at);


CREATE TABLE golden_questions (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question           TEXT NOT NULL,
    expected_answer    TEXT,
    expected_doc_paths TEXT[],
    ask_as_role        TEXT,
    expect_refusal     BOOLEAN NOT NULL DEFAULT FALSE,
    category           TEXT,                  -- 'single-doc' | 'multi-dept' | 'no-answer' | 'acl'
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE eval_runs (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    git_commit     TEXT,
    prompt_version TEXT,
    config         JSONB NOT NULL DEFAULT '{}'::jsonb,
    summary        JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE eval_results (
    run_id       UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    question_id  UUID NOT NULL REFERENCES golden_questions(id),
    retrieved    JSONB,
    answer       TEXT,
    metrics      JSONB,
    PRIMARY KEY (run_id, question_id)
);

-- ------------------------------------------------------------
-- 8. SEED DATA  (unchanged from v1)
-- ------------------------------------------------------------

INSERT INTO roles (name) VALUES
    ('engineer'), ('hr_partner'), ('finance_analyst'), ('legal_counsel'), ('exec');

INSERT INTO role_department_access (role_id, department)
SELECT r.id, d.dept
FROM roles r
JOIN LATERAL (
    VALUES
        ('engineer',        'engineering'),
        ('engineer',        'security'),
        ('engineer',        'company'),
        ('hr_partner',      'people-group'),
        ('hr_partner',      'company'),
        ('finance_analyst', 'finance'),
        ('finance_analyst', 'company'),
        ('legal_counsel',   'legal'),
        ('legal_counsel',   'people-group'),
        ('legal_counsel',   'company'),
        ('exec',            'engineering'),
        ('exec',            'security'),
        ('exec',            'company'),
        ('exec',            'people-group'),
        ('exec',            'finance'),
        ('exec',            'legal'),
        ('exec',            'sales'),
        ('exec',            'marketing'),
        ('exec',            'product')
) AS d(role_name, dept) ON d.role_name = r.name;

-- ------------------------------------------------------------
-- 9. REFERENCE RETRIEVAL QUERY (lives in app code)
--
-- Per-department ANN: the app runs ONE such query PER allowed
-- department (in parallel), then merges. department is a
-- LITERAL in each query so the planner picks the matching
-- partial index. $1 = query embedding, k = 25.
--
--   SELECT c.id, c.content, c.heading_path, c.document_id,
--          1 - (e.embedding <=> $1) AS dense_score
--   FROM chunk_embeddings_bge_1024 e
--   JOIN chunks c ON c.id = e.chunk_id AND c.status = 'active'
--   WHERE e.department = 'finance'          -- literal per query
--   ORDER BY e.embedding <=> $1
--   LIMIT 25;
--
-- Lexical branch (single query, all allowed departments):
--
--   SELECT id, ts_rank(tsv, plainto_tsquery('english', $2)) AS score
--   FROM chunks
--   WHERE status = 'active' AND department = ANY($3)
--     AND tsv @@ plainto_tsquery('english', $2)
--   ORDER BY score DESC
--   LIMIT 25;
--
-- Merge dense (across departments) + lexical with RRF in app
-- code, then send top ~25 to the reranker.
-- ------------------------------------------------------------
