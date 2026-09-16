-- =====================================================================
-- Migration 002 — identity seed, honest trace outcomes, eval seed.
--
-- Closes the three gaps:
--   Gap 1: users / user_roles were empty → identity was never exercised.
--   Gap 2: query_traces.outcome collapsed floor- and model-refusals;
--          stop_reason/outcome may not exist yet on older DBs.
--   Gap 3: no golden questions to drive the eval harness.
--
-- Safe to re-run (IF NOT EXISTS / ON CONFLICT / NOT EXISTS guards).
-- =====================================================================


-- ------------------------------------------------------------
-- 1. query_traces: stop_reason + three-valued outcome
-- ------------------------------------------------------------

ALTER TABLE query_traces ADD COLUMN IF NOT EXISTS stop_reason TEXT;
ALTER TABLE query_traces ADD COLUMN IF NOT EXISTS outcome     TEXT;

-- Migrate any legacy values before tightening the check:
--   'not_found' was only ever written by the floor branch.
UPDATE query_traces SET outcome = 'refused_by_floor'
WHERE outcome = 'not_found';

ALTER TABLE query_traces DROP CONSTRAINT IF EXISTS query_traces_outcome_check;
ALTER TABLE query_traces ADD CONSTRAINT query_traces_outcome_check
    CHECK (outcome IS NULL OR outcome IN
           ('answered', 'refused_by_floor', 'refused_by_model'));


-- ------------------------------------------------------------
-- 2. Identity seed: one user per role (+ email is the CLI key)
-- ------------------------------------------------------------

INSERT INTO users (email, display_name) VALUES
    ('eng@example.com',     'Erin Engineer'),
    ('hr@example.com',      'Harper HR'),
    ('finance@example.com', 'Frank Finance'),
    ('legal@example.com',   'Lena Legal'),
    ('exec@example.com',    'Evelyn Exec')
ON CONFLICT (email) DO NOTHING;

INSERT INTO user_roles (user_id, role_id)
SELECT u.id, r.id
FROM (VALUES
    ('eng@example.com',     'engineer'),
    ('hr@example.com',      'hr_partner'),
    ('finance@example.com', 'finance_analyst'),
    ('legal@example.com',   'legal_counsel'),
    ('exec@example.com',    'exec')
) AS m(email, role_name)
JOIN users u ON u.email = m.email
JOIN roles r ON r.name  = m.role_name
ON CONFLICT (user_id, role_id) DO NOTHING;


-- ------------------------------------------------------------
-- 3. golden_questions: identity-keyed asks
--    (ask_as_role kept for legacy rows; the eval prefers email)
-- ------------------------------------------------------------

ALTER TABLE golden_questions ADD COLUMN IF NOT EXISTS ask_as_email TEXT;

-- Seed. expected_doc_paths entries are matched by PREFIX in the eval
-- runner, so folder-level prefixes are valid expectations — tighten
-- them to exact document paths once you've eyeballed real retrievals.
INSERT INTO golden_questions
    (question, ask_as_email, expect_refusal, expected_doc_paths, category)
SELECT v.question, v.email, v.refuse, v.paths, v.category
FROM (VALUES
    -- Straightforward single-department answers
    ('How do I submit travel expenses for reimbursement?',
     'finance@example.com', FALSE,
     ARRAY['content/handbook/finance/'], 'single-doc'),

    ('What is the parental leave policy?',
     'hr@example.com', FALSE,
     ARRAY['content/handbook/people-group/', 'content/handbook/people-policies/',
           'content/handbook/total-rewards/'], 'single-doc'),

    ('What is the company mission?',
     'eng@example.com', FALSE,
     ARRAY['content/handbook/company/'], 'single-doc'),

    -- ACL: same parental-leave question, asked by someone whose roles
    -- grant finance+company only. people-group chunks must never enter
    -- the candidate pool, so this MUST come back as a refusal — and it
    -- must be indistinguishable, to the user, from a plain no-answer.
    ('What is the parental leave policy?',
     'finance@example.com', TRUE,
     NULL, 'acl'),

    -- Multi-department: engineer sees engineering + security + company
    ('What is the process for getting access to production systems?',
     'eng@example.com', FALSE,
     ARRAY['content/handbook/engineering/', 'content/handbook/security/'],
     'multi-dept'),

    -- True no-answer: nothing in the corpus, even for the exec who can
    -- see everything. Exercises the floor / model-refusal path without
    -- ACL as a confound.
    ('What is the reimbursement rate for commuting by hot air balloon?',
     'exec@example.com', TRUE,
     NULL, 'no-answer')
) AS v(question, email, refuse, paths, category)
WHERE NOT EXISTS (
    SELECT 1 FROM golden_questions g
    WHERE g.question = v.question
      AND g.ask_as_email IS NOT DISTINCT FROM v.email
);
