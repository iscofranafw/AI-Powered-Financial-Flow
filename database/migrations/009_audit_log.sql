-- Migration 009: independent, append-only service audit trail and RPC cleanup.
-- Apply after 001-008.
BEGIN;

-- Drop legacy 3-argument import_transactions_atomic overload from 005_ai_forecast.sql
-- if present, ensuring only the protected 4-argument version remains.
DROP FUNCTION IF EXISTS public.import_transactions_atomic(uuid, uuid, jsonb);

CREATE TABLE public.audit_log (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 actor_id uuid NOT NULL,
 action text NOT NULL CHECK (action IN (
   'update_transaction', 'delete_transaction', 'approve_transactions', 'change_owner_status')),
 target_table text NOT NULL CHECK (target_table IN ('transaksi', 'profiles')),
 target_id uuid NOT NULL,
 toko_id uuid,
 before_json jsonb,
 after_json jsonb,
 created_at timestamptz NOT NULL DEFAULT now(),
 CHECK (before_json IS NOT NULL OR after_json IS NOT NULL)
);
-- UUID identities deliberately have no cascading FK: deleted targets/accounts
-- must not erase their audit history.
CREATE INDEX audit_log_shop_created ON public.audit_log(toko_id, created_at DESC);
CREATE INDEX audit_log_target ON public.audit_log(target_table, target_id, created_at DESC);
ALTER TABLE public.audit_log ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.audit_log FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT, INSERT ON public.audit_log TO service_role;
CREATE POLICY audit_log_service_insert ON public.audit_log
 FOR INSERT TO service_role WITH CHECK (true);
CREATE POLICY audit_log_service_read ON public.audit_log
 FOR SELECT TO service_role USING (true);
COMMENT ON TABLE public.audit_log IS
 'Independent append-only server audit records; usulan.audit_history is secondary. '
 'Service mutations and audit records are executed atomically in database RPCs (e.g. mutate_finance_audited).';
COMMIT;
