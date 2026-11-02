-- P1-07: PostgreSQL arbitrates duplicate chat confirmations across workers.
-- Existing duplicate proposal IDs must be reviewed before applying this index;
-- migration failure deliberately leaves all existing financial rows untouched.
BEGIN;
CREATE UNIQUE INDEX transaksi_shop_proposal_unique
 ON public.transaksi (toko_id, (usulan->>'proposal_id'))
 WHERE nullif(usulan->>'proposal_id', '') IS NOT NULL;
COMMIT;
