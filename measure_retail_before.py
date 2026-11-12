"""Diagnostik read-only untuk reproduksi klasifikasi dataset sintetis retail.

CATATAN:
- Skrip ini BUKAN kode produksi dan TIDAK mengubah data di Supabase maupun file kode aplikasi.
- Dataset yang dievaluasi (`data_sintetis_retail_365hari.csv`) adalah DATA SINTETIS (BUKAN data toko asli).
- Mengukur kondisi SEBELUM perbaikan retail apa pun (gate threshold tetap 0.80 / margin 0.04).
"""

import csv
import logging
import os
import tempfile
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from ai.category_discovery import (
    discover_categories_from_transactions,
    suggest_default_keywords,
)
from ai.classifier import _similarity_evaluate, classify
from ai.hybrid_classifier import clear_classification_cache, hybrid_bulk_classify
from ai.rag_engine import INDEX_ROOT_DIR, build_and_save_index, clear_memory_cache
from ai.text_normalizer import clean_text_for_storage, normalize
from database.client import privileged_client
from services.finance_service import plan_category_keywords
from services.import_category_policy import preview_category_label, missing_category_types

RETAIL_CSV = Path(r"C:\Users\frana\Downloads\data_sintetis_retail_365hari.csv")
BAKERY_CSV = Path(r"C:\Users\frana\Downloads\data_sintetis_bakery_365hari.csv")

DONI_MART_ID = "f4bbebfa-cedb-486c-9fed-7fa312612ecf"
RAFI_MART_ID = "c26e69a7-c16a-47c7-a197-2e581bd4fa84"


def _load_csv_rows(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = []
        for idx, r in enumerate(csv.DictReader(fh), start=2):
            rows.append({
                "baris": idx,
                "tanggal": (r.get("tanggal") or "").strip(),
                "deskripsi": clean_text_for_storage(r.get("deskripsi") or ""),
                "nominal": (r.get("nominal") or "0").strip(),
                "tipe": (r.get("tipe") or "pengeluaran").strip().lower(),
                "usulan": {},
                "error": "",
            })
        return rows


def _unique_descriptions(rows):
    freq = Counter((r["tipe"], r["deskripsi"]) for r in rows)
    return sorted(freq.items(), key=lambda x: (-x[1], x[0][1]))


def _evaluate_gate_table(cats, unique_items, toko_id=None):
    rows = []
    for (tp, desc), count in unique_items:
        dec, diag = _similarity_evaluate(tp, desc, cats, toko_id=toko_id)
        rows.append({
            "tipe": tp,
            "deskripsi": desc,
            "count": count,
            "top_cat": diag.get("top_category") or "-",
            "top_score": diag.get("top_score"),
            "second_score": diag.get("second_score"),
            "margin": diag.get("margin"),
            "gate_passed": bool(diag.get("gate_passed")),
            "dec": dec,
        })
    return rows


def _build_temp_faiss_entries(shop_id: str, cats, business_type: str):
    entries = []
    for c in cats:
        cname = str(c.get("nama", "")).strip()
        if c.get("sistem") or cname.casefold() == "others":
            continue
        desc = str(c.get("deskripsi") or "").strip()
        kws = [str(k).strip() for k in (c.get("kata_kunci") or []) if str(k).strip()]
        parts = [cname] + ([desc] if desc else []) + ([" ".join(kws)] if kws else [])
        text = ". ".join(parts)
        entries.append({
            "id": f"cat_{c['id']}",
            "type": "category",
            "store_id": shop_id,
            "transaction_type": c["tipe"],
            "category_id": c["id"],
            "category_name": cname,
            "text": text,
            "normalized_desc": normalize(text),
            "business_type": business_type,
            "verification_source": "category_definition",
        })
        for kw_idx, kw in enumerate(kws):
            kw_text = kw
            entries.append({
                "id": f"cat_{c['id']}_kw_{kw_idx}",
                "type": "category",
                "store_id": shop_id,
                "transaction_type": c["tipe"],
                "category_id": c["id"],
                "category_name": cname,
                "text": kw_text,
                "normalized_desc": normalize(kw_text),
                "business_type": business_type,
                "verification_source": "category_keyword",
            })
    return entries


def main():
    logging.basicConfig(level=logging.WARNING)
    retail_rows = _load_csv_rows(RETAIL_CSV)
    unique_retail = _unique_descriptions(retail_rows)
    print("=== DATASET SINTETIS RETAIL (DATA SINTETIS - BUKAN DATA TOKO ASLI) ===")
    print(f"File: {RETAIL_CSV} | Total rows: {len(retail_rows)} | Unique descriptions: {len(unique_retail)}")

    sb = privileged_client()

    # 1. Inspect Doni Mart & Rafi Mart read-only from Supabase
    for label, tid in [("Doni Mart", DONI_MART_ID), ("Rafi Mart", RAFI_MART_ID)]:
        toko = sb.table("toko").select("*").eq("id", tid).single().execute().data
        cats = sb.table("kategori").select("*").eq("toko_id", tid).execute().data or []
        print(f"\n=== 1. DB CATEGORIES (READ-ONLY): {label} ({tid}) | bidang_usaha={toko.get('bidang_usaha')!r} ===")
        for c in cats:
            print(
                f"  - [{c['tipe']}] {c['nama']} | sistem={c.get('sistem')} | modal={c.get('bukan_operasional')} "
                f"| desc={c.get('deskripsi')!r} | kata_kunci({len(c.get('kata_kunci') or [])})={c.get('kata_kunci')}"
            )

    doni_cats = sb.table("kategori").select("*").eq("toko_id", DONI_MART_ID).execute().data or []
    rafi_cats = sb.table("kategori").select("*").eq("toko_id", RAFI_MART_ID).execute().data or []

    # 2. Reproduce exact Doni Mart current state (0 active expense categories)
    print("\n=== 2. REPRODUKSI STATE ACTUAL DONI MART SAAT INI (DRY-RUN, 0 DB WRITES) ===")
    doni_gate = _evaluate_gate_table(doni_cats, unique_retail, toko_id=DONI_MART_ID)
    print(f"Doni Mart Similarity Gate PASS: {sum(1 for g in doni_gate if g['gate_passed'])}/{len(doni_gate)}")
    clear_classification_cache(DONI_MART_ID)
    doni_classified = hybrid_bulk_classify(DONI_MART_ID, doni_cats, retail_rows, business_type="Retail")
    doni_cat_map = {c["id"]: c["nama"] for c in doni_cats}
    doni_preview_col = Counter(preview_category_label(r, doni_cat_map) for r in doni_classified)
    print("Tahap 4 missing types:", missing_category_types(retail_rows, doni_cats, []))
    doni_status_col = Counter(r.get("status_kategori") for r in doni_classified)
    doni_reason_col = Counter(r.get("reason_code") for r in doni_classified)
    print(f"Kolom Preview Kategori: {dict(doni_preview_col)}")
    print(f"Kolom 'status_kategori': {dict(doni_status_col)}")
    print(f"reason_code: {dict(doni_reason_col)}")

    # 3. Test Stage 2 Category Discovery on data_sintetis_retail_365hari.csv (live Ollama, 0 DB writes)
    print("\n=== 3. SIMULASI TAHAP 2 DISCOVERY + TAHAP 5 PADA DATA RETAIL (0 DB WRITES) ===")
    discovery = discover_categories_from_transactions(
        rows=retail_rows,
        existing_categories=doni_cats,
        business_type="Retail",
        max_recommendations_per_type=7,
    )
    print(f"Coverage stats (vs Doni Mart existing): {discovery.get('coverage_stats')}")
    exp_recs = discovery.get("pengeluaran", [])
    print(f"LLM Discovery proposed {len(exp_recs)} pengeluaran categories:")
    discovered_cats = [dict(c) for c in doni_cats]
    for rec in exp_recs:
        cid = str(uuid.uuid4())
        kws_llm_contoh = suggest_default_keywords(
            rec["nama"], "pengeluaran", business_type="Retail",
            contoh=rec.get("contoh"), llm_keywords=rec.get("kata_kunci"),
        )
        print(
            f"  + {rec['nama']!r} | desc={rec.get('deskripsi')!r} | contoh={rec.get('contoh')} "
            f"| suggested_kw({len(rec.get('kata_kunci') or [])})={rec.get('kata_kunci')}"
        )
        discovered_cats.append({
            "id": cid,
            "toko_id": "sim-doni-discovered",
            "nama": rec["nama"],
            "tipe": "pengeluaran",
            "deskripsi": rec.get("deskripsi", ""),
            "sistem": False,
            "bukan_operasional": False,
            "kata_kunci": rec.get("kata_kunci") or kws_llm_contoh,
        })

    # Build temporary isolated FAISS index in temp dir so nothing touches vector_store/ or DB
    with tempfile.TemporaryDirectory() as tmpdir:
        sim_shop = "sim-doni-discovered"
        entries = _build_temp_faiss_entries(sim_shop, [c for c in discovered_cats if c["tipe"] == "pengeluaran"], "Retail")
        with patch("ai.rag_engine.INDEX_ROOT_DIR", tmpdir), \
             patch("ai.rag_engine.get_remote_version", return_value=1):
            clear_memory_cache(sim_shop)
            clear_classification_cache(sim_shop)
            build_and_save_index(sim_shop, "pengeluaran", entries=entries, version=1)
            sim_gate = _evaluate_gate_table(discovered_cats, unique_retail, toko_id=sim_shop)
            print(f"\nSimilarity Gate (Discovered Retail Categories): PASS={sum(1 for g in sim_gate if g['gate_passed'])}/{len(sim_gate)}")
            for g in sim_gate:
                print(
                    f"  [pengeluaran] ({g['count']:3d}x) {g['deskripsi']:34s} -> "
                    f"top1={str(g['top_cat']):28s} ({g['top_score']}), margin={g['margin']}, gate={g['gate_passed']}"
                )
            sim_classified = hybrid_bulk_classify(sim_shop, discovered_cats, retail_rows, business_type="Retail")
            sim_map = {c["id"]: c["nama"] for c in discovered_cats}
            sim_dist = Counter(sim_map.get(r.get("kategori_id"), "NULL/Perlu Tinjauan") for r in sim_classified)
            sim_src = Counter(r.get("source") for r in sim_classified)
            print(f"Distribusi 1,460 baris (Discovered Retail): {dict(sim_dist)}")
            print(f"Sources: {dict(sim_src)}")
            print("Per-description hasil akhir (Discovered Retail):")
            seen = {}
            for r in sim_classified:
                seen[r["deskripsi"]] = (
                    sim_map.get(r.get("kategori_id"), "Perlu Tinjauan"),
                    r.get("status_kategori"),
                    r.get("source"),
                    r.get("skor"),
                )
            for (tp, desc), cnt in unique_retail:
                cname, st_k, src, sk = seen[desc]
                print(f"  {cnt:3d}x {desc:34s} -> {cname:28s} ({st_k}, {src}, skor={sk})")

    # 4. Also measure Rafi Mart BEFORE (existing 7 expense cats, kata_kunci=[]) & after simulated suggest_default_keywords
    print("\n=== 4. PENGUKURAN RAFI MART SAAT INI (kata_kunci=[], 0 DB WRITES) & SIMULASI AUTO-FILL DEFAULT ===")
    rafi_gate_before = _evaluate_gate_table(rafi_cats, unique_retail, toko_id=None)
    print(f"Rafi Mart BEFORE Similarity Gate PASS: {sum(1 for g in rafi_gate_before if g['gate_passed'])}/{len(rafi_gate_before)}")
    for g in rafi_gate_before:
        print(
            f"  ({g['count']:3d}x) {g['deskripsi']:34s} -> "
            f"top1={str(g['top_cat']):28s} ({g['top_score']}), margin={g['margin']}, gate={g['gate_passed']}"
        )
    clear_classification_cache(RAFI_MART_ID)
    rafi_classified_before = hybrid_bulk_classify(RAFI_MART_ID, rafi_cats, retail_rows, business_type="Retail")
    rafi_map = {c["id"]: c["nama"] for c in rafi_cats}
    print(f"Rafi Mart BEFORE Distribusi 1,460 baris: {dict(Counter(rafi_map.get(r.get('kategori_id'), 'NULL/Perlu Tinjauan') for r in rafi_classified_before))}")
    print(f"Rafi Mart BEFORE Sources: {dict(Counter(r.get('source') for r in rafi_classified_before))}")
    print(f"Rafi Mart BEFORE status_kategori: {dict(Counter(r.get('status_kategori') for r in rafi_classified_before))}")
    seen_rafi = {}
    for r in rafi_classified_before:
        seen_rafi[r["deskripsi"]] = (
            rafi_map.get(r.get("kategori_id"), "Perlu Tinjauan"),
            r.get("status_kategori"),
            r.get("source"),
            r.get("skor"),
        )
    for (tp, desc), cnt in unique_retail:
        cname, st_k, src, sk = seen_rafi[desc]
        print(f"  {cnt:3d}x {desc:34s} -> {cname:28s} ({st_k}, {src}, skor={sk})")

    print("\nSimulasi auto_fill_category_keywords() pada kategori Rafi Mart (tanpa menulis DB):")
    rafi_history = (sb.table('transaksi').select('kategori_id,deskripsi,tipe')
                    .eq('toko_id', RAFI_MART_ID).eq('status_persetujuan', 'disetujui')
                    .eq('status_kategori', 'dikonfirmasi').order('tanggal', desc=True).limit(1000).execute().data or [])
    print('Approved confirmed history rows:', len(rafi_history))
    # Use the production read-only planner. The former per-category simulation
    # omitted current CSV examples, duplicated income templates and evicted new
    # descriptions behind history at the 20-keyword limit.
    keyword_plan = plan_category_keywords(rafi_cats, rafi_history, 'Retail', retail_rows)
    rafi_filled_cats = keyword_plan['categories']
    for warning in keyword_plan['warnings']:
        print('WARNING:', warning)
    print('Keyword ownership audit:', keyword_plan['audit'])
    for c in rafi_filled_cats:
        if not c.get('sistem'):
            print(f"  - [{c['tipe']}] {c['nama']}: {len(c.get('kata_kunci') or [])} kw: {c.get('kata_kunci')}")

    with tempfile.TemporaryDirectory() as tmpdir2:
        sim_rafi = "sim-rafi-autofill"
        entries_r = _build_temp_faiss_entries(sim_rafi, [c for c in rafi_filled_cats if c["tipe"] == "pengeluaran"], "Retail")
        with patch("ai.rag_engine.INDEX_ROOT_DIR", tmpdir2), \
             patch("ai.rag_engine.get_remote_version", return_value=1):
            clear_memory_cache(sim_rafi)
            clear_classification_cache(sim_rafi)
            build_and_save_index(sim_rafi, "pengeluaran", entries=entries_r, version=1)
            rafi_gate_after = _evaluate_gate_table(rafi_filled_cats, unique_retail, toko_id=sim_rafi)
            print(f"\nRafi Mart (Simulated auto_fill_category_keywords) Similarity Gate PASS: {sum(1 for g in rafi_gate_after if g['gate_passed'])}/{len(rafi_gate_after)}")
            for g in rafi_gate_after:
                print(
                    f"  ({g['count']:3d}x) {g['deskripsi']:34s} -> "
                    f"top1={str(g['top_cat']):28s} ({g['top_score']}), margin={g['margin']}, gate={g['gate_passed']}"
                )
            filled_classified = hybrid_bulk_classify(sim_rafi, rafi_filled_cats, retail_rows, business_type="Retail")
            print("Rafi Mart AFTER Distribusi 1,460 baris:", dict(Counter(rafi_map.get(r.get('kategori_id'), 'NULL/Perlu Tinjauan') for r in filled_classified)))



if __name__ == "__main__":
    main()
