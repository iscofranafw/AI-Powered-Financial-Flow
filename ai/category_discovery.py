import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ai.classifier import classify, detect_type
from ai.hybrid_classifier import classify_ambiguous_batch_ai
from ai.ollama_client import LLMUnavailable, ask_json
from ai.text_normalizer import clean_text_for_storage, normalize

DISCOVERY_SCHEMA = {
    'type': 'object',
    'properties': {
        'rekomendasi': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'nama': {'type': 'string'},
                    'deskripsi': {'type': 'string'},
                    'contoh': {'type': 'array', 'items': {'type': 'string'}},
                },
                'required': ['nama', 'deskripsi'],
            },
        }
    },
    'required': ['rekomendasi'],
}


def _is_covered_by_existing(norm_desc, tipe, existing_categories, business_type='Umum'):
    result = classify(tipe, norm_desc, existing_categories, business_type=business_type)
    return result['source'] == 'llm' and bool(result['kategori_id'])


def _is_name_duplicate_or_equivalent(
    new_name: str,
    existing_names: Set[str],
) -> bool:
    """Check if proposed category name is duplicate or semantically equivalent to existing names."""
    clean_new = normalize(new_name)
    if not clean_new:
        return True

    for ex in existing_names:
        clean_ex = normalize(ex)
        if clean_new == clean_ex:
            return True
        # Check simple substring identity if length is meaningful
        if len(clean_new) >= 5 and len(clean_ex) >= 5:
            if clean_new in clean_ex or clean_ex in clean_new:
                return True

    return False


def _discover_for_type_with_llm(
    tipe: str,
    uncovered_samples: List[Tuple[str, str, int]],  # (norm_desc, raw_desc, count)
    existing_names: List[str],
    business_type: str,
    max_count: int = 7,
) -> List[Dict[str, Any]]:
    """Query Llama 3 8B to discover broad, grounded categories from uncovered transaction patterns."""
    if not uncovered_samples:
        return []

    sample_lines = [
        f"- \"{raw}\" (frekuensi: {cnt}x)"
        for _, raw, cnt in uncovered_samples[:25]
    ]
    sample_text = "\n".join(sample_lines)

    existing_str = ", ".join(f"'{name}'" for name in existing_names) if existing_names else "(Belum ada)"

    system_prompt = (
        f"Kamu adalah Senior Financial Accounting Expert untuk UMKM Indonesia bidang '{business_type}'.\n"
        f"Tugasmu: Menganalisis contoh transaksi {tipe} yang belum tercakup oleh kategori toko saat ini, "
        f"dan merekomendasikan kategori keuangan yang relevan, bermakna, dan cukup luas (maksimal {max_count} kategori).\n\n"
        "ATURAN KETAT:\n"
        "1. Kategori harus BERMAKNA dan CUKUP LUAS untuk mengelompokkan transaksi sejenis, BUKAN nama produk/merek spesifik "
        "(Contoh BAIK: 'Stok Barang Dagangan', 'Bahan Baku', 'Jasa Servis', 'Penjualan Toko'; "
        "Contoh BURUK: 'Rokok Dunhill', 'Indomie Goreng', 'Oli MPX').\n"
        f"2. DILARANG mengusulkan nama yang sudah ada atau semakna dengan kategori toko yang sudah aktif: [{existing_str}].\n"
        f"3. DILARANG mengusulkan lebih dari {max_count} kategori.\n"
        "4. Berikan deskripsi singkat dan 2-3 contoh deskripsi transaksi yang termasuk ke dalam kategori tersebut.\n"
        "5. Balasan WAJIB berupa JSON valid sesuai schema."
    )

    user_payload = {
        'tipe_transaksi': tipe,
        'bidang_usaha': business_type,
        'kategori_aktif_toko': existing_names,
        'contoh_transaksi_uncovered': [raw for _, raw, _ in uncovered_samples[:25]],
    }

    try:
        reply = ask_json(
            system=system_prompt,
            user=json.dumps(user_payload, ensure_ascii=False),
            schema=DISCOVERY_SCHEMA,
            num_predict=512,
        )
        recommendations = reply.get('rekomendasi', [])
        valid_recs = []
        seen_names: Set[str] = set(n.casefold() for n in existing_names)

        for rec in recommendations:
            raw_name = clean_text_for_storage(rec.get('nama', '')).strip()
            # Enforce max 80 chars, meaningful length
            if not raw_name or len(raw_name) < 3 or len(raw_name) > 80:
                continue
            # Exclude reserved system categories
            if raw_name.casefold() in ('others', 'tambahan modal'):
                continue
            # Exclude duplicates
            if _is_name_duplicate_or_equivalent(raw_name, seen_names):
                continue

            seen_names.add(raw_name.casefold())
            valid_recs.append({
                'nama': raw_name,
                'tipe': tipe,
                'deskripsi': str(rec.get('deskripsi', '')).strip()[:300] or f"Pos {tipe} untuk {raw_name}",
                'contoh': [str(c).strip() for c in rec.get('contoh', [])[:3] if str(c).strip()],
                'source': 'ai_discovery',
            })
            if len(valid_recs) >= max_count:
                break

        return valid_recs

    except Exception as exc:
        raise LLMUnavailable('Penemuan kategori memerlukan Ollama llama3:8b. ' + str(exc)) from exc


def discover_categories_from_transactions(
    rows: List[Dict[str, Any]],
    existing_categories: List[Dict[str, Any]],
    business_type: str = 'Umum',
    max_recommendations_per_type: int = 7,
) -> Dict[str, Any]:
    """Analyze uploaded transaction dataset and discover appropriate categories without exceeding 7 AI recommendations per type.

    Workflow:
    1. Group transactions by transaction type (pemasukan / pengeluaran).
    2. Check coverage against existing store categories.
    3. Identify uncovered transaction descriptions.
    4. If transactions are well-covered, avoid proposing redundant categories.
    5. Propose up to 7 income and 7 expense categories using Llama 3 8B only.
    6. Ensure strict deduplication and financial integrity.
    """
    if not rows:
        return {
            'pengeluaran': [],
            'pemasukan': [],
            'existing_coverage': {'pengeluaran': 0.0, 'pemasukan': 0.0},
        }

    # Group and count unique normalized descriptions by type
    desc_freq: Dict[str, Dict[str, Tuple[str, int]]] = {'pengeluaran': {}, 'pemasukan': {}}

    for r in rows:
        raw_desc = str(r.get('deskripsi') or '')
        clean_desc = clean_text_for_storage(raw_desc)
        norm_desc = normalize(clean_desc)
        if not norm_desc:
            continue

        tp = (r.get('tipe') or '').strip().lower()
        if not tp or tp not in ('pemasukan', 'pengeluaran'):
            try:
                tp = detect_type(clean_desc)
            except Exception:
                tp = 'pengeluaran'

        if norm_desc not in desc_freq[tp]:
            desc_freq[tp][norm_desc] = (clean_desc, 0)
        curr_raw, curr_count = desc_freq[tp][norm_desc]
        desc_freq[tp][norm_desc] = (curr_raw, curr_count + 1)

    result_recommendations: Dict[str, List[Dict[str, Any]]] = {'pengeluaran': [], 'pemasukan': []}
    coverage_stats: Dict[str, Dict[str, Any]] = {}

    for tp in ('pengeluaran', 'pemasukan'):
        active_existing = [c for c in existing_categories if c.get('tipe') == tp]
        existing_names = [c['nama'] for c in active_existing]

        items = desc_freq[tp]
        total_unique = len(items)
        if total_unique == 0:
            coverage_stats[tp] = {'total_unique': 0, 'covered': 0, 'uncovered': 0, 'coverage_pct': 100.0}
            continue

        covered_count = 0
        uncovered_samples: List[Tuple[str, str, int]] = []

        coverage = {}
        if active_existing:
            descriptions = list(items)
            for start in range(0, len(descriptions), 8):
                batch = list(enumerate(descriptions[start:start + 8], 1))
                decisions = classify_ambiguous_batch_ai(tp, batch, active_existing, business_type=business_type)
                coverage.update({desc: decisions[ident]['source'] == 'llm' and bool(decisions[ident]['kategori_id'])
                                 for ident, desc in batch})
        for norm_desc, (raw_desc, count) in items.items():
            if coverage.get(norm_desc, False):
                covered_count += 1
            else:
                uncovered_samples.append((norm_desc, raw_desc, count))

        coverage_pct = round((covered_count / total_unique) * 100, 1) if total_unique > 0 else 100.0
        coverage_stats[tp] = {
            'total_unique': total_unique,
            'covered': covered_count,
            'uncovered': len(uncovered_samples),
            'coverage_pct': coverage_pct,
        }

        # If uncovered descriptions exist, discover suitable categories
        if uncovered_samples:
            # Sort by frequency descending for representative sampling
            uncovered_samples.sort(key=lambda x: x[2], reverse=True)

            # Try LLM discovery first
            discovered = _discover_for_type_with_llm(
                tipe=tp,
                uncovered_samples=uncovered_samples,
                existing_names=existing_names,
                business_type=business_type,
                max_count=max_recommendations_per_type,
            )

            # Strictly cap at max_recommendations_per_type
            result_recommendations[tp] = discovered[:max_recommendations_per_type]
        else:
            result_recommendations[tp] = []

    return {
        'pengeluaran': result_recommendations['pengeluaran'],
        'pemasukan': result_recommendations['pemasukan'],
        'coverage_stats': coverage_stats,
    }

