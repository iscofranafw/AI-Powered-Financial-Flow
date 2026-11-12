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
            'minItems': 1,
            'items': {
                'type': 'object',
                'properties': {
                    'nama': {'type': 'string'},
                    'deskripsi': {'type': 'string'},
                    'contoh': {'type': 'array', 'minItems': 3, 'items': {'type': 'string'}},
                    'kata_kunci': {'type': 'array', 'minItems': 1, 'items': {'type': 'string'}},
                },
                'required': ['nama', 'deskripsi', 'contoh', 'kata_kunci'],
            },
        }
    },
    'required': ['rekomendasi'],
}

# Reference profiles are loaded from data/knowledge/category_keyword_profiles.json.
# Each group records its actual data/knowledge or data/samples source; no synthetic
# 365-day descriptions are embedded in application code.
# Bakery: bahan, kemasan, operasional, gaji, promosi, logistik, penjualan:
#   data/knowledge/bakery/panduan_operasional_bakery.md + data/samples/bakery.csv.
# Retail: stok, kemasan, perlengkapan/furnitur, operasional, gaji, promosi,
#   logistik, administrasi, penjualan: data/knowledge/retail/panduan_operasional_retail.md
#   + data/samples/grosir.csv. Product terms absent from the user's CSV stay out
#   of generic Bahan Baku; retail references do not inject kitchen ingredients.
# Warung: bahan, kemasan, operasional, gaji, promosi, penjualan:
#   data/knowledge/warung_makan/panduan_operasional_warung_makan.md
#   + data/samples/warung_makan.csv; logistik uses the guide's pesan antar section.
# Bengkel: komponen, alat, operasional, gaji, promosi, jasa, penjualan:
#   data/knowledge/bengkel_motor/panduan_operasional_bengkel_motor.md
#   + data/samples/bengkel_motor.csv. JSON records the sources per group;
#   regression tests verify that every reference phrase occurs in its cited file.
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def _reference_profiles():
    path = Path(__file__).resolve().parents[1] / 'data/knowledge/category_keyword_profiles.json'
    profiles = json.loads(path.read_text(encoding='utf-8'))
    # Product-group lists are parsed from the cited guide, never copied from a
    # synthetic CSV. These terms ground a *matching uploaded description* only;
    # they are not injected as keywords into arbitrary categories.
    for groups in profiles.values():
        for group in groups:
            terms = []
            for source in group.get('grounding_sources', []):
                for line in (path.parents[1] / Path(source['path']).relative_to('data')).read_text(encoding='utf-8').splitlines():
                    if any(line.startswith(prefix) for prefix in source['line_prefixes']):
                        body = line.split(':**', 1)[-1].split('—', 1)[0]
                        terms.extend(t.strip(' .') for t in re.split(r'[,/]', body) if t.strip(' .'))
            group['grounding_terms'] = terms
    return profiles

_DOMAIN_KEYWORD_PATTERNS = {}  # Compatibility: profiles live in the cited corpus.
_GENERIC_KEYWORD_PATTERNS = []  # No cross-domain product injection.


def _canonical_domain(business_type: str) -> str:
    """Map free-text bidang_usaha string to canonical keyword template domain key."""
    field = normalize(business_type or '')
    if any(k in field for k in ('bakery', 'roti', 'kue', 'pastry', 'cake')):
        return 'bakery'
    if any(k in field for k in ('retail', 'minimarket', 'kelontong', 'sembako', 'grosir', 'mart', 'warung sembako')):
        return 'retail'
    if any(k in field for k in ('warung makan', 'restoran', 'rumah makan', 'kuliner', 'cafe', 'katering')):
        return 'warung_makan'
    if any(k in field for k in ('bengkel', 'otomotif', 'sparepart', 'motor')):
        return 'bengkel_motor'
    return 'umum'


def extract_keywords_from_samples(samples: Optional[List[str]], max_items: int = 20) -> List[str]:
    """Retain real descriptions first; compact variants must not displace coverage.

    The embedding normalizer removes numbers, but retains unit tokens. Removing
    units from every keyword previously weakened short ingredient descriptions.
    Keep one complete user description per normalized pattern, then variants.
    """
    extracted, secondary, seen = [], [], set()
    for raw in samples or []:
        cleaned = clean_text_for_storage(raw).strip()
        key = normalize(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        if 3 <= len(cleaned) <= 80:
            extracted.append(cleaned)
        core = re.sub(r'\b\d+(?:[.,]\d+)?\s*(?:kg|g|gr|gram|l|lt|liter|ml|pcs|pak|pack|dus|box|botol|bh|buah|unit|rb|ribu)?\b', '', cleaned, flags=re.I)
        core = re.sub(r'\s+', ' ', core).strip(' ,.-')
        secondary.extend([core, re.sub(r'^(?:beli|bayar|biaya|belanja|pesan|pembayaran|uang)\s+', '', core, flags=re.I)])
        if len(extracted) >= max_items:
            return extracted
    for phrase in secondary:
        key = normalize(phrase)
        if 3 <= len(phrase) <= 80 and key and key not in seen:
            seen.add(key)
            extracted.append(phrase)
            if len(extracted) >= max_items:
                break
    return extracted


def suggest_default_keywords(
    category_name: str,
    tipe: str = 'pengeluaran',
    business_type: str = 'Umum',
    contoh: Optional[List[str]] = None,
    llm_keywords: Optional[List[str]] = None,
    max_keywords: int = 20,
) -> List[str]:
    """Seed from user examples, then discovery, then cited domain references (max 20)."""
    clean_name = str(category_name or '').strip()
    if not clean_name or clean_name.casefold() == 'others':
        return []
    norm_name = clean_text_for_storage(clean_name).casefold()
    domain = _canonical_domain(business_type)
    merged: List[str] = []
    seen: Set[str] = set()

    def _add_many(items: Optional[List[str]]):
        for item in items or []:
            kw = clean_text_for_storage(str(item or '')).strip()[:80]
            if len(kw) < 2:
                continue
            key = kw.casefold()
            if key not in seen:
                seen.add(key)
                merged.append(kw)
                if len(merged) >= max_keywords:
                    return True
        return False

    # User CSV first, discovery second, cited domain references last.
    _add_many(extract_keywords_from_samples(contoh, max_items=max_keywords))
    if len(merged) < max_keywords:
        _add_many(llm_keywords)
    if len(merged) < max_keywords:
        for group in _reference_profiles().get(domain, []):
            if group.get('tipe', tipe) != tipe:
                continue
            if any(alias in norm_name for alias in group['aliases']):
                # Sources and excerpts for every group are recorded in the corpus JSON.
                _add_many(group['keywords'])
                if len(merged) >= max_keywords:
                    break
    return merged[:max_keywords]


def matching_transaction_samples(category, rows):
    """Ground discovery/CSV category examples in this upload, never another store.

    Exact cited examples and explicit CSV category assignments take precedence;
    keyword overlap only supplies additional examples for the same transaction type.
    """
    cited = {normalize(x) for x in category.get('contoh', [])}
    phrases = [normalize(x) for x in category.get('kata_kunci', []) if normalize(x)]
    name = normalize(category.get('nama', ''))
    exact, fallback = [], []
    for row in rows:
        if row.get('tipe') != category.get('tipe'):
            continue
        raw = str(row.get('deskripsi') or '').strip()
        norm = normalize(raw)
        explicit = normalize(row.get('csv_category') or row.get('kategori') or row.get('kategori_asli') or '')
        if norm and (norm in cited or (explicit and explicit == name)):
            if raw not in exact:
                exact.append(raw)
        elif norm and any(phrase in norm for phrase in phrases):
            if raw not in fallback:
                fallback.append(raw)
    # A generic model word such as "biaya" must not expand grounded examples
    # into unrelated groups during the initial UI merge.
    return exact or fallback



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
        f'Kelompokkan SEMUA transaksi {tipe} usaha {business_type} menjadi kategori akuntansi luas, maksimal {max_count}. '
        f'Jangan usulkan nama produk/merek, Others, Tambahan Modal, atau kategori yang sudah ada: {existing_str}. '
        'Pilih nama berdasarkan tujuan transaksi. Gaji/upah adalah imbalan kerja, bukan perlengkapan. '
        'Kemasan/perlengkapan adalah barang dipakai toko, bukan persediaan untuk dijual. '
        'Usulkan kategori hanya jika ada bukti di CSV. Cakup seluruh sampel, bukan hanya kelompok pertama. '
        'Setiap rekomendasi harus memiliki nama, deskripsi singkat, contoh ASLI dari CSV, dan kata_kunci item konkret CSV. '
        'Salin contoh persis. Bila kelompok hanya punya satu atau dua contoh unik, ulangi contoh itu untuk memenuhi tiga elemen schema. '
        'Jangan mengarang contoh atau menambahkan keterangan ke dalam string contoh. referensi_opsional hanya panduan; pilih kelompok yang didukung CSV dan tambahkan kelompok baru bila perlu. Jangan lupakan kelompok dengan satu sampel. Jawab JSON sesuai schema.'
    )
    if _canonical_domain(business_type) == 'retail' and tipe == 'pengeluaran':
        system_prompt += (
            ' Untuk usaha dagang/retail/grosir: barang untuk dijual kembali wajib Stok Barang Dagangan / Persediaan '
            '(BUKAN Bahan Baku). Tagihan listrik/air/internet wajib Operasional (BUKAN Jasa Servis). '
            'Pembungkus belanja toko adalah Perlengkapan atau Kemasan, bukan barang untuk dijual kembali.'
        )

    user_payload = {
        'tipe_transaksi': tipe,
        'bidang_usaha': business_type,
        'referensi_opsional': [g for g in _reference_profiles().get(_canonical_domain(business_type), [])
                               if g.get('tipe') == tipe],
        'kategori_aktif_toko': existing_names,
        'contoh_transaksi_uncovered': sorted({raw for _, raw, _ in uncovered_samples[:25]}),
    }

    try:
        recommendations = []
        for attempt in range(2):
            reply = ask_json(
                system=system_prompt + (' Rekomendasi sebelumnya kosong; kelompokkan sampel yang diberikan.' if attempt else ''),
                user=json.dumps(user_payload, ensure_ascii=False),
                schema=DISCOVERY_SCHEMA,
                num_predict=4096,
            )
            recommendations = reply.get('rekomendasi', [])
            semantic_error = (_canonical_domain(business_type) == 'retail' and tipe == 'pengeluaran'
                              and any(normalize(r.get('nama', '')) in ('bahan baku', 'jasa servis') for r in recommendations))
            if recommendations and not semantic_error:
                break
            if semantic_error:
                system_prompt += '\nKoreksi jawaban sebelumnya: barang retail HARUS Stok Barang Dagangan; utilitas HARUS Operasional. Kembalikan semua kelompok CSV.'
                recommendations = []
        if not recommendations:
            raise ValueError('Discovery mengembalikan rekomendasi kosong setelah dua percobaan. Coba ulang atau tentukan kategori manual.')
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
            contoh_list = matching_transaction_samples(
                {**rec, 'nama': raw_name, 'tipe': tipe},
                [{'deskripsi': raw, 'tipe': tipe} for _, raw, _ in uncovered_samples],
            )
            # Empty/malformed model examples are repaired by the constrained
            # grounding pass against real CSV rows before proposals reach the UI.
            llm_kw = [str(k).strip() for k in (rec.get('kata_kunci') or []) if str(k).strip()]
            valid_recs.append({
                'nama': raw_name,
                'tipe': tipe,
                'deskripsi': str(rec.get('deskripsi', '')).strip()[:300] or f"Pos {tipe} untuk {raw_name}",
                'contoh': contoh_list,
                'kata_kunci': suggest_default_keywords(
                    raw_name, tipe, business_type=business_type, contoh=contoh_list, llm_keywords=llm_kw
                ),
                'source': 'ai_discovery',
                'discovery_keywords': llm_kw,
            })
            if len(valid_recs) >= max_count:
                break

        return valid_recs

    except Exception as exc:
        raise LLMUnavailable('Penemuan kategori memerlukan Ollama llama3:8b. ' + str(exc)) from exc


def ground_discovered_categories(recommendations, samples, tipe, business_type):
    """Assign actual upload examples to proposed definitions before keyword seeding.

    A second, constrained classification pass prevents a discovery model's sparse
    or mislabeled examples from becoming authoritative keyword matches. Nothing is
    persisted; the owner still reviews every proposed category.
    """
    if not recommendations:
        return []
    candidates = [dict(id=f'proposal_{i}', nama=r['nama'], tipe=tipe,
                       deskripsi=r['deskripsi'], sistem=False, kata_kunci=[])
                  for i, r in enumerate(recommendations)]
    grouped = {c['id']: [] for c in candidates}
    items = list(enumerate([raw for _, raw, _ in samples], 1))
    from ai.category_evidence import resolve_sample_owner, compile_keyword_plan
    evidence = []
    for ident, raw in items:
        # A separate short purpose task avoids feeding unreliable discovery
        # keywords back into classification as if the owner had confirmed them.
        decision = (classify(tipe, raw, candidates, business_type=business_type)
                    if len(candidates) == 1 else
                    resolve_sample_owner(tipe, raw, candidates, business_type))
        cid = decision.get('kategori_id')
        if cid in grouped and raw not in grouped[cid]:
            grouped[cid].append(raw)
            evidence.append({'tipe': tipe, 'deskripsi': raw, 'kategori_id': cid})
    seeded = [{**c, 'kata_kunci': r.get('discovery_keywords', [])}
              for c, r in zip(candidates, recommendations)]
    plan = compile_keyword_plan(seeded, evidence, business_type)
    result = []
    for category, rec in zip(plan['categories'], recommendations):
        examples = grouped[category['id']]
        if examples:
            result.append({**rec, 'contoh': examples, 'kata_kunci': category['kata_kunci'],
                           'keyword_audit': [a for a in plan['audit'] if a['category_id'] == category['id']]})
    if not result:
        raise LLMUnavailable('Tidak ada usulan kategori yang dapat dipetakan ke CSV. Tentukan kategori manual atau coba ulang.')
    return result


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
                coverage.update({desc: decisions[ident]['source'] in ('llm', 'similarity') and bool(decisions[ident]['kategori_id'])
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
            result_recommendations[tp] = ground_discovered_categories(
                discovered[:max_recommendations_per_type], uncovered_samples, tp, business_type
            )
        else:
            result_recommendations[tp] = []

    return {
        'pengeluaran': result_recommendations['pengeluaran'],
        'pemasukan': result_recommendations['pemasukan'],
        'coverage_stats': coverage_stats,
    }

