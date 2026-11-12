"""Compile store/CSV evidence into category keywords without changing category IDs.

Ownership is transaction-type scoped. Confirmed history and explicitly mapped CSV
rows outrank template words. An unresolved collision is removed from the keyword
view, never resolved by weakening the classification gate or choosing a tie.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from ai.ollama_client import ask_json
from ai.text_normalizer import clean_text_for_storage, normalize


def bakery_fuel_owner(description, categories, business_type):
    """Return (applicable, unique compatible ID) for the owner's fuel policy.

    Source: owner's bug report, 12 Nov 2026, section 2: LPG in a bakery is
    operational fuel, never an ingredient. Category-role aliases are taken from
    data/knowledge/category_keyword_profiles.json. No CSV phrase is a default.
    Multiple fuel/utility roles abstain rather than breaking a similarity tie.
    Retail resale is deliberately outside this bakery-only policy.
    """
    from ai.category_discovery import _canonical_domain, _reference_profiles
    if _canonical_domain(business_type) != 'bakery' or not re.search(r'\b(?:gas|lpg|elpiji)\b', normalize(description)):
        return False, None
    allowed = [c for c in categories if c.get('tipe') == 'pengeluaran' and not c.get('sistem')]
    fuel = [c['id'] for c in allowed if re.search(r'\b(?:gas|lpg|elpiji|bahan bakar)\b', normalize(c['nama']))]
    aliases = [a for g in _reference_profiles().get('bakery', [])
               if 'operasional' in g['aliases'] for a in g['aliases']]
    candidates = fuel or [c['id'] for c in allowed if any(a in normalize(c['nama']) for a in aliases)]
    return True, candidates[0] if len(candidates) == 1 else None


def reconcile_purpose_rows(rows, categories, business_type):
    """Correct only the retrieval/planning view; saved tenant rows stay intact."""
    output = []
    for row in rows:
        applies, owner = bakery_fuel_owner(row.get('deskripsi', ''), categories, business_type)
        output.append({**row, 'kategori_id': owner} if applies and row.get('tipe') == 'pengeluaran' else dict(row))
    return output


def phrase_key(text):
    text = normalize(text)
    text = re.sub(r'^(?:beli|bayar|biaya|belanja|pembayaran|pembelian|stok|kulakan)\s+', '', text)
    return ' '.join(re.sub(r'\b(?:kg|gr|gram|liter|lt|ml|pcs|dus|box|pak|pack|galon|slop|lusin)\b', '', text).split())


def business_reference_context(description, business_type):
    """Quote relevant shipped guide paragraphs, without a product lookup table.

    Source paths come from the cited profiles in data/knowledge. This is only
    context for proposed assignments, never a category ID or gate override.
    """
    from ai.category_discovery import _canonical_domain, _reference_profiles
    ignored = {'beli', 'bayar', 'biaya', 'belanja', 'pembelian', 'pembayaran', 'untuk', 'yang', 'dari', 'dengan'}
    terms = {t for t in normalize(description).split() if len(t) >= 4 and t not in ignored}
    root = Path(__file__).resolve().parents[1]
    paths = {p for group in _reference_profiles().get(_canonical_domain(business_type), [])
             for p in group['sources'] if p.endswith('.md')}
    ranked = []
    for source in sorted(paths):
        for line in (root / source).read_text(encoding='utf-8').splitlines():
            if not line.startswith('- ') or 'http' in line:
                continue
            overlap = terms & set(normalize(line).split())
            if overlap:
                ranked.append((len(overlap), source, line))
    return [{'source': source, 'text': line[:1000]}
            for _, source, line in sorted(ranked, key=lambda x: -x[0])[:3]]


def resolve_sample_owner(tipe, description, categories, business_type):
    """A small purpose-only discovery task, not a similarity-gate decision.

    These are proposed examples for the owner's category review, not permission
    to create categories. No product names or synthetic descriptions live here.
    """
    applies, owner = bakery_fuel_owner(description, categories, business_type)
    if applies and tipe == 'pengeluaran':
        return {'kategori_id': owner, 'reason': 'Owner policy: bakery fuel is operational, not an ingredient'}
    allowed = [c for c in categories if c.get('tipe') == tipe
               and not c.get('sistem') and str(c.get('nama', '')).casefold() != 'others']
    # Compact ordinal choices avoid making a small local model copy long UUIDs.
    choices = [{'id': str(i), 'name': c['nama'], 'definition': c.get('deskripsi', ''),
                'owner_examples': (c.get('contoh') or c.get('kata_kunci') or [])[:6]}
               for i, c in enumerate(allowed)]
    if not normalize(description):
        return {'kategori_id': None, 'reason': 'Deskripsi kosong'}
    if not choices:
        return {'kategori_id': None, 'reason': 'Tidak ada kategori yang tersedia'}
    # Last-resort domain evidence is limited to words actually in this upload.
    # The merchandise lists are read from cited guide paragraphs, not embedded
    # here. Explicit internal-use statements and category ambiguities still go
    # to the purpose task; a profile never creates a new allowed category.
    from ai.category_discovery import _canonical_domain, _reference_profiles
    text = normalize(description)
    if (tipe == 'pengeluaran' and text.split()[0] in ('beli', 'belanja', 'kulakan', 'stok', 'pembelian')
            and not re.search(r'\b(?:untuk|buat|dipakai|digunakan|kebutuhan)\b', text)):
        matched = []
        for group in _reference_profiles().get(_canonical_domain(business_type), []):
            terms = [term for term in group.get('grounding_terms', [])
                     if f' {normalize(term)} ' in f' {text} ']
            if group.get('tipe') == tipe and terms:
                role_words = set(' '.join(normalize(a) for a in group['aliases']).split())
                role_words.update(('dan', 'atau', 'biaya', 'pembelian', 'toko'))
                # A qualified owner category (e.g. stock of a specific product
                # family) is not interchangeable with a general inventory role.
                # Its definition must be considered by the purpose task instead.
                ids = [c['id'] for c in allowed
                       if any(alias in normalize(c['nama']) for alias in group['aliases'])
                       and set(normalize(c['nama']).split()) <= role_words]
                if len(ids) == 1:
                    matched.append((ids[0], terms, group['grounding_sources']))
        if len({m[0] for m in matched}) == 1:
            return {'kategori_id': matched[0][0], 'reason': 'Matching uploaded product in cited business guide',
                    'reference_terms': matched[0][1], 'references': matched[0][2]}
    schema = {'type': 'object', 'properties': {
        'reason': {'type': 'string'},
        'category_id': {'type': 'string', 'enum': ['', *[c['id'] for c in choices]]},
    }, 'required': ['reason', 'category_id']}
    system = (
        'Classify the economic PURPOSE of a transaction using the supplied owner category definitions. '
        'Use only a supplied category_id, or empty if none fits. Explain briefly before choosing. '
        'Utilities, staff wages, marketing, packaging and order delivery are different purposes. '
        'Delivery of customer orders is transportation, not staff wages. '
        'If a specific cost category is absent, use a compatible general operating-cost category; '
        'never classify a service fee as merchandise. '
        'For income distinguish product sales, service income and owner capital. Do not add categories. '
    )
    # Domain semantics, not a product lookup. The distinction is documented in
    # data/knowledge/retail/panduan_operasional_retail.md, sections 2, 3 and 5.
    if _canonical_domain(business_type) == 'retail' and tipe == 'pengeluaran':
        system += (
            'For a trading store, products purchased without explicit internal-use information '
            'are presumed for resale, not production or operating fuel. '
            'Wrapping used to hand purchases to customers is packaging/operating supplies, '
            'unless the description explicitly says the wrapping itself is being sold. '
        )
        packaging_aliases = [a for g in _reference_profiles().get('retail', [])
                             if 'kemasan' in g['aliases'] for a in g['aliases']]
        has_packaging_category = any(
            any(alias in normalize(c['nama'] + ' ' + (c.get('deskripsi') or ''))
                for alias in packaging_aliases) for c in allowed)
        if not has_packaging_category:
            # Owner's retail import policy: consumable packaging is operating
            # expense unless the owner supplied an explicit packaging category.
            # Generic furniture/equipment does not imply a packaging category.
            system += (
                'No explicit packaging category has been supplied. Classify store wrapping costs '
                'as operating expenses, not the general furniture/equipment category. '
            )
    reply = ask_json(system, json.dumps({'business': business_type, 'type': tipe,
                     'transaction': description, 'choices': choices,
                     'business_references': business_reference_context(description, business_type)}, ensure_ascii=False),
                     schema, num_predict=256)
    ident = reply.get('category_id', '')
    valid = {str(i): c for i, c in enumerate(allowed)}
    return {'kategori_id': valid[ident]['id'] if ident in valid else None,
            'reason': str(reply.get('reason', ''))}


def compile_keyword_plan(categories, evidence_rows, business_type='Umum', prefer_existing=False):
    """Pure, reviewable plan shared by auto-fill, import and corpus building.

    evidence_rows must already be tenant-scoped and owner-confirmed, or assigned
    to proposed categories during discovery. This function does not query or
    mutate a store, infer new category IDs, or break ambiguous history ties.
    """
    from ai.category_discovery import suggest_default_keywords
    cats = [dict(c) for c in categories]
    evidence_rows = reconcile_purpose_rows(evidence_rows, cats, business_type)
    allowed = {str(c['id']): c for c in cats if not c.get('sistem')
               and str(c.get('nama', '')).casefold() != 'others'}
    samples = defaultdict(list)
    owners = defaultdict(set)
    priority = {}
    for row in evidence_rows:
        cid = str(row.get('kategori_id') or '')
        cat = allowed.get(cid)
        if not cat or row.get('tipe', cat.get('tipe')) != cat.get('tipe'):
            continue
        raw = clean_text_for_storage(row.get('deskripsi', ''))
        if raw:
            key = (cat['tipe'], phrase_key(raw))
            rank = row.get('_source_priority', 1)
            if rank > priority.get(key, -1):
                owners[key] = {cid}
                priority[key] = rank
            elif rank == priority[key]:
                owners[key].add(cid)
    for row in sorted(evidence_rows, key=lambda r: -r.get('_sample_priority', 0)):
        cid = str(row.get('kategori_id') or '')
        cat = allowed.get(cid)
        raw = clean_text_for_storage(row.get('deskripsi', ''))
        if cat and row.get('tipe', cat['tipe']) == cat['tipe'] and raw:
            if owners.get((cat['tipe'], phrase_key(raw))) == {cid} and raw not in samples[cid]:
                samples[cid].append(raw)
    candidates = {}
    for cid, cat in allowed.items():
        # During index rebuild, retain the owner's stored 20-keyword coverage.
        # All historical examples are already indexed as transaction documents;
        # reseeding them into this limited bank would evict new CSV examples.
        candidates[cid] = suggest_default_keywords(cat['nama'], cat['tipe'], business_type,
                           contoh=[] if prefer_existing else samples[cid], llm_keywords=cat.get('kata_kunci'))
    claims = defaultdict(set)
    for cid, kws in candidates.items():
        for kw in kws:
            claims[(allowed[cid]['tipe'], phrase_key(kw))].add(cid)
    audit, warnings = [], []
    for cid, cat in allowed.items():
        cleaned = []
        for kw in candidates[cid]:
            applies, owner = bakery_fuel_owner(kw, cats, business_type)
            if applies and cat.get('tipe') == 'pengeluaran' and owner != cat['id']:
                audit.append({'category_id': cat['id'], 'keyword': kw, 'reason': 'owner_fuel_policy'})
                continue
            key = (cat['tipe'], phrase_key(kw))
            # A shorter keyword can be owned by the uniquely confirmed longer
            # example. E.g. the same delivery token in wages and transport must
            # not create two score-1 vectors when history owns it in transport.
            supported = set()
            for (tp, sample), ids in owners.items():
                if tp == cat['tipe'] and key[1] and (key[1] == sample or f' {key[1]} ' in f' {sample} '):
                    supported.update(ids)
            if supported:
                keep = supported == {cid}
                reason = 'unique_evidence_owner' if len(supported) == 1 else 'conflicting_evidence'
            else:
                peers = claims[key]
                # Disambiguate subtype words supplied in category names, e.g.
                # two owner categories sharing a broad sales reference profile.
                name_words = {peer: set(normalize(allowed[peer]['nama']).split()) for peer in peers}
                common = set.intersection(*name_words.values()) if name_words else set()
                matching = {peer for peer, words in name_words.items()
                            if (words - common) & set(normalize(kw).split())}
                if len(matching) == 1:
                    keep, reason = cid in matching, 'category_subtype'
                else:
                    keep, reason = len(peers) == 1, 'ambiguous_keyword'
            if keep:
                cleaned.append(kw)
            else:
                audit.append({'category_id': cat['id'], 'keyword': kw, 'reason': reason,
                              'evidence_owners': sorted(supported)})
        cat['kata_kunci'] = cleaned[:20]
        cat['contoh'] = samples[cid]
        if not cleaned and not cat.get('bukan_operasional'):
            warnings.append(f"Kategori {cat['nama']} belum mempunyai kata kunci unik; isi atau tinjau melalui Kelola Kategori.")
    return {'categories': cats, 'audit': audit, 'warnings': warnings}
