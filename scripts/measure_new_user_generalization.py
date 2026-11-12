"""Read-only evidence for new user categories; no training on held-out cases."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.category_discovery import suggest_default_keywords
from ai.classifier import classify, _similarity_evaluate
from ai.hybrid_classifier import clear_classification_cache
from ai.rag_engine import build_and_save_index, clear_memory_cache
from measure_retail_before import _build_temp_faiss_entries


def evaluate_new_user_data():
    data = json.loads((ROOT / 'tests/fixtures/new_user_categories.json').read_text(encoding='utf-8'))
    heldout = {c['deskripsi'] for c in data['cases']}
    retail_reference = json.loads((ROOT / 'tests/fixtures/retail_22_descriptions.json').read_text(encoding='utf-8'))
    bakery_reference = json.loads((ROOT / 'tests/fixtures/bakery_16_descriptions.json').read_text(encoding='utf-8'))
    assert heldout.isdisjoint({r['deskripsi'] for r in retail_reference})
    assert heldout.isdisjoint(set(bakery_reference))
    for path in [Path.home() / 'Downloads/data_sintetis_retail_365hari.csv', Path.home() / 'Downloads/data_sintetis_bakery_365hari.csv']:
        import csv
        if path.exists():
            assert heldout.isdisjoint({r['deskripsi'] for r in csv.DictReader(path.open(encoding='utf-8-sig'))})
    assert heldout.isdisjoint({s for c in data['categories'] for s in c['contoh']})
    cats = [{**c, 'kata_kunci': suggest_default_keywords(c['nama'], c['tipe'], c['domain'], contoh=c['contoh'])}
            for c in data['categories']]
    lookup = {c['id']: c for c in cats}
    results = []
    with tempfile.TemporaryDirectory() as temp, patch('ai.rag_engine.INDEX_ROOT_DIR', temp), patch('ai.rag_engine.get_remote_version', return_value=1):
        for domain in sorted({c['domain'] for c in cats}):
            shop = 'new-user-evidence-' + domain
            selected = [c for c in cats if c['domain'] == domain]
            clear_memory_cache(shop)
            clear_classification_cache(shop)
            for tp in ('pengeluaran', 'pemasukan'):
                entries = _build_temp_faiss_entries(shop, [c for c in selected if c['tipe'] == tp], domain)
                build_and_save_index(shop, tp, entries=entries, version=1)
            for case in data['cases']:
                expected = lookup[case['expected']]
                if expected['domain'] != domain:
                    continue
                tp = expected['tipe']
                _, diag = _similarity_evaluate(tp, case['deskripsi'], selected, toko_id=shop)
                result = classify(tp, case['deskripsi'], selected, toko_id=shop, business_type=domain)
                results.append({**case, 'domain': domain, 'tipe': tp, 'predicted': result['kategori_id'],
                                'correct': result['kategori_id'] == case['expected'],
                                'source': result['source'], 'gate': diag})
    return results


if __name__ == '__main__':
    results = evaluate_new_user_data()
    output = ROOT / 'evidence/retail_fix/generalization.json'
    output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'total':len(results),'correct':sum(r['correct'] for r in results), 'unresolved':sum(r['predicted'] is None for r in results)},indent=2))
