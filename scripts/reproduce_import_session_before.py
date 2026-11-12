"""Execute the original approval block against a new isolated in-memory shop.

No production database is contacted. Keeps the exact original UI mutation code.
"""
import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'evidence/import_session_fix'


def reproduce():
    source = (OUT / 'import_view_before.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    approval = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'btn_approve and (not missing_types)')
    cats, tx = [], []
    before = dict(kategori=copy.deepcopy(cats), transaksi=copy.deepcopy(tx))
    def create(token, name, tipe, **kwargs):
        cats.append(dict(id=str(len(cats)), nama=name, tipe=tipe, **kwargs))
    state = {}
    st = MagicMock(session_state=state)
    finance = SimpleNamespace(list_categories=lambda token: copy.deepcopy(cats),
                              create_category=create, auto_fill_category_keywords=lambda *a, **k: None)
    scope = dict(st=st, finance=finance, token='test-only', btn_approve=True, missing_types=[],
                 all_reviewed_exp=[dict(nama='Operasional', tipe='pengeluaran')],
                 all_reviewed_inc=[], fill_existing_kw=False, rows_to_import=[])
    from services.import_category_policy import missing_category_types
    scope['missing_category_types'] = missing_category_types
    wrapped = ast.parse('def run():\n    pass')
    wrapped.body[0].body = [approval]
    exec(compile(ast.fix_missing_locations(wrapped), '<original-approval>', 'exec'), scope)
    scope['run']()
    # Removing the upload clears session keys, but cannot undo database writes.
    state.clear()
    after = dict(kategori=copy.deepcopy(cats), transaksi=copy.deepcopy(tx))
    assert len(after['kategori']) == 4 and not after['transaksi']
    # Removing a recommendation changes only its review list, not either cache.
    state.update(csv_category_approved=True, csv_approved_categories=[{'id': 'deleted'}],
                 csv_proposals=[{'kategori_id': 'deleted'}], review_expense_cats=[{'id': 'deleted'}])
    state['review_expense_cats'].pop(0)
    assert state['csv_proposals'][0]['kategori_id'] == 'deleted'
    result = dict(shop='new-local-test-shop', mode='in-memory, original approval code',
                  before=before, after_cancel=after, deletion_stale_cache=state)
    (OUT / 'reproduction_before.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    reproduce()
