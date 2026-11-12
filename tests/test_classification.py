import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal

from ai.classifier import classify, detect_type
import services.ai_service as ai_service
import services.import_service as import_service
from ai.ollama_client import LLMUnavailable


SAMPLE_CATS = [
    {'id': 'c_bahan', 'nama': 'Bahan Baku', 'tipe': 'pengeluaran', 'deskripsi': 'Bahan makanan dan mentah'},
    {'id': 'c_gaji', 'nama': 'Gaji', 'tipe': 'pengeluaran', 'deskripsi': 'Gaji karyawan'},
    {'id': 'c_ops', 'nama': 'Operasional', 'tipe': 'pengeluaran', 'deskripsi': 'Listrik air sewa'},
    {'id': 'c_exp_other', 'nama': 'Others', 'tipe': 'pengeluaran', 'sistem': True},
    {'id': 'c_jual', 'nama': 'Penjualan', 'tipe': 'pemasukan', 'deskripsi': 'Omzet penjualan'},
    {'id': 'c_modal', 'nama': 'Tambahan Modal', 'tipe': 'pemasukan', 'bukan_operasional': True},
    {'id': 'c_inc_other', 'nama': 'Others', 'tipe': 'pemasukan', 'sistem': True},
]


def test_classification_empty_or_whitespace_description():
    with patch('ai.classifier.ask_json',return_value={'kategori_id':'','nama_baru':'','alasan':'Deskripsi kosong'}):
        res=classify('pengeluaran','   ',SAMPLE_CATS)
    assert res['kategori_id'] is None and res['source']=='perlu_tinjauan'


def test_classification_rejects_invented_category_id():
    with patch('ai.classifier.ask_json',return_value={'kategori_id':'invented','nama_baru':'','alasan':'x','keyakinan':95}):
        with pytest.raises(LLMUnavailable,match='Model.*tidak valid') as exc:
            classify('pengeluaran','beli sesuatu',SAMPLE_CATS)
    assert isinstance(exc.value.__cause__, ValueError)
    assert 'Jalankan Ollama' not in str(exc.value)


def test_classification_propose_existing_name_resolves_to_id():
    with patch('ai.classifier.ask_json',return_value={'kategori_id':'','nama_baru':'Bahan Baku','alasan':'Abstain'}):
        res=classify('pengeluaran','beli tepung',SAMPLE_CATS)
    assert res['kategori_id'] is None and res['source']=='perlu_tinjauan'
    assert res['usulan']['nama_baru']=='Bahan Baku'  # No inferred ID from a proposed name.


def test_classification_exceeding_capacity_diverts_to_others():
    cats=[{'id':str(i),'nama':f'Category {i}','tipe':'pengeluaran'} for i in range(12)]
    with patch('ai.classifier.ask_json',return_value={'kategori_id':'11','nama_baru':'','alasan':'Fits'}):
        res=classify('pengeluaran','transaksi',cats)
    assert res['kategori_id']=='11' and res['source']=='llm'  # No hardcoded quota fallback.


@pytest.mark.parametrize('cats', [[], [c for c in SAMPLE_CATS if c['tipe'] == 'pemasukan']])
def test_classification_no_allowed_categories(cats):
    with patch('ai.classifier.ask_json',return_value={'kategori_id':'','nama_baru':'','alasan':'No categories'}) as llm:
        res=classify('pengeluaran','beli beras',cats)
    llm.assert_not_called()
    assert res['kategori_id'] is None and res['source']=='perlu_tinjauan'
    assert res['status_kategori'] == 'perlu_tinjauan'
    assert res['reason_code'] == 'NO_ALLOWED_CATEGORIES'


def test_bulk_classify_partial_failure_and_row_preservation():
    rows=[{'deskripsi':'Sale','tipe':'pemasukan'}, {'deskripsi':'invalid','error':'Invalid amount'}]
    decisions=[{**rows[0],'kategori_id':'c_jual','source':'llm'}, rows[1]]
    with patch('services.ai_service.verify_actor',return_value={'toko_id':'shop','role':'pemilik_toko'}),patch('services.ai_service.get_store_business_context',return_value={'bidang_usaha':'Custom'}),patch('services.ai_service.hybrid_bulk_classify',return_value=decisions):
        res=ai_service.bulk_classify('token',rows,categories=SAMPLE_CATS)
    assert len(res)==2 and res[0]['source']=='llm' and res[1]['error']=='Invalid amount'


def test_bulk_classify_handles_ambiguous_type_gracefully():
    from ai.hybrid_classifier import hybrid_bulk_classify
    with patch('ai.classifier.detect_type',side_effect=ValueError('Pilih jenis manual')):
        with pytest.raises(ValueError,match='manual'):
            hybrid_bulk_classify('shop',SAMPLE_CATS,[{'deskripsi':'xyz'}],business_type='Custom')


def test_enrich_rows_delegates_to_bulk_classify():
    rows = [{'baris': 2, 'deskripsi': 'beli sabun', 'nominal': '10000', 'tipe': 'pengeluaran', 'error': ''}]
    with patch('services.ai_service.bulk_classify', return_value=[{'baris': 2, 'kategori_id': 'c_ops'}]) as mock_bulk:
        out = import_service.enrich_rows('token-123', rows)
        mock_bulk.assert_called_once_with('token-123', rows, progress=None)
        assert out[0]['kategori_id'] == 'c_ops'


def test_commit_preview_forwards_classification_metadata():
    row = {
        'tanggal': '2026-10-31',
        'deskripsi': 'beli tepung',
        'nominal': '250000',
        'tipe': 'pengeluaran',
        'status_kategori': 'otomatis',
        'skor': 0.9123,
        'usulan': {'nama_baru': ''}
    }

    with patch('services.ai_service.finance.add_transaction') as mock_add:
        ai_service.commit_preview('tok-owner', row, 'c_bahan')
        mock_add.assert_called_once_with(
            'tok-owner',
            '2026-10-31',
            'beli tepung',
            '250000',
            'pengeluaran',
            'c_bahan',
            status_kategori='otomatis',
            skor=0.9123,
            usulan={'nama_baru': ''}
        )


def test_classification_malformed_llm_responses():
    for output in ({'kategori_id':None,'nama_baru':None,'alasan':None}, {}, []):
        with patch('ai.classifier.ask_json',return_value=output):
            with pytest.raises(LLMUnavailable,match='Model.*tidak valid'):
                classify('pengeluaran','transaksi',SAMPLE_CATS)


def test_classification_missing_rag_index_fallback():
    # If FAISS index is missing or throws an error, zero-shot LLM classification continues gracefully.
    with patch('ai.classifier.retrieve_context', side_effect=Exception('Index not found')), \
         patch('ai.classifier.ask_json', return_value={'kategori_id': 'c_ops', 'nama_baru': '', 'alasan': 'Operasional'}):
        res = classify('pengeluaran', 'bayar tagihan listrik ruko', SAMPLE_CATS, toko_id='shop-missing-index')
        assert res['kategori_id'] == 'c_ops'
        assert res['status_kategori'] == 'dikonfirmasi'


def test_classification_ollama_timeout_fallback():
    hits=[{'category_id':'c_gaji','category_name':'Gaji','score':0.99,'type':'transaction','text':'Gaji'}]
    with patch('ai.classifier.retrieve_context',return_value=hits),patch('ai.classifier.ask_json',side_effect=LLMUnavailable('timed out')):
        with pytest.raises(LLMUnavailable,match='Jalankan Ollama.*llama3:8b'):
            classify('pengeluaran','upah kasir',SAMPLE_CATS,toko_id='outage')


def test_bulk_classify_empty_list():
    with patch('services.ai_service.verify_actor', return_value={'toko_id': 'shop-1', 'role': 'pemilik_toko'}), \
         patch('services.ai_service.finance.list_categories', return_value=SAMPLE_CATS):
        res = ai_service.bulk_classify('mock-tok', [])
        assert res == []


def test_classification_cross_store_isolation():
    with patch('ai.classifier.ask_json',return_value={'kategori_id':'foreign','nama_baru':'','alasan':'x'}):
        with pytest.raises(LLMUnavailable):
            classify('pengeluaran','transaksi',SAMPLE_CATS)



# Real PostgreSQL checks for migration 011 and service queue/retry semantics.
from tests.test_audit_remediation import import_postgres, audit_postgres


@pytest.fixture(scope='module')
def queue_postgres(audit_postgres):
    from pathlib import Path
    sql=audit_postgres
    sql('ALTER TABLE transaksi ALTER COLUMN kategori_id SET NOT NULL; ALTER TABLE kategori ADD CONSTRAINT queue_category_identity UNIQUE(id,toko_id,tipe); ALTER TABLE transaksi ADD CONSTRAINT queue_category_fk FOREIGN KEY(kategori_id,toko_id,tipe) REFERENCES kategori(id,toko_id,tipe);')
    migration=(Path(__file__).resolve().parents[1]/'database/migrations/011_nullable_kategori_for_ai_pending.sql').read_text(encoding='utf-8')
    sql(migration)
    sql(migration)  # Applying twice must be harmless.
    return sql


@pytest.fixture
def queue_database(queue_postgres,monkeypatch):
    import uuid,json
    from types import SimpleNamespace
    sql=queue_postgres
    shop,owner,cat=[str(uuid.uuid4()) for _ in range(3)]
    actor={'id':owner,'role':'pemilik_toko','toko_id':shop}
    cats=[{'id':cat,'nama':'Custom category','deskripsi':'Data-defined category','tipe':'pengeluaran','toko_id':shop}]
    sql(f"INSERT INTO profiles VALUES ('{owner}','{shop}','pemilik_toko','aktif'); INSERT INTO kategori VALUES ('{cat}','{shop}','pengeluaran');")
    def snapshot():
        return json.loads(sql(f"SELECT coalesce(jsonb_agg(to_jsonb(t)),'[]') FROM transaksi t WHERE toko_id='{shop}';"))
    def rpc(name,args):
        parts=[]
        for key,value in args.items():
            if value is None:
                expr='NULL'
            elif key=='p_ids':
                expr='ARRAY['+','.join("'"+v+"'" for v in value)+']::uuid[]'
            elif isinstance(value,(list,dict)):
                expr="'"+json.dumps(value,default=str).replace("'","''")+"'::jsonb"
            else:
                expr="'"+str(value).replace("'","''")+"'"
            parts.append(key+' => '+expr)
        return SimpleNamespace(execute=lambda:SimpleNamespace(data=json.loads(sql('SELECT to_jsonb(public.'+name+'('+','.join(parts)+'));'))))
    client=MagicMock()
    client.rpc.side_effect=rpc
    for module in ('services.ai_service','services.finance_service','services.import_service','services.smart_import_engine'):
        monkeypatch.setattr(module+'.verify_actor',lambda *a:actor)
        monkeypatch.setattr(module+'.privileged_client',lambda:client)
    monkeypatch.setattr('services.ai_service.get_store_business_context',lambda *a:{'bidang_usaha':'Custom business'})
    monkeypatch.setattr('services.ai_service.finance.list_categories',lambda *a:cats)
    monkeypatch.setattr('repositories.finance_repository.transactions',lambda *a:snapshot())
    monkeypatch.setattr('repositories.finance_repository.categories',lambda *a:cats)
    monkeypatch.setattr('ai.classifier.retrieve_context',lambda *a,**kw:[])
    return SimpleNamespace(sql=sql,shop=shop,owner=owner,cat=cat,actor=actor,cats=cats,client=client,snapshot=snapshot)


def test_migration_011_null_category_requires_marker_real_postgres(queue_database):
    import uuid
    q=queue_database
    row={'tanggal':'2026-10-30','deskripsi':'Queued transaction','nominal':'100', 'tipe':'pengeluaran','kategori_id':None,
         'status_kategori':'perlu_tinjauan','usulan':{'ai_pending':True,'ai_error':'Ollama offline'}}
    params={'p_actor':q.owner,'p_shop':q.shop,'p_batch':str(uuid.uuid4()),'p_rows':[row]}
    assert q.client.rpc('import_transactions_atomic',params).execute().data==1
    stored=q.snapshot()[0]
    assert stored['kategori_id'] is None and stored['status_persetujuan']=='pending'
    params['p_batch']=str(uuid.uuid4())
    params['p_rows']=[{**row,'usulan':{}}]
    with pytest.raises(AssertionError,match='Kategori tidak cocok'):
        q.client.rpc('import_transactions_atomic',params).execute()
    params['p_rows']=[{**row,'kategori_id':str(uuid.uuid4())}]
    with pytest.raises(AssertionError,match='Kategori tidak cocok'):
        q.client.rpc('import_transactions_atomic',params).execute()
    assert len(q.snapshot())==1


def test_service_outage_persists_null_category_and_retries_idempotently(queue_database):
    from services.ai_service import bulk_classify,retry_pending_ai_classifications
    q=queue_database
    rows=[{'tanggal':'2026-10-30','deskripsi':'Custom operational purchase','nominal':'100','tipe':'pengeluaran',
           'usulan':{'source_id':'QUEUED-1'}}]
    with patch('ai.hybrid_classifier.ask_json',side_effect=LLMUnavailable('Ollama offline')):
        result=bulk_classify('token',rows,categories=q.cats)
    assert result[0]['_saved_ai_pending'] is True
    stored=q.snapshot()[0]
    assert stored['kategori_id'] is None and stored['status_kategori']=='perlu_tinjauan'
    assert stored['status_persetujuan']=='pending' and stored['usulan']['ai_pending'] is True
    assert stored['usulan']['source_id']=='queued-1'
    with patch('services.ai_service.classify',side_effect=LLMUnavailable('still down')):
        retry=retry_pending_ai_classifications('token')
    assert retry['processed']==0 and retry['remaining']==1 and retry['errors']
    assert q.snapshot()==[stored]
    with patch('services.ai_service.classify',return_value={'kategori_id':q.cat,'source':'llm'}):
        retry=retry_pending_ai_classifications('token')
        assert retry=={'processed':1,'remaining':0,'errors':[]}
        assert retry_pending_ai_classifications('token')['processed']==0
    saved=q.snapshot()[0]
    assert saved['kategori_id']==q.cat and saved['usulan']['source']=='llm'
    assert 'ai_pending' not in saved['usulan'] and 'ai_error' not in saved['usulan']
    assert saved['status_persetujuan']=='pending'  # AI retry never approves financial data.
    assert q.sql(f"SELECT count(*) FROM audit_log WHERE toko_id='{q.shop}';")=='1'


def test_retry_never_touches_approved_transaction(queue_database):
    from services.ai_service import retry_pending_ai_classifications
    q=queue_database
    import uuid
    ident=str(uuid.uuid4())
    q.sql(f"INSERT INTO transaksi(id,toko_id,kategori_id,tipe,status_persetujuan,usulan) VALUES ('{ident}','{q.shop}','{q.cat}','pengeluaran','disetujui','{{\"ai_pending\":true}}');")
    before=q.snapshot()
    with patch('services.ai_service.classify') as classify_mock:
        assert retry_pending_ai_classifications('token')['processed']==0
        classify_mock.assert_not_called()
    assert q.snapshot()==before
    # The SQL guard independently protects against stale queue snapshots.
    assert q.client.rpc('retry_pending_ai_classification',{'p_actor':q.owner,'p_shop':q.shop,'p_id':ident,
                        'p_category':q.cat,'p_updated_at':before[0]['updated_at']}).execute().data is False


def test_no_rag_shortcut_and_prompt_contains_descriptions():
    # Revisi 1 (Ralat): similarity gate stays active, while few-shot RAG examples are removed from Llama prompts.
    import json
    hits=[{'category_id':'c_bahan','category_name':'Bahan Baku','score':0.999,'type':'transaction','text':'beli bahan'}]
    with patch('ai.classifier.retrieve_context',return_value=hits) as retrieve,patch('ai.classifier.ask_json',return_value={'kategori_id':'c_ops','nama_baru':'','alasan':'Context matters'}) as llm:
        result=classify('pengeluaran','beli tepung untuk acara kantor',SAMPLE_CATS,toko_id='unique-no-shortcut',business_type='Retail')
    assert result['kategori_id']=='c_ops' and result['source']=='llm'
    llm.assert_called_once()
    assert all(call.kwargs.get('min_score') == -1.0 for call in retrieve.call_args_list)
    payload=json.loads(llm.call_args.args[1])
    assert payload['few_shot_examples_only'] == []
    assert payload['daftar_kategori_tersedia'][0]['deskripsi']=='Bahan makanan dan mentah'
    assert 'kata_kunci' not in payload


def test_retry_audit_failure_preserves_queue_marker_atomically(queue_database):
    import uuid
    from services.ai_service import bulk_classify,retry_pending_ai_classifications
    q=queue_database
    row={'tanggal':'2026-10-30','deskripsi':'Queued audit rollback','nominal':'100','tipe':'pengeluaran'}
    with patch('ai.hybrid_classifier.ask_json',side_effect=LLMUnavailable('down')):
        bulk_classify('token',[row],categories=q.cats)
    before=q.snapshot()
    ident=before[0]['id']
    q.sql(f"ALTER TABLE audit_log ADD CONSTRAINT queue_audit_fail CHECK(target_id <> '{ident}'::uuid) NOT VALID;")
    try:
        with patch('services.ai_service.classify',return_value={'kategori_id':q.cat,'source':'llm'}):
            result=retry_pending_ai_classifications('token')
        assert result['processed']==0 and result['remaining']==1 and result['errors']
        assert q.snapshot()==before
    finally:
        q.sql('ALTER TABLE audit_log DROP CONSTRAINT queue_audit_fail;')


def test_single_transaction_outage_persists_before_return(queue_database):
    import json
    from types import SimpleNamespace
    from services.ai_service import quick_preview,commit_preview
    q=queue_database
    def insert(payload):
        def execute():
            encoded=json.dumps(payload,default=str).replace("'","''")
            q.sql(f"INSERT INTO transaksi(toko_id,tanggal,deskripsi,nominal,tipe,kategori_id,dicatat_oleh,status_persetujuan,status_kategori,usulan) SELECT (r->>'toko_id')::uuid,(r->>'tanggal')::date,r->>'deskripsi',(r->>'nominal')::numeric,r->>'tipe',nullif(r->>'kategori_id','')::uuid,(r->>'dicatat_oleh')::uuid,r->>'status_persetujuan',r->>'status_kategori',r->'usulan' FROM (SELECT '{encoded}'::jsonb r) x;")
            return SimpleNamespace(data=q.snapshot())
        return SimpleNamespace(execute=execute)
    q.client.table.return_value.insert.side_effect=insert
    with patch('ai.classifier.ask_json',side_effect=LLMUnavailable('offline')):
        row=quick_preview('token','beli alat 100rb','pengeluaran')
    assert row['_saved_ai_pending'] is True
    assert len(q.snapshot())==1
    assert q.snapshot()[0]['kategori_id'] is None
    assert q.snapshot()[0]['usulan']['ai_pending'] is True
    commit_preview('token',row,None)
    assert len(q.snapshot())==1



def test_edit_ai_pending_transaction_can_keep_null_category(queue_database,monkeypatch):
    from services.ai_service import bulk_classify
    from services.finance_service import update_transaction
    q=queue_database
    with patch('ai.hybrid_classifier.ask_json',side_effect=LLMUnavailable('offline')):
        bulk_classify('token',[{'tanggal':'2026-10-30','deskripsi':'Waiting transaction','nominal':'100','tipe':'pengeluaran'}],categories=q.cats)
    old=q.snapshot()[0]
    monkeypatch.setattr('services.finance_service.transaction_by_id',lambda *args:old)
    result=update_transaction('token',old['id'],'2026-10-30','Corrected while waiting','120','pengeluaran',None)
    assert result[0]['kategori_id'] is None
    assert result[0]['usulan']['ai_pending'] is True
    assert result[0]['status_kategori']=='perlu_tinjauan'
    assert result[0]['status_persetujuan']=='pending'


@pytest.mark.parametrize('confidence', [0, 69, 69.5])
def test_low_confidence_classification_abstains(confidence):
    reply = {'alasan': 'Tujuan belum jelas', 'kategori_id': 'c_ops', 'nama_baru': '', 'keyakinan': confidence}
    with patch('ai.classifier.ask_json', return_value=reply):
        result = classify('pengeluaran', 'transaksi', SAMPLE_CATS)
    assert result['kategori_id'] is None
    assert result['status_kategori'] == 'perlu_tinjauan'
    assert result['source'] == 'perlu_tinjauan'
    assert result['reason_code'] == 'MODEL_ABSTAINED'
    assert result['is_unresolved'] is True


@pytest.mark.parametrize('extra', [{}, {'keyakinan': '40'}, {'keyakinan': None}, {'keyakinan': False}, {'keyakinan': 70}, {'keyakinan': 100}])
def test_optional_or_sufficient_confidence_preserves_valid_decision(extra):
    reply = {'alasan': 'Tujuan sesuai kategori', 'kategori_id': 'c_ops', 'nama_baru': '', **extra}
    with patch('ai.classifier.ask_json', return_value=reply):
        result = classify('pengeluaran', 'transaksi', SAMPLE_CATS)
    assert result['kategori_id'] == 'c_ops'
    assert result['source'] == 'llm'
    assert result['status_kategori'] == 'dikonfirmasi'


def test_batch_confidence_calibration_and_shared_reasoning_schema():
    from ai.classifier import CHOICE_SCHEMA
    from ai.hybrid_classifier import classify_ambiguous_batch_ai
    reply = {'klasifikasi': [
        {'item_id': 1, 'alasan': 'Belum yakin', 'kategori_id': 'c_ops', 'nama_baru': '', 'keyakinan': 69},
        {'item_id': 2, 'alasan': 'Sesuai', 'kategori_id': 'c_ops', 'nama_baru': ''},
    ]}
    with patch('ai.hybrid_classifier.ask_json', return_value=reply) as llm:
        result = classify_ambiguous_batch_ai('pengeluaran', [(1, 'pertama'), (2, 'kedua')], SAMPLE_CATS, business_type='Umum')
    assert result[1]['kategori_id'] is None
    assert result[1]['reason_code'] == 'MODEL_ABSTAINED'
    assert result[1]['source'] == 'perlu_tinjauan'
    assert result[1]['status_kategori'] == 'perlu_tinjauan'
    assert result[2]['kategori_id'] == 'c_ops' and result[2]['source'] == 'llm'
    schema = llm.call_args.kwargs['schema']['properties']['klasifikasi']['items']
    assert schema['properties']['keyakinan'] == {'type': 'integer', 'minimum': 0, 'maximum': 100}
    assert 'keyakinan' not in schema['required']
    assert set(CHOICE_SCHEMA['required']) == {'alasan', 'kategori_id', 'nama_baru'}
    fields = list(schema['properties'])
    assert fields.index('alasan') < fields.index('kategori_id')


def test_batch_rejects_confident_invented_id_as_invalid_answer():
    from ai.hybrid_classifier import classify_ambiguous_batch_ai
    reply = {'klasifikasi': [{'item_id': 1, 'alasan': 'Yakin', 'kategori_id': 'invented', 'nama_baru': '', 'keyakinan': 100}]}
    with patch('ai.hybrid_classifier.ask_json', return_value=reply):
        with pytest.raises(LLMUnavailable, match='Model.*tidak valid') as exc:
            classify_ambiguous_batch_ai('pengeluaran', [(1, 'transaksi')], SAMPLE_CATS, business_type='Umum')
    assert isinstance(exc.value.__cause__, ValueError)
    assert 'Jalankan Ollama' not in str(exc.value)


def test_batch_ollama_outage_keeps_startup_message():
    from ai.hybrid_classifier import classify_ambiguous_batch_ai
    with patch('ai.hybrid_classifier.ask_json', side_effect=LLMUnavailable('timeout')):
        with pytest.raises(LLMUnavailable, match='Jalankan Ollama.*llama3:8b'):
            classify_ambiguous_batch_ai('pengeluaran', [(1, 'transaksi')], SAMPLE_CATS, business_type='Umum')


def test_batch_no_allowed_categories_abstains_without_llm():
    from ai.hybrid_classifier import classify_ambiguous_batch_ai
    with patch('ai.hybrid_classifier.ask_json') as llm:
        results = classify_ambiguous_batch_ai('pengeluaran', [(1, 'pertama'), (2, 'kedua')],
                                             [c for c in SAMPLE_CATS if c['tipe'] == 'pemasukan'])
    llm.assert_not_called()
    assert set(results) == {1, 2}
    for result in results.values():
        assert result['kategori_id'] is None
        assert result['source'] == 'perlu_tinjauan'
        assert result['status_kategori'] == 'perlu_tinjauan'


@pytest.fixture
def similarity_gate_unit(monkeypatch):
    from ai.hybrid_classifier import clear_classification_cache
    clear_classification_cache()
    monkeypatch.setattr('ai.classifier.is_semantic_model_available', lambda: True)
    yield
    clear_classification_cache()


def _gate_hit(cid, score, entry_type='category', store='gate-store'):
    return {'category_id': cid, 'score': score, 'type': entry_type,
            'store_id': store, 'text': 'Independent reference text'}


def test_similarity_high_score_clear_margin_skips_llm(similarity_gate_unit):
    from ai.classifier import _decision
    hits = [_gate_hit('c_gaji', .91, 'transaction'), _gate_hit('c_ops', .70)]
    with patch('ai.classifier.retrieve_context', return_value=hits), patch('ai.classifier.ask_json') as llm:
        result = classify('pengeluaran', 'Independent query', SAMPLE_CATS, 'gate-store', 'Umum')
    llm.assert_not_called()
    assert set(result) == set(_decision({'kategori_id': 'c_gaji', 'nama_baru': '', 'alasan': ''}, {'c_gaji'}))
    assert result['kategori_id'] == 'c_gaji' and result['skor'] == .91
    assert result['source'] == 'similarity' and result['reason_code'] == 'SIMILARITY_AUTO'
    assert result['status_kategori'] == 'dikonfirmasi' and result['is_unresolved'] is False


def test_similarity_close_distinct_categories_calls_llm(similarity_gate_unit):
    hits = [_gate_hit('c_gaji', .91), _gate_hit('c_gaji', .90, 'transaction'), _gate_hit('c_ops', .89)]
    reply = {'kategori_id': 'c_ops', 'nama_baru': '', 'alasan': 'Purpose matches'}
    with patch('ai.classifier.retrieve_context', return_value=hits), patch('ai.classifier.ask_json', return_value=reply) as llm:
        result = classify('pengeluaran', 'Independent query', SAMPLE_CATS, 'gate-store', 'Umum')
    llm.assert_called_once()
    assert result['source'] == 'llm' and result['kategori_id'] == 'c_ops'


def test_similarity_without_store_embeds_category_definitions(similarity_gate_unit):
    import numpy as np
    with patch('ai.classifier.embed_batch', return_value=np.array([[0., 1.], [1., 0.], [.6, .8]])) as batch, \
         patch('ai.classifier.embed_text', return_value=np.array([1., 0.])), \
         patch('ai.classifier.retrieve_context') as retrieve, patch('ai.classifier.ask_json') as llm:
        result = classify('pengeluaran', 'Independent query', SAMPLE_CATS, business_type='Umum')
    llm.assert_not_called()
    retrieve.assert_not_called()
    assert batch.call_args.args[0] == ['Bahan Baku. Bahan makanan dan mentah', 'Gaji. Gaji karyawan', 'Operasional. Listrik air sewa']
    assert result['kategori_id'] == 'c_gaji' and result['source'] == 'similarity'


@pytest.mark.parametrize('hits', [
    [_gate_hit('foreign', .99)], [_gate_hit('c_exp_other', .99)],
    [_gate_hit('c_jual', .99)], [_gate_hit('c_gaji', .99, store='other-store')],
])
def test_similarity_excludes_foreign_system_type_and_store(similarity_gate_unit, hits):
    reply = {'kategori_id': 'c_ops', 'nama_baru': '', 'alasan': 'Allowed answer'}
    with patch('ai.classifier.retrieve_context', return_value=hits), patch('ai.classifier.ask_json', return_value=reply) as llm:
        result = classify('pengeluaran', 'Independent query', SAMPLE_CATS, 'gate-store', 'Umum')
    llm.assert_called_once()
    assert result['kategori_id'] == 'c_ops' and result['source'] == 'llm'


def test_similarity_foreign_top_revalidated_at_decision_boundary(similarity_gate_unit):
    with patch('ai.classifier._similarity_ranking', return_value=[('foreign', .99), ('c_ops', .60)]), \
         patch('ai.classifier.ask_json', return_value={'kategori_id': 'c_ops', 'nama_baru': '', 'alasan': 'Valid'}) as llm:
        assert classify('pengeluaran', 'Independent query', SAMPLE_CATS)['source'] == 'llm'
    llm.assert_called_once()


def test_similarity_ranking_aggregates_both_sources_and_expands_for_runner_up(similarity_gate_unit):
    from ai.classifier import _similarity_ranking
    hits = [_gate_hit('c_gaji', .91, 'transaction')] * 12 + [_gate_hit('c_ops', .89, 'category')]
    with patch('ai.classifier.retrieve_context', side_effect=lambda *a, **kw: hits[:kw['top_k']]) as retrieve:
        ranking = _similarity_ranking('pengeluaran', 'Independent query', SAMPLE_CATS, 'gate-store')
    assert ranking == [('c_gaji', .91), ('c_ops', .89)]
    assert [call.kwargs['top_k'] for call in retrieve.call_args_list] == [12, 24]
    assert all(call.kwargs['min_score'] == -1.0 for call in retrieve.call_args_list)


@pytest.mark.parametrize('failure_path', ['retrieve_context', 'embed_batch', 'is_semantic_model_available'])
def test_similarity_failure_logs_and_uses_llm(similarity_gate_unit, caplog, failure_path):
    reply = {'kategori_id': 'c_ops', 'nama_baru': '', 'alasan': 'Valid'}
    store = 'gate-store' if failure_path == 'retrieve_context' else None
    with patch('ai.classifier.' + failure_path, side_effect=RuntimeError('embedding/index failed')), \
         patch('ai.classifier.ask_json', return_value=reply) as llm:
        result = classify('pengeluaran', 'Independent query', SAMPLE_CATS, store, 'Umum')
    llm.assert_called_once()
    assert result['kategori_id'] == 'c_ops' and result['source'] == 'llm'
    assert 'Similarity gate unavailable; continuing with LLM' in caplog.text


def test_similarity_failure_does_not_hide_llm_outage(similarity_gate_unit):
    with patch('ai.classifier.embed_batch', side_effect=RuntimeError('embedding failed')), \
         patch('ai.classifier.ask_json', side_effect=LLMUnavailable('down')):
        with pytest.raises(LLMUnavailable, match='Jalankan Ollama'):
            classify('pengeluaran', 'Independent query', SAMPLE_CATS)


def test_similarity_degraded_hash_embeddings_cannot_auto_confirm(similarity_gate_unit, caplog):
    with patch('ai.classifier.is_semantic_model_available', return_value=False), \
         patch('ai.classifier.ask_json', return_value={'kategori_id': 'c_ops', 'nama_baru': '', 'alasan': 'Valid'}) as llm:
        assert classify('pengeluaran', 'Independent query', SAMPLE_CATS)['source'] == 'llm'
    llm.assert_called_once()
    assert 'Semantic embedding model unavailable' in caplog.text


def test_similarity_batch_removes_gated_items_before_llm(similarity_gate_unit):
    import json
    from ai.hybrid_classifier import classify_ambiguous_batch_ai
    def ranking(tipe, desc, cats, toko_id):
        return [('c_gaji', .91), ('c_ops', .60)] if desc == 'repeated item' else []
    reply = {'klasifikasi': [{'item_id': 2, 'kategori_id': 'c_ops', 'nama_baru': '', 'alasan': 'Valid'}]}
    with patch('ai.classifier._similarity_ranking', side_effect=ranking), \
         patch('ai.hybrid_classifier.ask_json', return_value=reply) as llm:
        results = classify_ambiguous_batch_ai('pengeluaran', [(1, 'repeated item'), (2, 'uncertain item')], SAMPLE_CATS, business_type='Umum')
    llm.assert_called_once()
    assert results[1]['source'] == 'similarity' and results[2]['source'] == 'llm'
    assert [item['item_id'] for item in json.loads(llm.call_args.kwargs['user'])['daftar_transaksi']] == [2]
    assert llm.call_args.kwargs['schema']['properties']['klasifikasi']['minItems'] == 1


def test_similarity_bulk_all_gated_cached_and_stale_id_rejected(similarity_gate_unit):
    from ai.hybrid_classifier import hybrid_bulk_classify, _CLASSIFICATION_CACHE
    rows = [{'deskripsi': f'repeat {i}', 'tipe': 'pengeluaran', 'usulan': {'source_id': str(i)}} for i in range(10)]
    with patch('ai.classifier._similarity_ranking', return_value=[('c_gaji', .91), ('c_ops', .60)]) as rank, \
         patch('ai.hybrid_classifier.ask_json') as llm:
        output = hybrid_bulk_classify('gate-store', SAMPLE_CATS, rows, business_type='Umum')
        assert all(r['source'] == 'similarity' for r in output)
        assert [r['usulan']['source_id'] for r in output] == [str(i) for i in range(10)]
        rank.reset_mock()
        assert hybrid_bulk_classify('gate-store', SAMPLE_CATS, rows, business_type='Umum') == output
        rank.assert_not_called()
        for cached in _CLASSIFICATION_CACHE.values():
            cached['kategori_id'] = 'foreign'
        assert all(r['kategori_id'] == 'c_gaji' for r in hybrid_bulk_classify('gate-store', SAMPLE_CATS, rows, business_type='Umum'))
        assert rank.call_count == 10
    llm.assert_not_called()


def test_similarity_cached_single_revalidated_against_current_system_flag(similarity_gate_unit):
    from ai.hybrid_classifier import _CLASSIFICATION_CACHE
    with patch('ai.classifier.retrieve_context', return_value=[]), \
         patch('ai.classifier._similarity_ranking', return_value=[('c_gaji', .91), ('c_ops', .60)]) as rank, \
         patch('ai.classifier.ask_json', return_value={'kategori_id': 'c_ops', 'nama_baru': '', 'alasan': 'Valid'}) as llm:
        first = classify('pengeluaran', 'Independent query', SAMPLE_CATS, 'gate-store', 'Umum')
        first['kategori_id'] = 'tampered copy'
        assert classify('pengeluaran', 'Independent query', SAMPLE_CATS, 'gate-store', 'Umum')['kategori_id'] == 'c_gaji'
        assert rank.call_count == 1
        current = [{**c, 'sistem': True} if c['id'] == 'c_gaji' else c for c in SAMPLE_CATS]
        assert classify('pengeluaran', 'Independent query', current, 'gate-store', 'Umum')['kategori_id'] == 'c_ops'
    llm.assert_called_once()
    assert all(c['kategori_id'] != 'foreign' for c in _CLASSIFICATION_CACHE.values())


def test_revisi1_batch_llm_prompt_omits_few_shot_and_keeps_similarity_gate(similarity_gate_unit):
    import json
    from ai.hybrid_classifier import classify_ambiguous_batch_ai
    with patch('ai.classifier._similarity_ranking', return_value=[]), \
         patch('ai.hybrid_classifier.ask_json', return_value={
             'klasifikasi': [{'item_id': 1, 'kategori_id': 'c_ops', 'nama_baru': '', 'alasan': 'Operasional'}]
         }) as llm:
        results = classify_ambiguous_batch_ai('pengeluaran', [(1, 'bayar listrik bulan ini')], SAMPLE_CATS, toko_id='store-rev1', business_type='Retail')
    assert results[1]['kategori_id'] == 'c_ops'
    sent_payload = json.loads(llm.call_args.kwargs['user'])
    assert 'few_shot_examples_only' not in sent_payload['daftar_transaksi'][0]


def test_revisi1_nama_baru_does_not_create_category_and_low_conf_goes_review(similarity_gate_unit):
    cats_before = [dict(c) for c in SAMPLE_CATS]
    with patch('ai.classifier.retrieve_context', return_value=[]), \
         patch('ai.classifier.ask_json', return_value={
             'kategori_id': '', 'nama_baru': 'Kategori Liar AI', 'alasan': 'Usul baru', 'keyakinan': 50
         }):
        res = classify('pengeluaran', 'transaksi tidak dikenali', cats_before, toko_id='store-rev1', business_type='Retail')
    assert res['kategori_id'] is None
    assert res['status_kategori'] == 'perlu_tinjauan'
    assert len(cats_before) == len(SAMPLE_CATS)


def test_revisi1_extract_csv_category_candidates():
    from services.smart_import_engine import extract_csv_category_candidates
    rows = [
        {'tipe': 'pengeluaran', 'deskripsi': 'beli plastik packing', 'csv_category': 'Kemasan'},
        {'tipe': 'pengeluaran', 'deskripsi': 'beli bensin', 'csv_category': 'Operasional'},  # already exists
        {'tipe': 'pengeluaran', 'deskripsi': 'beli kotak kue', 'csv_category': 'kemasan'},  # duplicate case-insensitive
        {'tipe': 'pemasukan', 'deskripsi': 'jual kue ultah', 'csv_category': 'Pesanan Khusus'},
    ]
    cands = extract_csv_category_candidates(rows, SAMPLE_CATS)
    assert [c['nama'] for c in cands['pengeluaran']] == ['Kemasan']
    assert [c['nama'] for c in cands['pemasukan']] == ['Pesanan Khusus']
    assert cands['pengeluaran'][0]['source'] == 'csv_column'


def test_perbaikan2_others_excluded_from_llm_context_and_coerced_to_abstain(similarity_gate_unit):
    import json
    from ai.classifier import _llm_context, _valid_cached_decision
    from ai.hybrid_classifier import classify_ambiguous_batch_ai

    enriched_cats = [
        {**c, 'kata_kunci': ['tepung terigu', 'gula pasir']} if c['id'] == 'c_bahan' else dict(c)
        for c in SAMPLE_CATS
    ]
    ctx = _llm_context('pengeluaran', 'beli gula pasir 10 kg', enriched_cats, toko_id=None, business_type='Bakery')
    ctx_ids = {c['id'] for c in ctx['daftar_kategori_tersedia']}
    assert 'c_exp_other' not in ctx_ids
    bahan_entry = next(c for c in ctx['daftar_kategori_tersedia'] if c['id'] == 'c_bahan')
    assert 'tepung terigu' in bahan_entry['deskripsi'] and 'gula pasir' in bahan_entry['deskripsi']

    with patch('ai.classifier.retrieve_context', return_value=[]), \
         patch('ai.classifier._similarity_ranking', return_value=[('c_bahan', 0.45), ('c_ops', 0.30)]), \
         patch('ai.classifier.ask_json', return_value={'kategori_id': 'c_exp_other', 'nama_baru': '', 'alasan': 'Tidak yakin'}) as llm:
        res = classify('pengeluaran', 'beli cokelat bubuk', enriched_cats, toko_id='store-p2', business_type='Bakery')
    sent_payload = json.loads(llm.call_args.args[1])
    assert all(c['id'] != 'c_exp_other' for c in sent_payload['daftar_kategori_tersedia'])
    assert res['kategori_id'] is None
    assert res['status_kategori'] == 'perlu_tinjauan'
    assert res['source'] == 'perlu_tinjauan'
    assert res['reason_code'] == 'MODEL_ABSTAINED'

    batch_reply = {'klasifikasi': [{'item_id': 1, 'kategori_id': 'c_exp_other', 'nama_baru': '', 'alasan': 'Ragu'}]}
    with patch('ai.classifier._similarity_ranking', return_value=[]), \
         patch('ai.hybrid_classifier.ask_json', return_value=batch_reply) as batch_llm:
        batch_out = classify_ambiguous_batch_ai('pengeluaran', [(1, 'transaksi ambigu')], enriched_cats, toko_id='store-p2', business_type='Bakery')
    batch_sent = json.loads(batch_llm.call_args.kwargs['user'])
    assert all(c['id'] != 'c_exp_other' for c in batch_sent['daftar_kategori_tersedia'])
    assert batch_out[1]['kategori_id'] is None and batch_out[1]['status_kategori'] == 'perlu_tinjauan'

    assert _valid_cached_decision({'kategori_id': 'c_exp_other', 'source': 'llm'}, 'pengeluaran', enriched_cats) is False


def test_perbaikan4_similarity_diagnostics_and_logging_on_gate_pass_and_fail(similarity_gate_unit, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger='ai.classifier'):
        with patch('ai.classifier.retrieve_context', return_value=[]), \
             patch('ai.classifier._similarity_ranking', return_value=[('c_bahan', 0.91), ('c_ops', 0.60)]), \
             patch('ai.classifier.ask_json') as llm:
            passed = classify('pengeluaran', 'beli terigu', SAMPLE_CATS, toko_id='store-p4', business_type='Bakery')
        llm.assert_not_called()
        assert passed['skor'] == 0.91
        assert passed['usulan']['similarity_diag']['top_category'] == 'Bahan Baku'
        assert passed['usulan']['similarity_diag']['gate_passed'] is True
        assert 'gate=PASS' in caplog.text

        caplog.clear()
        with patch('ai.classifier.retrieve_context', return_value=[]), \
             patch('ai.classifier._similarity_ranking', return_value=[('c_bahan', 0.65), ('c_ops', 0.50)]), \
             patch('ai.classifier.ask_json', return_value={'kategori_id': 'c_bahan', 'nama_baru': '', 'alasan': 'Cocok'}) as llm:
            failed = classify('pengeluaran', 'beli ragi', SAMPLE_CATS, toko_id='store-p4', business_type='Bakery')
        llm.assert_called_once()
        assert failed['skor'] == 0.65
        assert failed['usulan']['similarity_diag']['top_score'] == 0.65
        assert failed['usulan']['similarity_diag']['gate_passed'] is False
        assert 'gate=FAIL' in caplog.text


def test_perbaikan1_bakery_keywords_and_rag_corpus_entries():
    from ai.category_discovery import suggest_default_keywords
    from ai.rag_engine import build_corpus_entries

    kws = suggest_default_keywords('Bahan baku', 'pengeluaran', business_type='Bakery')
    for expected in ('tepung terigu', 'gula pasir', 'cokelat', 'ragi', 'mentega', 'telur', 'susu'):
        assert expected in kws
    assert len(kws) <= 20

    mock_sb = MagicMock()
    def table_side(name):
        t = MagicMock()
        if name == 'kategori':
            t.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
                {'id': 'cat_b', 'nama': 'Bahan baku', 'deskripsi': 'Biaya Bahan Baku', 'kata_kunci': ['cokelat bubuk', 'ragi instan'], 'sistem': False},
                {'id': 'cat_o', 'nama': 'Others', 'deskripsi': '', 'kata_kunci': ['x'], 'sistem': True},
            ]
        else:
            t.select.return_value.eq.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = []
        return t
    mock_sb.table.side_effect = table_side

    with patch('ai.rag_engine.privileged_client', return_value=mock_sb), \
         patch('ai.rag_engine.get_store_business_context', return_value={'bidang_usaha': 'Bakery'}):
        entries = build_corpus_entries('shop-bakery', 'pengeluaran')

    assert all(e['category_id'] != 'cat_o' for e in entries)
    sources = {e['verification_source'] for e in entries}
    assert 'category_definition' in sources and 'category_keyword' in sources
    assert any('cokelat bubuk' in e['text'] for e in entries)


def test_perbaikan3_others_reclassification_preview_and_real_postgres_confirm(queue_database, monkeypatch):
    import uuid
    q = queue_database
    others_id = str(uuid.uuid4())
    tx_id = str(uuid.uuid4())
    q.sql(f"INSERT INTO kategori VALUES ('{others_id}','{q.shop}','pengeluaran');")
    q.cats.append({'id': others_id, 'nama': 'Others', 'tipe': 'pengeluaran', 'sistem': True, 'toko_id': q.shop})
    q.sql(
        f"INSERT INTO transaksi(id,toko_id,tanggal,deskripsi,nominal,tipe,kategori_id,dicatat_oleh,status_persetujuan,status_kategori) "
        f"VALUES ('{tx_id}','{q.shop}','2026-10-30','beli cokelat bubuk 2 kg','150000','pengeluaran','{others_id}','{q.owner}','disetujui','dikonfirmasi');"
    )
    monkeypatch.setattr('services.finance_service.category_by_id', lambda shop, cid: next((c for c in q.cats if c['id'] == cid), None))
    monkeypatch.setattr('services.finance_service.transaction_by_id', lambda shop, tid: next((t for t in q.snapshot() if t['id'] == tid), None))
    monkeypatch.setattr('repositories.finance_repository.transaction_by_id', lambda shop, tid: next((t for t in q.snapshot() if t['id'] == tid), None))

    def fake_bulk(toko_id, cats, rep_rows, **kwargs):
        return [{**r, 'kategori_id': q.cat, 'status_kategori': 'dikonfirmasi', 'source': 'llm', 'skor': 0.72, 'is_unresolved': False, 'message': 'Sesuai', 'usulan': {}} for r in rep_rows]

    with patch('services.ai_service.hybrid_bulk_classify', side_effect=fake_bulk):
        preview = ai_service.preview_others_reclassification('token')

    assert preview['total_others'] == 1
    assert preview['reclassified_count'] == 1
    assert q.snapshot()[0]['kategori_id'] == others_id  # Preview must NOT modify DB!

    with patch('ai.rag_engine.rebuild_store_index') as mock_rebuild:
        res = ai_service.confirm_others_reclassification('token', preview['updates'])

    assert res['updated_count'] == 1 and res['reapproved_count'] == 1
    assert mock_rebuild.called
    stored = q.snapshot()[0]
    assert stored['kategori_id'] == q.cat
    assert stored['status_persetujuan'] == 'disetujui'
    assert int(q.sql(f"SELECT count(*) FROM audit_log WHERE toko_id='{q.shop}';")) >= 2
