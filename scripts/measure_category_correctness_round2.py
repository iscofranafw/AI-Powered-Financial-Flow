"""Read-only round-two measurements; no database mutations, real local Llama."""
import ast,json,re,sys,tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ai.category_evidence import compile_keyword_plan
from ai.category_discovery import ground_discovered_categories, discover_categories_from_transactions
from ai.hybrid_classifier import hybrid_bulk_classify,clear_classification_cache
from ai.rag_engine import clear_memory_cache,build_and_save_index,build_corpus_entries
from services.finance_service import plan_category_keywords
from measure_retail_before import _build_temp_faiss_entries,_load_csv_rows,_unique_descriptions,_evaluate_gate_table,RETAIL_CSV,BAKERY_CSV
OUT=ROOT/'evidence/category_correctness_round2'
snapshot=json.loads((OUT/'input_snapshot.json').read_text(encoding='utf-8'))
retail=_load_csv_rows(RETAIL_CSV);bakery=_load_csv_rows(BAKERY_CSV)
output={}

class SnapshotQuery:
 def __init__(self, rows): self.rows=rows
 def select(self, fields): return self
 def eq(self, key, value): self.rows=[r for r in self.rows if r.get(key)==value];return self
 def order(self, *args, **kwargs): return self
 def limit(self, count): self.rows=self.rows[:count];return self
 def execute(self): return SimpleNamespace(data=self.rows)

class SnapshotClient:
 def __init__(self,cats,history,shop):
  self.tables={'kategori':[{**c,'toko_id':shop} for c in cats],
               'transaksi':[{**r,'id':r.get('id',str(i)),'toko_id':shop,
                              'status_persetujuan':'disetujui','status_kategori':'dikonfirmasi'} for i,r in enumerate(history)]}
 def table(self,name): return SnapshotQuery(list(self.tables[name]))
 # No insert/update/delete/RPC interface: production corpus calls can only read.

def evaluate(label,cats,rows,business,history=None):
 shop='round2-'+label
 with tempfile.TemporaryDirectory() as tmp,patch('ai.rag_engine.INDEX_ROOT_DIR',tmp),patch('ai.rag_engine.get_remote_version',return_value=1),patch('ai.rag_engine.get_store_business_context',return_value={'bidang_usaha':business}):
  clear_memory_cache(shop);clear_classification_cache(shop)
  for tp in ('pengeluaran','pemasukan'):
   if history is not None:
    with patch('ai.rag_engine.privileged_client',return_value=SnapshotClient(cats,history,shop)):
     entries=build_corpus_entries(shop,tp)
   else:
    entries=_build_temp_faiss_entries(shop,[c for c in cats if c['tipe']==tp],business)
   build_and_save_index(shop,tp,entries=entries,version=1)
  gate=_evaluate_gate_table(cats,_unique_descriptions(rows),shop)
  result=hybrid_bulk_classify(shop,cats,rows,business_type=business)
  names={c['id']:c['nama'] for c in cats}
  mapped={r['deskripsi']:dict(kategori_id=r.get('kategori_id'),category=names.get(r.get('kategori_id'),'Perlu Tinjauan'),source=r['source'],reason_code=r.get('reason_code'),message=r.get('message')) for r in result}
  data={'gate':gate,'pass':sum(r['gate_passed'] for r in gate),'distribution':dict(Counter(names.get(r.get('kategori_id'),'Perlu Tinjauan') for r in result)),'classified':mapped,'categories':cats}
  if history:
   truth={r['deskripsi']:r['kategori_id'] for r in history}
   data['correct']=sum(r.get('kategori_id')==truth.get(r['deskripsi']) for r in result)
   data['total']=len(result)
  output[label]=data
  (OUT/'measurements.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  print(label,'gate',data['pass'],'/',len(gate),data['distribution'],flush=True)
  return data

shaz=snapshot['shaz'];rafi=snapshot['rafi']
# Snapshot baseline equals the previous round's stored keyword view, without
# silently including new history-derived phrases in the before condition.
# Previous round tables already contain the baseline distribution; no need to
# call Llama again for baseline guesses.
with tempfile.TemporaryDirectory() as tmp,patch('ai.rag_engine.INDEX_ROOT_DIR',tmp),patch('ai.rag_engine.get_remote_version',return_value=1),patch('ai.rag_engine.get_store_business_context',return_value={'bidang_usaha':'Bakery'}):
 shop='round2-bakery-before';cats=shaz['categories']
 build_and_save_index(shop,'pengeluaran',entries=_build_temp_faiss_entries(shop,[c for c in cats if c['tipe']=='pengeluaran'],'Bakery'),version=1)
 output['bakery_before']={'gate':_evaluate_gate_table(cats,_unique_descriptions(bakery),shop)}

bplan=compile_keyword_plan(shaz['categories'],shaz['history']['pengeluaran'],'Bakery')
(OUT/'bakery_plan.json').write_text(json.dumps(bplan,ensure_ascii=False,indent=2),encoding='utf-8')
evaluate('bakery_after',bplan['categories'],bakery,'Bakery')

raw=json.loads((OUT/'discovery_raw_before.json').read_text(encoding='utf-8'))
samples=[(d,d,n) for (tp,d),n in _unique_descriptions(retail)]
recs=discover_categories_from_transactions(retail,snapshot['doni']['categories'],'Retail')['pengeluaran']
cats=[dict(c,id='disc_'+str(i)) for i,c in enumerate(recs)]
(OUT/'discovery_plan.json').write_text(json.dumps(recs,ensure_ascii=False,indent=2),encoding='utf-8')
evaluate('discovery_after',cats,retail,'Retail')

history=rafi['history']['pengeluaran']+rafi['history']['pemasukan']
plan=plan_category_keywords(rafi['categories'],history,'Retail',retail)
(OUT/'rafi_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
evaluate('rafi_after',plan['categories'],retail,'Retail')
evaluate('rafi_existing_expenses_before',rafi['categories'],rafi['history']['pengeluaran'],'Retail',rafi['history']['pengeluaran'])
evaluate('rafi_existing_expenses_after',plan['categories'],rafi['history']['pengeluaran'],'Retail',rafi['history']['pengeluaran'])
# Income is measured against the already-approved stored IDs, not invented labels.
income=rafi['history']['pemasukan']
baseplan=compile_keyword_plan(rafi['categories'],history,'Retail')
evaluate('income_before',rafi['categories'],income,'Retail',income)
evaluate('income_after',baseplan['categories'],income,'Retail',income)
