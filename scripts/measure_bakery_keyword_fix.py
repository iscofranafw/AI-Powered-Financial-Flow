"""Read-only bakery before/after comparison with isolated FAISS indexes."""
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from database.client import privileged_client
from ai.category_discovery import suggest_default_keywords
from ai.hybrid_classifier import hybrid_bulk_classify,clear_classification_cache
from ai.rag_engine import build_and_save_index,clear_memory_cache
from measure_retail_before import _build_temp_faiss_entries,_evaluate_gate_table,_unique_descriptions,_load_csv_rows,BAKERY_CSV
spec=importlib.util.spec_from_file_location('discovery_before',ROOT/'evidence/retail_fix/category_discovery_before.py')
before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
shop='e1ebf331-aa99-4cd9-a255-c0682c2b9857'
sb=privileged_client()
cats=sb.table('kategori').select('*').eq('toko_id',shop).execute().data or []
history=(sb.table('transaksi').select('kategori_id,deskripsi,tipe').eq('toko_id',shop)
         .eq('status_persetujuan','disetujui').eq('status_kategori','dikonfirmasi').order('tanggal',desc=True).limit(1000).execute().data or [])
rows=_load_csv_rows(BAKERY_CSV)
output={}
with tempfile.TemporaryDirectory() as temp,patch('ai.rag_engine.INDEX_ROOT_DIR',temp),patch('ai.rag_engine.get_remote_version',return_value=1):
 for label in ('before','after'):
  selected=[]
  for cat in cats:
   c=dict(cat)
   if not c.get('sistem') and not c.get('bukan_operasional'):
    samples=[r['deskripsi'] for r in history if r.get('kategori_id')==c['id'] and r.get('tipe')==c['tipe']]
    c['kata_kunci']=(cat.get('kata_kunci') or []) if label=='before' else (cat.get('kata_kunci') or suggest_default_keywords(c['nama'],c['tipe'],'Bakery',contoh=samples))
   selected.append(c)
  sim='bakery-evidence-'+label
  clear_memory_cache(sim);clear_classification_cache(sim)
  for tp in ('pengeluaran','pemasukan'):
   entries=_build_temp_faiss_entries(sim,[c for c in selected if c['tipe']==tp],'Bakery')
   if label == 'before':
    for entry in entries:
     if entry['verification_source'] == 'category_keyword':
      name=entry['category_name']; kw=entry['text']
      entry['text']=kw if name.casefold() in kw.casefold() else f'{name}: {kw}'
   build_and_save_index(sim,tp,entries=entries,version=1)
  gate=_evaluate_gate_table(selected,_unique_descriptions(rows),toko_id=sim)
  classified=hybrid_bulk_classify(sim,selected,rows,business_type='Bakery')
  names={c['id']:c['nama'] for c in selected}
  output[label]={'gate':gate,'pass':sum(g['gate_passed'] for g in gate),'total':len(gate),
    'distribution':dict(Counter(names.get(r.get('kategori_id'),'Perlu Tinjauan') for r in classified)),
    'sources':dict(Counter(r['source'] for r in classified)),
    'keyword_source':'existing stored Shaz keywords' if label=='before' else 'preserve existing; fill missing from approved confirmed history then references'}
  print(label,output[label]['pass'],output[label]['distribution'],flush=True)
(ROOT/'evidence/retail_fix/bakery_final_comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
