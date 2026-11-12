"""Rebuild the original recorded discovery response for reproducible gate-only comparison.
The six proposals below are evidence from the pre-edit diagnostic run, not production defaults.
No model answers are replaced during classification or accuracy evaluation.
"""
import importlib.util,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from measure_retail_before import _load_csv_rows,RETAIL_CSV,_unique_descriptions,_build_temp_faiss_entries,_evaluate_gate_table,RAFI_MART_ID
from ai.rag_engine import build_and_save_index,clear_memory_cache
from database.client import privileged_client
spec=importlib.util.spec_from_file_location('before',ROOT/'evidence/retail_fix/category_discovery_before.py')
before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
recorded=[
('Bahan Baku','Transaksi pengeluaran untuk bahan-bahan yang digunakan dalam operasional toko, seperti rokok, mie instan, minyak goreng, susu UHT, sabun mandi, air mineral, beras, telur ayam, kopi sachet, deterjen, sampo sachet, gula pasir, teh celup, snack.', ['tepung','gula','minyak','telur','bahan produksi','bahan baku utama','bahan baku','bahan mentah','input','operasional']),
('Jasa Servis','Transaksi pengeluaran untuk jasa-jasa yang digunakan dalam operasional toko, seperti ojek, internet wifi, listrik toko, air PDAM.', ['jasa','servis','operasional','utility']),
('Gaji dan Upah','Transaksi pengeluaran untuk gaji pegawai gudang, gaji kasir harian, dan upah lainnya.', ['gaji kasir','upah karyawan','lembur staf','bonus karyawan','shift kasir','gaji karyawan','upah harian','lembur','bonus staf','gaji','upah','pegawai','kasir']),
('Promosi dan Pemasaran','Transaksi pengeluaran untuk biaya promosi instagram dan biaya pemasaran lainnya.', ['biaya iklan','spanduk banner','promosi media sosial','diskon promo','promosi','pemasaran','marketing','advertising']),
('Perlengkapan dan Perawatan','Transaksi pengeluaran untuk perlengkapan dan perawatan toko, seperti plastik dan kantong kertas.', ['perlengkapan','perawatan','toko','inventory']),
('Pengeluaran Lainnya','Transaksi pengeluaran lainnya yang tidak termasuk kategori di atas, seperti biaya antar pesanan ojek.', ['pengeluaran','lainnya','various','others']),
]
discovered=[dict(id=f'recorded_{i}',nama=n,deskripsi=d,kata_kunci=k,tipe='pengeluaran',sistem=False) for i,(n,d,k) in enumerate(recorded)]
rafi=privileged_client().table('kategori').select('*').eq('toko_id',RAFI_MART_ID).execute().data or []
for c in rafi:
 if not c.get('sistem') and not c.get('bukan_operasional'):
  c['kata_kunci']=before.suggest_default_keywords(c['nama'],c['tipe'],'Retail')
result={}
with tempfile.TemporaryDirectory() as temp,patch('ai.rag_engine.INDEX_ROOT_DIR',temp),patch('ai.rag_engine.get_remote_version',return_value=1):
 for name,cats in [('discovery',discovered),('rafi_autofill',rafi)]:
  shop='baseline-recorded-'+name
  entries=_build_temp_faiss_entries(shop,[c for c in cats if c['tipe']=='pengeluaran'],'Retail')
  for entry in entries:
   if entry['verification_source']=='category_keyword':
    n=entry['category_name'];kw=entry['text'];entry['text']=kw if n.casefold() in kw.casefold() else f'{n}: {kw}'
  clear_memory_cache(shop);build_and_save_index(shop,'pengeluaran',entries=entries,version=1)
  gate=_evaluate_gate_table(cats,_unique_descriptions(_load_csv_rows(RETAIL_CSV)),toko_id=shop)
  result[name]={'categories':cats,'gate':gate,'pass':sum(r['gate_passed'] for r in gate)}
  print(name,result[name]['pass'],flush=True)
(ROOT/'evidence/retail_fix/retail_baseline_gates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
