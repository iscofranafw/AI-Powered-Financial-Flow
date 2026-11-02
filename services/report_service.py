import csv, io
from decimal import Decimal
from collections import defaultdict
from services.finance_service import dashboard_summary
from services.auth_service import verify_actor
from repositories.finance_repository import transactions,categories
from ai.ollama_client import ask_json, LLMUnavailable


def period_rows(token,start=None,end=None):
    actor=verify_actor(token,{'pemilik_toko'})
    rows=[x for x in transactions(actor['toko_id']) if x['status_persetujuan']=='disetujui']
    if start:rows=[x for x in rows if x['tanggal']>=str(start)]
    if end:rows=[x for x in rows if x['tanggal']<=str(end)]
    return rows

def metrics(rows,cats):
    cmap={c['id']:c for c in cats}
    income=expense=capital=Decimal(0)
    by_category=defaultdict(Decimal);by_month=defaultdict(lambda:{'pemasukan':Decimal(0),'pengeluaran':Decimal(0)})
    for t in rows:
        amount=Decimal(str(t['nominal']));type_=t['tipe']
        if type_=='pemasukan':
            income+=amount
            if cmap.get(t['kategori_id'],{}).get('bukan_operasional'):capital+=amount
        else:
            expense+=amount
            by_category[cmap.get(t['kategori_id'],{}).get('nama','?')]+=amount
        by_month[t['tanggal'][:7]][type_]+=amount
    return {'pemasukan':income,'pengeluaran':expense,'tambahan_modal':capital,
            'omzet':income-capital,'arus_kas_operasional':income-capital-expense,
            'arus_kas_total':income-expense,'pengeluaran_per_kategori':dict(by_category),
            'bulanan':dict(sorted(by_month.items())),'jumlah_disetujui':len(rows)}

def narrative(data):
    expected={k:float(data[k]) for k in ('pemasukan','pengeluaran','arus_kas_operasional','arus_kas_total')}
    backup=f"Pemasukan Rp{expected['pemasukan']:,.0f}; pengeluaran Rp{expected['pengeluaran']:,.0f}; arus kas operasional Rp{expected['arus_kas_operasional']:,.0f}."
    try:
        import json
        schema = {'type':'object','properties':{
            'teks':{'type':'string'},
            'angka':{'type':'object','properties':{k:{'type':'number'} for k in expected},
                     'required':list(expected)}},
            'required':['teks','angka']}
        result=ask_json('Tulis satu kalimat singkat laporan keuangan UMKM. Jangan mengubah atau menghitung angka. JSON berisi teks dan angka persis yang diterima.',
                        json.dumps(expected),schema)
        if all(abs(float(result.get('angka',{}).get(k,1e100))-v)<.005 for k,v in expected.items()):
            candidate=str(result['teks'])[:700]
            if not any(ch.isdigit() for ch in candidate):return candidate
    except (LLMUnavailable,ValueError,KeyError,TypeError):
        # Safe fallback: if LLM generation fails or outputs invalid figures, return deterministic narrative backup
        pass
    return backup

def export_csv(rows,cats):
    cmap={c['id']:c['nama'] for c in cats}
    out=io.StringIO();writer=csv.writer(out)
    writer.writerow(['tanggal','deskripsi','nominal','tipe','kategori','status'])
    for r in rows:
        writer.writerow([r['tanggal'],r['deskripsi'],r['nominal'],r['tipe'],cmap.get(r['kategori_id'],''),r['status_persetujuan']])
    return out.getvalue().encode('utf-8-sig')

def export_excel(rows,cats):
    from openpyxl import Workbook
    out=io.BytesIO();wb=Workbook();ws=wb.active;ws.title='Transaksi'
    cmap={c['id']:c['nama'] for c in cats}
    ws.append(['Tanggal','Deskripsi','Nominal','Jenis','Kategori','Status'])
    for row in rows:ws.append([row['tanggal'],row['deskripsi'],float(row['nominal']),row['tipe'],cmap.get(row['kategori_id'],''),row['status_persetujuan']])
    ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
    for col,width in zip('ABCDEF',[15,54,18,20,28,20]):ws.column_dimensions[col].width=width
    wb.save(out);return out.getvalue()
