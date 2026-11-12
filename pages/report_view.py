import streamlit as st
import pandas as pd
from datetime import date, timedelta
from services import report_service as report
from services import forecast_service as forecast_svc
from services.finance_service import list_categories
from services.validation_service import get_jakarta_today
from utils.ui_components import page_header, section_header, format_rupiah, render_recurring_expense_reminders


def render(token):
    page_header('Ringkasan Keuangan & Arus Kas', 'Analisis performa keuangan operasional toko dengan pemisahan setoran modal.')

    today_wib = get_jakarta_today()
    col_date1, col_date2 = st.columns(2)
    d1 = col_date1.date_input('Periode Mulai', today_wib - timedelta(days=90), key='report_start')
    d2 = col_date2.date_input('Periode Akhir', today_wib, key='report_end')
    if d1 > d2:
        st.error('Tanggal akhir harus setelah tanggal mulai.')
        return

    try:
        rows = report.period_rows(token, d1, d2)
        cats = list_categories(token)
        numbers = report.metrics(rows, cats)

        section_header('Metrik Utama Periode Terpilih')
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric('Pemasukan Total', format_rupiah(numbers['pemasukan']))
        col_kpi2.metric('Pengeluaran Total', format_rupiah(numbers['pengeluaran']))
        col_kpi3.metric('Omzet Penjualan Murni', format_rupiah(numbers['omzet']),
                        help='Pemasukan murni operasional penjualan tanpa memperhitungkan tambahan modal')

        col_kpi4, col_kpi5 = st.columns(2)
        col_kpi4.metric('Arus Kas Operasional', format_rupiah(numbers['arus_kas_operasional']),
                        help='Omzet penjualan dikurangi total pengeluaran usaha')
        col_kpi5.metric('Arus Kas Bersih Total', format_rupiah(numbers['arus_kas_total']),
                        help='Total seluruh dana masuk (termasuk modal) dikurangi seluruh pengeluaran')

        if numbers['bulanan'] or numbers['pengeluaran_per_kategori']:
            import plotly.express as px
            st.divider()
            col_chart1, col_chart2 = st.columns(2)

            if numbers['bulanan']:
                d = []
                for month, vals in numbers['bulanan'].items():
                    for typ, num in vals.items():
                        d.append({'Bulan': month, 'Jenis': typ.capitalize(), 'Nominal': float(num)})
                fig_bar = px.bar(
                    pd.DataFrame(d), x='Bulan', y='Nominal', color='Jenis', barmode='group',
                    title='Tren Arus Kas Bulanan',
                    color_discrete_map={'Pemasukan': '#0D9488', 'Pengeluaran': '#E11D48'}
                )
                fig_bar.update_layout(
                    font_family='Inter, sans-serif',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title='Nominal (Rp)',
                    xaxis_title='Bulan',
                    legend_title_text='Jenis',
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                col_chart1.plotly_chart(fig_bar, use_container_width=True)

            if numbers['pengeluaran_per_kategori']:
                cat_names = list(numbers['pengeluaran_per_kategori'].keys())
                cat_values = [float(v) for v in numbers['pengeluaran_per_kategori'].values()]
                palette = ['#0D9488', '#2563EB', '#D97706', '#64748B', '#7C3AED', '#DB2777', '#059669', '#E11D48']
                fig_pie = px.pie(
                    names=cat_names, values=cat_values,
                    title='Distribusi Pengeluaran per Kategori',
                    hole=0.45,
                    color_discrete_sequence=palette
                )
                fig_pie.update_layout(
                    font_family='Inter, sans-serif',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                col_chart2.plotly_chart(fig_pie, use_container_width=True)

        st.divider()
        section_header('Prediksi Pengeluaran & Tagihan Rutin', 'Perkiraan pengeluaran untuk 3–90 hari ke depan dan jadwal tagihan berkala.')
        try:
            rec_data = forecast_svc.detect_recurring_expenses(token)
            if rec_data.get('upcoming_3_days_items'):
                render_recurring_expense_reminders(rec_data)

            readiness = forecast_svc.get_forecast_readiness_status(token)
            if not readiness['ready']:
                st.info(f"ℹ️ **Status Peramalan:** {readiness['message']}")
            else:
                forecast_days = st.slider(
                    'Horizon Peramalan Pengeluaran (Hari ke Depan)',
                    min_value=3,
                    max_value=90,
                    value=30,
                    step=1,
                    help='Pilih rentang proyeksi antara 3 hingga 90 hari (default 30 hari).',
                )
                fcst = forecast_svc.train_and_forecast_expenses(token, days_ahead=int(forecast_days))
                if fcst.get('success'):
                    chosen_model = fcst.get('selected_model') or fcst.get('best_model') or 'Multi-Model Terbaik'
                    col_f1, col_f3, col_f4 = st.columns(3)
                    col_f1.metric(f'Estimasi Pengeluaran {forecast_days} Hari', format_rupiah(fcst['total_predicted_expense']))
                    col_f3.metric('Rentang Estimasi', f"{format_rupiah(fcst['total_predicted_lower'])} - {format_rupiah(fcst['total_predicted_upper'])}")
                    col_f4.metric('Rata-rata Harian Proyeksi', format_rupiah(fcst['average_daily_predicted']))

                    if fcst.get('projections'):
                        import plotly.express as px
                        df_proj = pd.DataFrame(fcst['projections'])
                        fig_line = px.line(
                            df_proj, x='ds', y='yhat',
                            title=f'Proyeksi Pengeluaran Harian {forecast_days} Hari ke Depan',
                            labels={'ds': 'Tanggal', 'yhat': 'Estimasi Pengeluaran (Rp)'}
                        )
                        fig_line.update_layout(
                            font_family='Inter, sans-serif',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_line, use_container_width=True)

                        st.dataframe(df_proj[['ds', 'yhat']].rename(columns={
                            'ds': 'Tanggal', 'yhat': 'Estimasi Pengeluaran (Rp)',
                        }), use_container_width=True, hide_index=True)

                    with st.expander('Detail teknis', expanded=False):
                        st.write(f"Model terpilih: {chosen_model}")
                        target_status = fcst.get('target_status', '')
                        if target_status:
                            if fcst.get('target_met'):
                                st.success(f"✅ Status Akurasi: {target_status}")
                            elif fcst.get('best_mape') is not None and fcst.get('best_mape') >= 20.0:
                                st.warning(f"⚠️ Peringatan Akurasi: {target_status}. Pola transaksi toko masih fluktuatif/jarang sehingga kesalahan persentase tetap tinggi.")
                            else:
                                st.info(f"ℹ️ Status Akurasi: {target_status}")

                        eval_res = forecast_svc.evaluate_forecast_accuracy(token)
                        if eval_res.get('success') and eval_res.get('ready'):
                            st.caption(f"Perbandingan model — evaluasi {eval_res.get('test_days', 28)} hari")
                            comp_rows = eval_res.get('model_comparison') or []
                            if comp_rows:
                                comp_df = pd.DataFrame([
                                    {
                                        'Model': r['model'],
                                        'MAPE (%)': f"{r['mape']:.2f}%" if r['mape'] is not None else 'N/A',
                                        'MAE (Rp)': format_rupiah(r['mae']),
                                        'RMSE (Rp)': format_rupiah(r['rmse']),
                                        'WAPE (%)': f"{r['wape']:.2f}%" if r['wape'] is not None else 'N/A',
                                        'Status': 'Terpilih (Terbaik)' if r['model'] == eval_res.get('best_model') else 'Kandidat',
                                    }
                                    for r in comp_rows
                                ])
                                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                            st.markdown(eval_res.get('message', ''))
                else:
                    st.info('Prediksi belum tersedia untuk data toko saat ini.')
                    with st.expander('Detail teknis', expanded=False):
                        st.write(fcst.get('message', 'Tidak ada proyeksi peramalan yang tersedia saat ini.'))
        except Exception as exc:
            st.error('Prediksi pengeluaran belum dapat ditampilkan. Silakan coba lagi.')
            with st.expander('Detail teknis', expanded=False):
                st.write(str(exc))

        st.divider()
        section_header('Ringkasan Eksekutif Keuangan', 'Analisis naratif kualitatif otomatis berdasarkan kalkulasi matematis arus kas.')

        if st.button('Buat Ringkasan Eksekutif'):
            with st.spinner('Menyusun ringkasan laporan...'):
                st.session_state['report_narrative'] = report.narrative(numbers)

        current_narrative = st.session_state.get('report_narrative')
        if current_narrative:
            st.markdown(f"<div class='narrative-card'>{current_narrative}</div>", unsafe_allow_html=True)

        st.divider()
        section_header('Ekspor Berkas Pembukuan', 'Unduh mutasi transaksi yang telah diverifikasi untuk pelaporan akuntansi eksternal.')
        btn_csv, btn_excel = st.columns(2)
        btn_csv.download_button(
            'Unduh Berkas CSV', report.export_csv(rows, cats),
            f'laporan_transaksi_{d1}_{d2}.csv', 'text/csv', use_container_width=True
        )
        btn_excel.download_button(
            'Unduh Laporan Excel (.xlsx)', report.export_excel(rows, cats),
            f'laporan_transaksi_{d1}_{d2}.xlsx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True
        )

    except Exception as exc:
        st.error(str(exc))
