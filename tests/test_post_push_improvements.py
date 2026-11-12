"""Forecast actions/date details and guide chips, using isolated test shops."""
import copy
from contextlib import ExitStack
from datetime import date, datetime
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from ai.guidance_prompts import get_guidance_starters
from ai.copilot_planner import plan_stage_a_deterministic
from ai.copilot_schema import ToolName
from utils.forecast_details import daily_expense_detail


@pytest.fixture
def report_ui():
    from services import forecast_service as forecast
    history = [dict(id=str(i), tanggal=d, deskripsi='tagihan internet toko', nominal=350000,
                    kategori_id='ops', tipe='pengeluaran', is_verified_due_date=True)
               for i, d in enumerate(('2026-10-22', '2026-10-29', '2026-11-05'))]
    cats = [dict(id='ops', nama='Operasional', tipe='pengeluaran')]
    with patch.object(forecast, '_get_eligible_expense_rows', return_value=({'toko_id': 'new-local-detail-shop'}, history, cats)), \
         patch.object(forecast, 'get_jakarta_now', return_value=datetime(2026, 11, 12, 12)):
        recurring = forecast.detect_recurring_expenses('test')
    assert recurring['recurring_items'][0]['next_expected_date'] == '2026-11-12'
    payload = dict(success=True, selected_model='ETS', total_predicted_expense=9647552.80,
        average_daily_predicted=321585.09, total_predicted_lower=659990.33, total_predicted_upper=19345678.99,
        projections=[dict(ds=d, yhat=v, yhat_lower=0, yhat_upper=v*2)
                     for d, v in [('2026-11-12', 400000), ('2026-11-13', 250000), ('2026-11-14', 300000)]])
    numbers = dict(pemasukan=0, pengeluaran=0, omzet=0, arus_kas_operasional=0, arus_kas_total=0,
                   bulanan={}, pengeluaran_per_kategori={})
    with ExitStack() as stack:
        for target, value in {
            'services.report_service.period_rows': [], 'pages.report_view.list_categories': cats,
            'services.report_service.metrics': numbers, 'services.report_service.export_csv': b'',
            'services.report_service.export_excel': b'', 'services.forecast_service.detect_recurring_expenses': recurring,
            'services.forecast_service.get_forecast_readiness_status': {'ready': True},
            'services.forecast_service.evaluate_forecast_accuracy': {},
        }.items():
            stack.enter_context(patch(target, return_value=value))
        train = stack.enter_context(patch('services.forecast_service.train_and_forecast_expenses', return_value=payload))
        app = AppTest.from_string("import streamlit as st\nfrom pages.report_view import render\nrender(st.session_state.get('test_token', 'test'))").run()
        yield app, train, payload, recurring


def forecast_tables(app):
    return [e for e in app.dataframe if 'Estimasi Pengeluaran (Rp)' in e.value.columns]


def flattened(node):
    yield node
    for child in getattr(node, 'children', {}).values():
        yield from flattened(child)


def test_prediction_button_directly_precedes_table_and_preserves_values(report_ui):
    app, train, payload, _ = report_ui
    train.assert_not_called()
    assert not forecast_tables(app)
    original = copy.deepcopy(payload)
    app.button(key='btn_calculate_forecast').click().run()
    assert not app.exception and not app.error
    train.assert_called_once_with('test', days_ahead=30)
    elements = list(flattened(app.main))
    button_pos = next(i for i, e in enumerate(elements) if getattr(e, 'label', '') == 'Hitung Prediksi')
    table_pos = next(i for i, e in enumerate(elements) if hasattr(getattr(e, 'value', None), 'columns') and 'Estimasi Pengeluaran (Rp)' in e.value.columns)
    assert not any(getattr(e, 'type', '') in ('metric', 'plotly_chart') for e in elements[button_pos+1:table_pos])
    assert list(forecast_tables(app)[0].value['Estimasi Pengeluaran (Rp)']) == [r['yhat'] for r in payload['projections']]
    assert payload == original


def test_date_detail_matches_real_recurring_detection_and_empty_day(report_ui):
    app, train, payload, recurring = report_ui
    app.button(key='btn_calculate_forecast').click().run()
    picker = next(s for s in app.selectbox if s.label == 'Buka detail tanggal')
    picker.select('2026-11-12').run()
    assert not app.exception and not app.error
    text = '\n'.join(str(e.value) for e in [*app.markdown, *app.caption])
    assert 'tagihan internet toko' in text and '350.000' in text and '400.000' in text
    assert 'Jadwal terkonfirmasi' in text and 'bukan kepastian' in text
    picker = next(s for s in app.selectbox if s.label == 'Buka detail tanggal')
    picker.select('2026-11-13').run()
    assert any('Tidak ada tagihan rutin jatuh tempo hari ini' in e.value for e in app.info)
    assert any('250.000' in e.value for e in app.markdown)
    train.assert_called_once()  # Selecting a date never retrains the model.


def test_horizon_and_token_changes_never_show_another_result(report_ui):
    app, train, _, _ = report_ui
    app.button(key='btn_calculate_forecast').click().run()
    app.slider[0].set_value(7).run()
    assert not forecast_tables(app) and train.call_count == 1
    app.button(key='btn_calculate_forecast').click().run()
    assert train.call_args.kwargs == {'days_ahead': 7}
    app.session_state['test_token'] = 'another-test-owner'
    app.run()
    assert not forecast_tables(app) and train.call_count == 2


def test_full_range_is_normal_wrapping_text_not_a_truncated_metric(report_ui):
    app, _, _, _ = report_ui
    app.button(key='btn_calculate_forecast').click().run()
    assert not any(e.label == 'Rentang Estimasi' for e in app.metric)
    text = '\n'.join(e.value for e in app.markdown)
    assert 'Batas bawah: Rp 659.990,33' in text
    assert 'Batas atas: Rp 19.345.678,99' in text


def test_detail_uses_only_returned_dates_and_marks_inferred_schedule():
    projections = [dict(ds='2026-11-12', yhat=100), dict(ds='2026-11-19', yhat=200)]
    schedule = [dict(description='internet', estimated_amount=50, next_expected_date='2026-11-12',
                     estimated_interval_days=7, is_verified_due_date=False)]
    original = copy.deepcopy(schedule)
    result = daily_expense_detail(projections, schedule, '2026-11-12')
    assert result['estimasi_total'] == 100 and result['tagihan'][0]['nominal'] == 50
    assert 'Perkiraan jadwal' in result['tagihan'][0]['sumber']
    assert not daily_expense_detail(projections, schedule, '2026-11-19')['tagihan']
    assert schedule == original
    with pytest.raises(ValueError):
        daily_expense_detail(projections, schedule, '2026-12-01')


@pytest.mark.parametrize('group', ['bakery', 'retail', 'warung_makan', 'bengkel_motor', 'lainnya'])
def test_four_guide_starters_are_sourced_and_route_to_knowledge(group):
    from ai.knowledge_rag import load_guide_chunks
    starters = get_guidance_starters(group)
    chunks = load_guide_chunks(group)
    assert len(starters) == 4
    for starter in starters:
        assert starter['group'] == group
        assert any(c['judul_bagian'] == starter['section'] and c['nama_file'] == starter['filename']
                   and c['sumber_url'] == starter['source_url'] for c in chunks)
        assert plan_stage_a_deterministic(starter['question']).tool_name == ToolName.GET_BUSINESS_GUIDANCE


def test_no_guide_means_no_invented_starter():
    with patch('ai.guidance_prompts.load_guide_chunks', return_value=[]):
        assert get_guidance_starters('unknown') == []


@pytest.mark.parametrize('surface', ['chat', 'floating'])
def test_chip_click_runs_chat_knowledge_rag_and_cites_source(surface):
    from ai.knowledge_rag import load_guide_chunks
    from services.chat_service import handle_chat_message
    actor = dict(id='test-guide-owner', toko_id='new-local-guide-shop', role='pemilik_toko')
    ctx = {'bidang_usaha': 'Bakery', 'nama': 'New Test Bakery'}
    chunk = next(c for c in load_guide_chunks('bakery') if c['judul_bagian'].startswith('3.'))
    chunk = {**chunk, 'score': .9}
    with ExitStack() as stack:
        for target in ('services.chat_service.verify_actor', 'ai.copilot_tools.verify_actor', 'ai.knowledge_rag.verify_actor'):
            stack.enter_context(patch(target, return_value=actor))
        for target in ('pages.chat_view.get_store_business_context', 'utils.floating_agent.get_store_business_context',
                       'ai.knowledge_rag.get_store_business_context'):
            stack.enter_context(patch(target, return_value=ctx))
        retrieve = stack.enter_context(patch('ai.knowledge_rag.retrieve_guidance_chunks', return_value=[chunk]))
        stack.enter_context(patch('ai.knowledge_rag.ask_text', return_value='Berikut panduan berdasarkan sumber yang tersedia.'))
        module = 'pages.chat_view' if surface == 'chat' else 'utils.floating_agent'
        handler = stack.enter_context(patch(module+'.handle_chat_message', side_effect=handle_chat_message))
        source = ("from pages.chat_view import render\nrender('test', "+repr(actor)+")" if surface == 'chat' else
                  "import streamlit as st\nst.session_state['floating_agent_open']=True\nfrom utils.floating_agent import render_floating_agent\nrender_floating_agent('test', "+repr(actor)+")")
        if surface == 'floating':
            # AppTest performs whole-app reruns rather than browser fragment events.
            import streamlit as st
            real_rerun = st.rerun
            stack.enter_context(patch('utils.floating_agent.st.rerun', side_effect=lambda *a, **k: real_rerun()))
        app = AppTest.from_string(source, default_timeout=30).run()
        assert not app.exception
        chips = [b for b in app.button if b.label.startswith('Menurut panduan usaha')]
        assert len(chips) == 4
        chips[0].click().run()
        assert not app.exception
        handler.assert_called_once()
        retrieve.assert_called_once()
        assert any(chunk['sumber_url'] in e.value for e in app.markdown)


def test_unreferenced_guidance_returns_no_reference_without_inventing():
    from services.chat_service import handle_chat_message
    actor = dict(id='test', toko_id='new-local-guide-shop', role='pemilik_toko')
    with patch('services.chat_service.verify_actor', return_value=actor), \
         patch('ai.copilot_tools.verify_actor', return_value=actor), \
         patch('ai.knowledge_rag.verify_actor', return_value=actor), \
         patch('ai.knowledge_rag.get_store_business_context', return_value={'bidang_usaha': 'Bakery'}), \
         patch('ai.knowledge_rag.retrieve_guidance_chunks', return_value=[]), \
         patch('ai.knowledge_rag.ask_text') as llm:
        result = handle_chat_message('test', 'Menurut panduan usaha bakery, bagaimana navigasi antargalaksi?')
    assert result['tool'] == ToolName.GET_BUSINESS_GUIDANCE.value
    assert result['message'] == 'belum ada referensi untuk pertanyaan itu'
    llm.assert_not_called()


@pytest.mark.parametrize('answer', [
    'Maaf, saya tidak menemukan informasi tentang topik tersebut dalam panduan yang tersedia.',
    'Saya belum memiliki referensi untuk pertanyaan ini.',
    'Tidak ada informasi tentang topik tersebut dalam panduan yang tersedia.',
    'Tidak menemukan panduan mengenai topik tersebut.',
    'Maaf, namun topik tersebut tidak relevan dengan panduan usaha ini. Informasi itu tidak terdapat dalam kutipan panduan terverifikasi.',
    'Informasi itu belum dibahas dalam panduan usaha yang tersedia.',
])
def test_model_refusal_is_normalized_without_unrelated_source_footer(answer):
    from ai.knowledge_rag import answer_business_guidance, load_guide_chunks
    chunk = {**load_guide_chunks('bakery')[0], 'score': .4}
    with patch('ai.knowledge_rag.retrieve_guidance_chunks', return_value=[chunk]), \
         patch('ai.knowledge_rag.ask_text', return_value=answer):
        result = answer_business_guidance('bakery', 'Topik yang tidak tersedia')
    assert result['found'] is False
    assert result['answer'] == 'belum ada referensi untuk pertanyaan itu'
    assert result['chunks']  # Retrieval diagnostics remain available internally.
