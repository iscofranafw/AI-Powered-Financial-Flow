from pathlib import Path
from unittest.mock import patch

import pytest

from ai import knowledge_rag
from ai.copilot_planner import plan_intent
from ai.copilot_schema import ToolName
from ai.copilot_tools import dispatch_tool
from ai.financial_chat import Intent
from ai.ollama_client import LLMUnavailable
from services.auth_service import AccessError
from services.chat_service import handle_chat_message


def test_map_business_type_covers_five_canonical_groups():
    assert knowledge_rag.map_business_type('Bakery & Pastry') == 'bakery'
    assert knowledge_rag.map_business_type('Toko Roti Rumahan') == 'bakery'
    assert knowledge_rag.map_business_type('Warung Makan Padang') == 'warung_makan'
    assert knowledge_rag.map_business_type('Kedai Kopi & Kuliner') == 'warung_makan'
    assert knowledge_rag.map_business_type('Grosir Sembako') == 'retail'
    assert knowledge_rag.map_business_type('Toko Kelontong & Minimarket') == 'retail'
    assert knowledge_rag.map_business_type('Retail Eceran') == 'retail'
    assert knowledge_rag.map_business_type('Bengkel Motor & Sparepart') == 'bengkel_motor'
    assert knowledge_rag.map_business_type('Jasa Laundry Kiloan') == 'lainnya'
    assert knowledge_rag.map_business_type(None) == 'lainnya'
    assert knowledge_rag.map_business_type('') == 'lainnya'


def test_all_five_business_guides_present_and_chunked_within_limits():
    for group in knowledge_rag.CANONICAL_BUSINESS_GROUPS:
        chunks = knowledge_rag.load_guide_chunks(group)
        assert chunks, f'Guide chunks missing for {group}'
        section_titles = {c['judul_bagian'] for c in chunks}
        # Expect all 8 operational guide sections
        assert len(section_titles) >= 8
        for c in chunks:
            assert c['jenis_usaha'] == group
            assert c['nama_file'].endswith('.md')
            assert c['judul_bagian']
            assert c['sumber_url']
            assert 1 <= len(c['text']) <= knowledge_rag.DEFAULT_CHUNK_MAX_CHARS + 50


def test_built_knowledge_store_indexes_exist_for_all_groups():
    status = knowledge_rag.get_knowledge_index_status()
    assert set(status.keys()) == set(knowledge_rag.CANONICAL_BUSINESS_GROUPS)
    for group, info in status.items():
        assert info['ready'] is True, f'Index missing for {group}'
        assert info['chunk_count'] >= 8
        assert info['embedding_dim'] == 384


def test_knowledge_rag_strict_per_business_group_isolation(tmp_path, monkeypatch):
    # Test using deterministic lightweight embeddings in isolated store_dir
    monkeypatch.setenv('USE_FALLBACK_EMBEDDINGS', '1')
    built = knowledge_rag.build_knowledge_index(store_dir=tmp_path)
    assert set(built.keys()) == set(knowledge_rag.CANONICAL_BUSINESS_GROUPS)

    bakery_hits = knowledge_rag.retrieve_guidance_chunks(
        'bakery',
        'stok bahan baku tepung wip barang jadi hpp roti',
        top_k=4,
        min_score=0.05,
        store_dir=tmp_path,
        auto_build=False,
    )
    assert bakery_hits
    assert all(h['jenis_usaha'] == 'bakery' for h in bakery_hits)
    assert all('oli bekas' not in h['text'].lower() for h in bakery_hits)

    retail_hits = knowledge_rag.retrieve_guidance_chunks(
        'retail',
        'perbedaan margin grosir dan eceran multi pricing overstock sembako',
        top_k=4,
        min_score=0.05,
        store_dir=tmp_path,
        auto_build=False,
    )
    assert retail_hits
    assert all(h['jenis_usaha'] == 'retail' for h in retail_hits)


def test_below_min_score_returns_no_reference_without_calling_llm(tmp_path, monkeypatch):
    monkeypatch.setenv('USE_FALLBACK_EMBEDDINGS', '1')
    knowledge_rag.build_knowledge_index(jenis_usaha='bakery', store_dir=tmp_path)

    with patch('ai.knowledge_rag.ask_text') as mock_ask:
        res = knowledge_rag.answer_business_guidance(
            'bakery',
            query='topik sama sekali tidak nyambung dengan kue ataupun usaha',
            min_score=0.99,  # Force below-threshold gate
            store_dir=tmp_path,
        )

    assert res['found'] is False
    assert res['answer'] == 'belum ada referensi untuk pertanyaan itu'
    assert res['citations'] == []
    mock_ask.assert_not_called()


def test_grounded_answer_calls_local_llm_with_citations(tmp_path, monkeypatch):
    monkeypatch.setenv('USE_FALLBACK_EMBEDDINGS', '1')
    knowledge_rag.build_knowledge_index(jenis_usaha='bengkel_motor', store_dir=tmp_path)

    with patch(
        'ai.knowledge_rag.ask_text',
        return_value='Gunakan work order (nota servis) dan pisahkan oli bekas ke drum limbah B3 sesuai panduan.',
    ) as mock_ask:
        res = knowledge_rag.answer_business_guidance(
            'bengkel_motor',
            query='pembukuan nota servis sparepart limbah oli bengkel motor',
            top_k=3,
            min_score=0.01,
            store_dir=tmp_path,
            store_name='Bengkel Jaya',
            raw_bidang_usaha='Bengkel Motor',
        )

    assert res['found'] is True
    assert res['jenis_usaha'] == 'bengkel_motor'
    assert res['citations']
    assert 'Rujukan Panduan:' in res['answer']
    assert 'http' in res['answer']
    mock_ask.assert_called_once()


def test_llm_unavailable_falls_back_gracefully_with_citations(tmp_path, monkeypatch):
    monkeypatch.setenv('USE_FALLBACK_EMBEDDINGS', '1')
    knowledge_rag.build_knowledge_index(jenis_usaha='warung_makan', store_dir=tmp_path)

    with patch('ai.knowledge_rag.ask_text', side_effect=LLMUnavailable('Ollama offline')):
        res = knowledge_rag.answer_business_guidance(
            'warung_makan',
            query='food cost bahan basah kering fifo warung makan',
            top_k=3,
            min_score=0.01,
            store_dir=tmp_path,
        )

    assert res['found'] is True
    assert res['llm_used'] is False
    assert 'Ollama lokal sedang tidak aktif' in res['answer']
    assert res['citations']


def test_actor_isolation_and_copilot_get_business_guidance(monkeypatch):
    monkeypatch.setenv('USE_FALLBACK_EMBEDDINGS', '1')
    decision = plan_intent('Bagaimana cara mengelola stok roti sisa dan HPP di bakery?')
    assert decision.tool_name == ToolName.GET_BUSINESS_GUIDANCE

    bakery_actor = {'id': 'u1', 'role': 'pemilik_toko', 'toko_id': 'toko_bakery_1'}
    retail_actor = {'id': 'u2', 'role': 'pemilik_toko', 'toko_id': 'toko_retail_1'}

    with patch('ai.knowledge_rag.verify_actor', return_value=bakery_actor), \
         patch('ai.copilot_tools.verify_actor', return_value=bakery_actor), \
         patch('services.chat_service.verify_actor', return_value=bakery_actor), \
         patch('ai.knowledge_rag.get_store_business_context', return_value={'id': 'toko_bakery_1', 'nama': 'Tewiwi Bakery', 'bidang_usaha': 'Bakery'}), \
         patch('ai.knowledge_rag.retrieve_guidance_chunks', return_value=[{
             'jenis_usaha': 'bakery',
             'nama_file': 'panduan_operasional_bakery.md',
             'judul_bagian': '3. Pengelolaan Stok & Supplier',
             'sumber_url': 'https://pos.mekari.com/blog/bisnis-bakery',
             'text': 'Terapkan FIFO dan FEFO untuk tepung dan bahan susu bakery.',
             'score': 0.72,
         }]) as mock_retrieve, \
         patch('ai.knowledge_rag.ask_text', return_value='Terapkan FIFO dan FEFO agar roti tidak banyak sisa.'):
        chat_res = handle_chat_message('tok_bakery', 'Bagaimana cara mengelola stok roti sisa dan bahan baku?')

    assert chat_res['intent'] == Intent.QUERY_BUSINESS_GUIDANCE
    assert chat_res['tool'] == 'get_business_guidance'
    assert chat_res['metadata']['jenis_usaha'] == 'bakery'
    mock_retrieve.assert_called_once()
    assert mock_retrieve.call_args.args[0] == 'bakery'

    with patch('ai.knowledge_rag.verify_actor', return_value=retail_actor), \
         patch('ai.knowledge_rag.get_store_business_context', return_value={'id': 'toko_retail_1', 'nama': 'Tryaki Mart', 'bidang_usaha': 'Grosir'}):
        res_retail = knowledge_rag.answer_guidance_for_actor('tok_retail', 'cara menghindari overstock grosir', min_score=0.99)

    assert res_retail['jenis_usaha'] == 'retail'
    assert res_retail['answer'] == 'belum ada referensi untuk pertanyaan itu'

    with patch('ai.knowledge_rag.verify_actor', side_effect=AccessError('Token tidak valid')), \
         patch('ai.copilot_tools.verify_actor', side_effect=AccessError('Token tidak valid')):
        with pytest.raises(AccessError):
            knowledge_rag.answer_guidance_for_actor('bad_tok', 'cara kelola stok')
        # Dispatch tool returns clean failure response without crash
        d_res = dispatch_tool('bad_tok', decision)
        assert d_res['success'] is False
        assert 'Akses ditolak' in d_res['message']
