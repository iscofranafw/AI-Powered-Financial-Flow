from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from ai.embedding_pipeline import EMBEDDING_DIM, embed_batch, embed_text, get_model_name
from ai.ollama_client import LLMUnavailable, ask_text
from ai.store_context import get_store_business_context
from services.auth_service import verify_actor

CANONICAL_BUSINESS_GROUPS = (
    'bakery',
    'warung_makan',
    'retail',
    'bengkel_motor',
    'lainnya',
)

BUSINESS_GROUP_LABELS = {
    'bakery': 'Bakery & Toko Roti',
    'warung_makan': 'Warung Makan & Kuliner',
    'retail': 'Retail, Toko Kelontong & Grosir',
    'bengkel_motor': 'Bengkel Motor & Sparepart',
    'lainnya': 'UMKM Umum (Lainnya)',
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE_DIR = _PROJECT_ROOT / 'data' / 'knowledge'
DEFAULT_KNOWLEDGE_STORE_DIR = _PROJECT_ROOT / 'knowledge_store'

DEFAULT_CHUNK_MIN_CHARS = 500
DEFAULT_CHUNK_MAX_CHARS = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.30
NO_REFERENCE_MESSAGE = 'belum ada referensi untuk pertanyaan itu'
UNVERIFIED_REFERENCE_MARKER = 'belum ada referensi terverifikasi'

_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\s)]+)\)')
_BARE_URL_RE = re.compile(r'(https?://[^\s),]+)')
_SECTION_HEADING_RE = re.compile(r'^##\s+(.+)$')
_SUBSECTION_HEADING_RE = re.compile(r'^###\s+(.+)$')

_INDEX_CACHE: dict[tuple[str, str], tuple[int, faiss.Index, list[dict[str, Any]]]] = {}

_BAKERY_KEYWORDS = ('bakery', 'roti', 'kue', 'pastry', 'cake', 'boulangerie', 'patisserie')
_WARUNG_KEYWORDS = (
    'warung makan',
    'warung_makan',
    'warung',
    'kuliner',
    'rumah makan',
    'restoran',
    'resto',
    'kafe',
    'cafe',
    'kedai',
    'warteg',
    'catering',
    'katering',
    'fnb',
    'f&b',
    'makanan',
)
_RETAIL_KEYWORDS = (
    'retail',
    'ritel',
    'grosir',
    'kelontong',
    'sembako',
    'minimarket',
    'mart',
    'swalayan',
    'eceran',
    'agen',
    'toko',
    'dagang',
)
_BENGKEL_KEYWORDS = (
    'bengkel motor',
    'bengkel_motor',
    'bengkel',
    'servis motor',
    'service motor',
    'sparepart',
    'suku cadang',
    'otomotif',
    'motor',
)


def map_business_type(bidang_usaha: str | None) -> str:
    """Map free-text toko.bidang_usaha in memory to a canonical knowledge folder without modifying DB."""
    if not bidang_usaha:
        return 'lainnya'

    normalized = re.sub(r'[_\-/]+', ' ', str(bidang_usaha).strip().lower())
    normalized = re.sub(r'\s+', ' ', normalized)
    if not normalized:
        return 'lainnya'

    if normalized in CANONICAL_BUSINESS_GROUPS:
        return normalized
    if any(kw in normalized for kw in _BAKERY_KEYWORDS):
        return 'bakery'
    if any(kw in normalized for kw in _BENGKEL_KEYWORDS):
        return 'bengkel_motor'
    if any(kw in normalized for kw in _WARUNG_KEYWORDS):
        return 'warung_makan'
    if any(kw in normalized for kw in _RETAIL_KEYWORDS):
        return 'retail'
    return 'lainnya'


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for _, url in _MD_LINK_RE.findall(text):
        cleaned = url.rstrip('.,;')
        if cleaned not in urls:
            urls.append(cleaned)
    for url in _BARE_URL_RE.findall(text):
        cleaned = url.rstrip('.,;)')
        if cleaned not in urls:
            urls.append(cleaned)
    return urls


def _strip_source_line_prefix(line: str) -> bool:
    return line.strip().lower().startswith('sumber:')


def parse_markdown_sections(markdown_text: str) -> list[dict[str, Any]]:
    """Split a Markdown guide document into level-2 sections with extracted source URLs."""
    sections: list[dict[str, Any]] = []
    current_title = 'Pendahuluan'
    current_lines: list[str] = []
    current_urls: list[str] = []

    def _flush():
        body = '\n'.join(current_lines).strip()
        if not body:
            return
        if current_title == 'Pendahuluan' and len(body) < 350 and not current_urls:
            return
        all_urls = list(current_urls)
        for u in _extract_urls(body):
            if u not in all_urls:
                all_urls.append(u)
        sumber_url = ', '.join(all_urls) if all_urls else UNVERIFIED_REFERENCE_MARKER
        sections.append(
            {
                'judul_bagian': current_title,
                'body': body,
                'sumber_urls': all_urls,
                'sumber_url': sumber_url,
            }
        )

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        sec_match = _SECTION_HEADING_RE.match(line.strip())
        if sec_match:
            _flush()
            current_title = sec_match.group(1).strip()
            current_lines = []
            current_urls = []
            continue

        if _strip_source_line_prefix(line):
            for u in _extract_urls(line):
                if u not in current_urls:
                    current_urls.append(u)
            continue

        if line.strip().startswith('# ') and not sections and not current_lines:
            continue

        sub_match = _SUBSECTION_HEADING_RE.match(line.strip())
        if sub_match:
            current_lines.append(sub_match.group(1).strip())
            continue

        current_lines.append(line)

    _flush()
    return sections


def _split_long_block(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    pieces: list[str] = []
    buf = ''
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            if buf:
                pieces.append(buf)
                buf = ''
            words = sent.split()
            wbuf = ''
            for w in words:
                candidate = f'{wbuf} {w}'.strip() if wbuf else w
                if len(candidate) <= max_chars:
                    wbuf = candidate
                else:
                    if wbuf:
                        pieces.append(wbuf)
                    wbuf = w[:max_chars]
            if wbuf:
                pieces.append(wbuf)
            continue
        candidate = f'{buf} {sent}'.strip() if buf else sent
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                pieces.append(buf)
            buf = sent
    if buf:
        pieces.append(buf)
    return pieces


def chunk_section_text(
    body: str,
    min_chars: int = DEFAULT_CHUNK_MIN_CHARS,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Chunk a Markdown section into ~500-800 character overlapping slices."""
    cleaned_body = re.sub(r'\n{3,}', '\n\n', body.strip())
    if not cleaned_body:
        return []
    if len(cleaned_body) <= max_chars:
        return [cleaned_body]

    raw_blocks = [b.strip() for b in cleaned_body.split('\n\n') if b.strip()]
    units: list[str] = []
    for block in raw_blocks:
        units.extend(_split_long_block(block, max_chars))

    chunks: list[str] = []
    current = ''
    for unit in units:
        candidate = f'{current}\n\n{unit}'.strip() if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap:].lstrip() if overlap > 0 else ''
            space_idx = tail.find(' ')
            if space_idx > 0:
                tail = tail[space_idx + 1 :].strip()
            if tail and len(tail) + 2 + len(unit) <= max_chars:
                current = f'{tail}\n\n{unit}'
            else:
                current = unit
        else:
            chunks.append(unit[:max_chars])
            current = ''

    if current:
        if chunks and len(current) < min_chars // 2 and len(chunks[-1]) + 2 + len(current) <= max_chars:
            chunks[-1] = f'{chunks[-1]}\n\n{current}'
        else:
            chunks.append(current)

    return chunks


def load_guide_chunks(
    jenis_usaha: str,
    knowledge_dir: Path | str | None = None,
    min_chars: int = DEFAULT_CHUNK_MIN_CHARS,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Read and chunk all Markdown operational guides for one canonical business group."""
    group = jenis_usaha if jenis_usaha in CANONICAL_BUSINESS_GROUPS else map_business_type(jenis_usaha)
    base_dir = Path(knowledge_dir) if knowledge_dir is not None else DEFAULT_KNOWLEDGE_DIR
    group_dir = base_dir / group
    if not group_dir.exists():
        return []

    all_chunks: list[dict[str, Any]] = []
    for md_path in sorted(group_dir.glob('*.md')):
        raw_md = md_path.read_text(encoding='utf-8')
        sections = parse_markdown_sections(raw_md)
        for sec in sections:
            slices = chunk_section_text(
                sec['body'],
                min_chars=min_chars,
                max_chars=max_chars,
                overlap=overlap,
            )
            for idx, text_slice in enumerate(slices):
                all_chunks.append(
                    {
                        'jenis_usaha': group,
                        'nama_file': md_path.name,
                        'judul_bagian': sec['judul_bagian'],
                        'sumber_url': sec['sumber_url'],
                        'sumber_urls': list(sec['sumber_urls']),
                        'chunk_index': idx,
                        'text': text_slice,
                    }
                )
    return all_chunks


def _normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _group_store_paths(group: str, store_dir: Path | str | None = None) -> tuple[Path, Path]:
    root = Path(store_dir) if store_dir is not None else DEFAULT_KNOWLEDGE_STORE_DIR
    folder = root / group
    return folder / 'index.faiss', folder / 'meta.json'


def build_knowledge_index(
    jenis_usaha: str | None = None,
    knowledge_dir: Path | str | None = None,
    store_dir: Path | str | None = None,
    min_chars: int = DEFAULT_CHUNK_MIN_CHARS,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, dict[str, Any]]:
    """Build separate FAISS IndexFlatIP knowledge indexes per business group."""
    targets = (
        [jenis_usaha if jenis_usaha in CANONICAL_BUSINESS_GROUPS else map_business_type(jenis_usaha)]
        if jenis_usaha
        else list(CANONICAL_BUSINESS_GROUPS)
    )
    results: dict[str, dict[str, Any]] = {}

    for group in targets:
        chunks = load_guide_chunks(
            group,
            knowledge_dir=knowledge_dir,
            min_chars=min_chars,
            max_chars=max_chars,
            overlap=overlap,
        )
        index_path, meta_path = _group_store_paths(group, store_dir=store_dir)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if chunks:
            embed_inputs = [f"{c['judul_bagian']}\n{c['text']}" for c in chunks]
            vecs = _normalize_vectors(embed_batch(embed_inputs))
            index.add(vecs)

        faiss.write_index(index, str(index_path))
        payload = {
            'jenis_usaha': group,
            'built_at': datetime.now(timezone.utc).isoformat(),
            'embedding_model': get_model_name(),
            'embedding_dim': EMBEDDING_DIM,
            'chunk_count': len(chunks),
            'chunks': chunks,
        }
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        _INDEX_CACHE.pop((str(index_path.parent.parent), group), None)
        results[group] = {
            'jenis_usaha': group,
            'chunk_count': len(chunks),
            'index_path': str(index_path),
            'meta_path': str(meta_path),
        }

    return results


def load_knowledge_index(
    jenis_usaha: str,
    store_dir: Path | str | None = None,
    auto_build: bool = True,
) -> tuple[faiss.Index | None, list[dict[str, Any]]]:
    """Load the isolated FAISS index and metadata for a single business group."""
    group = jenis_usaha if jenis_usaha in CANONICAL_BUSINESS_GROUPS else map_business_type(jenis_usaha)
    resolved_root = str(Path(store_dir) if store_dir is not None else DEFAULT_KNOWLEDGE_STORE_DIR)
    index_path, meta_path = _group_store_paths(group, store_dir=resolved_root)

    if (not index_path.exists() or not meta_path.exists()) and auto_build:
        build_knowledge_index(jenis_usaha=group, store_dir=resolved_root)

    if not index_path.exists() or not meta_path.exists():
        return None, []

    mtime_ns = max(index_path.stat().st_mtime_ns, meta_path.stat().st_mtime_ns)
    cache_key = (resolved_root, group)
    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1], cached[2]

    index = faiss.read_index(str(index_path))
    meta_data = json.loads(meta_path.read_text(encoding='utf-8'))
    chunks = meta_data.get('chunks', []) if isinstance(meta_data, dict) else []
    _INDEX_CACHE[cache_key] = (mtime_ns, index, chunks)
    return index, chunks


def get_knowledge_index_status(store_dir: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Inspect build status and chunk count of each business group's knowledge index."""
    resolved_root = Path(store_dir) if store_dir is not None else DEFAULT_KNOWLEDGE_STORE_DIR
    status: dict[str, dict[str, Any]] = {}
    for group in CANONICAL_BUSINESS_GROUPS:
        index_path, meta_path = _group_store_paths(group, store_dir=resolved_root)
        if index_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            status[group] = {
                'ready': True,
                'chunk_count': int(meta.get('chunk_count', 0)),
                'built_at': meta.get('built_at'),
                'embedding_model': meta.get('embedding_model'),
                'embedding_dim': meta.get('embedding_dim', EMBEDDING_DIM),
            }
        else:
            status[group] = {
                'ready': False,
                'chunk_count': 0,
                'built_at': None,
                'embedding_model': get_model_name(),
                'embedding_dim': EMBEDDING_DIM,
            }
    return status


def retrieve_guidance_chunks(
    jenis_usaha: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    store_dir: Path | str | None = None,
    auto_build: bool = True,
) -> list[dict[str, Any]]:
    """Search only the specified business group's FAISS knowledge index."""
    cleaned_query = str(query or '').strip()
    if not cleaned_query:
        return []

    group = jenis_usaha if jenis_usaha in CANONICAL_BUSINESS_GROUPS else map_business_type(jenis_usaha)
    index, chunks = load_knowledge_index(group, store_dir=store_dir, auto_build=auto_build)
    if index is None or not chunks or index.ntotal == 0:
        return []

    k = max(1, min(int(top_k), len(chunks), index.ntotal))
    qvec = _normalize_vectors(embed_text(cleaned_query))
    scores, indices = index.search(qvec, k)

    results: list[dict[str, Any]] = []
    for raw_score, raw_idx in zip(scores[0], indices[0]):
        if raw_idx < 0 or raw_idx >= len(chunks):
            continue
        score = float(raw_score)
        if score < float(min_score):
            continue
        item = dict(chunks[raw_idx])
        item['score'] = round(score, 4)
        item['jenis_usaha'] = group
        results.append(item)

    results.sort(key=lambda r: r['score'], reverse=True)
    return results


def _dedupe_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    citations: list[dict[str, Any]] = []
    for chunk in chunks:
        key = (
            str(chunk.get('nama_file', '')),
            str(chunk.get('judul_bagian', '')),
            str(chunk.get('sumber_url', '')),
        )
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                'nama_file': key[0],
                'judul_bagian': key[1],
                'sumber_url': key[2],
                'score': float(chunk.get('score', 0.0)),
            }
        )
    return citations


def _format_citation_footer(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return ''
    lines = ['**Rujukan Panduan:**']
    for item in citations:
        lines.append(
            f"- {item['nama_file']} — *{item['judul_bagian']}* | Sumber: {item['sumber_url']}"
        )
    return '\n'.join(lines)


def answer_business_guidance(
    jenis_usaha: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    store_dir: Path | str | None = None,
    store_name: str = '',
    raw_bidang_usaha: str = '',
) -> dict[str, Any]:
    """Answer an operational guidance question strictly grounded in the store's business group corpus."""
    group = jenis_usaha if jenis_usaha in CANONICAL_BUSINESS_GROUPS else map_business_type(jenis_usaha)
    chunks = retrieve_guidance_chunks(
        group,
        query,
        top_k=top_k,
        min_score=min_score,
        store_dir=store_dir,
    )

    if not chunks:
        return {
            'ok': True,
            'found': False,
            'jenis_usaha': group,
            'raw_bidang_usaha': raw_bidang_usaha or group,
            'store_name': store_name,
            'answer': NO_REFERENCE_MESSAGE,
            'citations': [],
            'chunks': [],
        }

    citations = _dedupe_citations(chunks)
    context_blocks: list[str] = []
    for idx, ch in enumerate(chunks, start=1):
        context_blocks.append(
            f"[{idx}] File: {ch['nama_file']} | Bagian: {ch['judul_bagian']} | "
            f"Sumber: {ch['sumber_url']} | Skor: {ch['score']:.2f}\n{ch['text']}"
        )
    context_text = '\n\n'.join(context_blocks)

    system_prompt = (
        'Anda adalah asisten panduan operasional UMKM Indonesia untuk aplikasi Financial Flow. '
        'Jawab pertanyaan pengguna dalam bahasa Indonesia yang ringkas, praktis, dan terstruktur '
        'HANYA berdasarkan kutipan panduan terverifikasi yang diberikan.\n'
        'Aturan ketat:\n'
        '1. Jangan mengarang fakta, angka, aturan hukum, atau rekomendasi di luar kutipan panduan.\n'
        '2. Jika kutipan menyatakan "belum ada referensi terverifikasi" untuk subtopik tertentu, '
        'sampaikan hal itu secara jujur.\n'
        '3. Jika kutipan belum menjawab inti pertanyaan dengan cukup, jawab: '
        f'"{NO_REFERENCE_MESSAGE}".\n'
        '4. Selalu sertakan nama bagian panduan dan URL sumber yang dirujuk.'
    )
    user_prompt = (
        f"Kelompok Usaha: {BUSINESS_GROUP_LABELS.get(group, group)} ({group})\n"
        f"Nama Toko: {store_name or '-'}\n"
        f"Bidang Usaha Toko: {raw_bidang_usaha or group}\n\n"
        f"Pertanyaan:\n{query}\n\n"
        f"Kutipan Panduan Terverifikasi:\n{context_text}"
    )

    llm_used = True
    try:
        answer_text = ask_text(system_prompt, user_prompt, num_predict=512).strip()
        if not answer_text:
            raise LLMUnavailable('Jawaban LLM kosong')
    except LLMUnavailable as exc:
        llm_used = False
        snippets = [
            f"- **{c['judul_bagian']}** ({c['nama_file']}): {c['text']}"
            for c in chunks
        ]
        answer_text = (
            f"Layanan Ollama lokal sedang tidak aktif ({exc}). "
            'Berikut kutipan langsung dari panduan usaha yang sesuai:\n\n'
            + '\n\n'.join(snippets)
        )

    if NO_REFERENCE_MESSAGE in answer_text.lower() and len(answer_text.strip()) <= 64:
        return {
            'ok': True,
            'found': False,
            'jenis_usaha': group,
            'raw_bidang_usaha': raw_bidang_usaha or group,
            'store_name': store_name,
            'answer': NO_REFERENCE_MESSAGE,
            'citations': citations,
            'chunks': chunks,
            'llm_used': llm_used,
        }

    footer = _format_citation_footer(citations)
    has_url_already = any(
        c['sumber_url'] in answer_text
        for c in citations
        if c['sumber_url'] and c['sumber_url'] != UNVERIFIED_REFERENCE_MARKER
    )
    final_answer = answer_text if has_url_already or not footer else f'{answer_text}\n\n{footer}'

    return {
        'ok': True,
        'found': True,
        'jenis_usaha': group,
        'raw_bidang_usaha': raw_bidang_usaha or group,
        'store_name': store_name,
        'answer': final_answer,
        'citations': citations,
        'chunks': chunks,
        'llm_used': llm_used,
    }


def answer_guidance_for_actor(
    token: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    store_dir: Path | str | None = None,
    jenis_usaha_override: str | None = None,
) -> dict[str, Any]:
    """Authenticate actor via verify_actor and query only that store's isolated knowledge base."""
    actor = verify_actor(token, ['superadmin', 'pemilik_toko', 'kasir'])
    store_id = actor.get('toko_id')
    ctx = get_store_business_context(store_id) if store_id else {}
    raw_bidang = (ctx.get('bidang_usaha') or '').strip() if ctx else ''
    store_name = (ctx.get('nama_toko') or ctx.get('nama') or '').strip() if ctx else ''

    # Store role uses its own toko.bidang_usaha; superadmin without store context may pick a view group.
    if jenis_usaha_override and actor.get('role') == 'superadmin' and not store_id:
        resolved_group = map_business_type(jenis_usaha_override)
    else:
        resolved_group = map_business_type(raw_bidang)

    result = answer_business_guidance(
        resolved_group,
        query=query,
        top_k=top_k,
        min_score=min_score,
        store_dir=store_dir,
        store_name=store_name,
        raw_bidang_usaha=raw_bidang or resolved_group,
    )
    result['toko_id'] = store_id
    return result
