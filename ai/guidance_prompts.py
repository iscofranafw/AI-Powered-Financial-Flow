"""Starter questions derived only from sections of the shipped business guides."""
import re

from ai.knowledge_rag import BUSINESS_GROUP_LABELS, load_guide_chunks, map_business_type


def get_guidance_starters(business_type):
    group = map_business_type(business_type)
    sections = {}
    # Prefer stock, daily cash, SOP and marketing sections; headings and sources
    # themselves are read from data/knowledge/<group>, never invented by an LLM.
    for chunk in load_guide_chunks(group):
        match = re.match(r'^(\d+)\.\s*(.+)$', chunk['judul_bagian'])
        if match and chunk.get('sumber_url', '').startswith('http'):
            sections.setdefault(int(match[1]), (chunk, match[2]))
    result = []
    for number in (3, 5, 7, 8):
        if number not in sections:
            continue
        chunk, heading = sections[number]
        question = f"Menurut panduan usaha {BUSINESS_GROUP_LABELS[group]}, bagaimana {heading.lower().replace('&', 'dan')}?"
        result.append({'question': question, 'group': group, 'section': chunk['judul_bagian'],
                       'filename': chunk['nama_file'], 'source_url': chunk['sumber_url']})
    return result
