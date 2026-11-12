"""Real local Knowledge RAG through the chat service, on new isolated test identities."""
import json
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai.guidance_prompts import get_guidance_starters
from ai.knowledge_rag import NO_REFERENCE_MESSAGE
from services.chat_service import handle_chat_message


def main():
    out = ROOT/'evidence/post_push_improvements'
    previous = json.loads((out/'guidance_live.json').read_text(encoding='utf-8')) if (out/'guidance_live.json').exists() else []
    verified = {(r['group'], r['prompt']['question']): r['response'] for r in previous
                if r.get('prompt') and r.get('response', {}).get('data', {}).get('found')}
    results = []
    with tempfile.TemporaryDirectory() as temporary:
        for group in ('bakery', 'retail', 'warung_makan', 'bengkel_motor', 'lainnya'):
            actor = dict(id=f'new-local-{group}-owner', toko_id=f'new-local-{group}-shop', role='pemilik_toko')
            ctx = dict(bidang_usaha=group, nama=f'New local {group} test shop')
            with ExitStack() as stack:
                for target in ('services.chat_service.verify_actor', 'ai.copilot_tools.verify_actor', 'ai.knowledge_rag.verify_actor'):
                    stack.enter_context(patch(target, return_value=actor))
                stack.enter_context(patch('ai.knowledge_rag.get_store_business_context', return_value=ctx))
                stack.enter_context(patch('ai.knowledge_rag.DEFAULT_KNOWLEDGE_STORE_DIR', Path(temporary)))
                for prompt in get_guidance_starters(group):
                    response = verified.get((group, prompt['question'])) or handle_chat_message('local-test-only', prompt['question'])
                    record = dict(group=group, prompt=prompt, response=response)
                    results.append(record)
                    (out/'guidance_live.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
                    assert response['tool'] == 'get_business_guidance'
                    assert response['citations'] and response['data']['found'], (group, prompt, response['message'])
                    assert all(c['nama_file'] == prompt['filename'] for c in response['citations'])
                    print(group, prompt['section'], 'cited', len(response['citations']), flush=True)
                unknown = 'Menurut panduan usaha '+group+', bagaimana navigasi pesawat antargalaksi dan reaktor fusi tokamak?'
                response = handle_chat_message('local-test-only', unknown)
                results.append(dict(group=group, query=unknown, out_of_reference=True, response=response))
                (out/'guidance_live.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
                assert response['message'] == NO_REFERENCE_MESSAGE, response['message']
                print(group, 'out-of-reference: correct abstention', flush=True)


if __name__ == '__main__':
    main()
