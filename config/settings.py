import os
from functools import lru_cache
from dotenv import load_dotenv


def _int_env(name, default, minimum, maximum):
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


@lru_cache(maxsize=1)
def get_settings():
    load_dotenv()
    required = ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'SUPABASE_SERVICE_ROLE_KEY']
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError('Isi konfigurasi .env: ' + ', '.join(missing))

    provider = os.getenv('OLLAMA_PROVIDER', 'local').strip().lower()
    if provider not in {'local', 'browser'}:
        raise RuntimeError('OLLAMA_PROVIDER harus bernilai local atau browser.')

    allowed_origin = os.getenv('LOCAL_AI_ALLOWED_ORIGIN', 'https://shazraz.my.id').strip().rstrip('/')
    bridge_base_path = os.getenv('LOCAL_AI_BRIDGE_BASE_PATH', '/financial-flow/_local-ai/v1').strip()
    if not bridge_base_path.startswith('/') or '..' in bridge_base_path:
        raise RuntimeError('LOCAL_AI_BRIDGE_BASE_PATH tidak valid.')

    return {key: os.getenv(key) for key in required} | {
        'OLLAMA_PROVIDER': provider,
        'OLLAMA_HOST': os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434'),
        'OLLAMA_MODEL': os.getenv('OLLAMA_MODEL', 'llama3:8b'),
        'LOCAL_AI_BRIDGE_PORT': _int_env('LOCAL_AI_BRIDGE_PORT', 8511, 1024, 65535),
        'LOCAL_AI_ALLOWED_ORIGIN': allowed_origin,
        'LOCAL_AI_BRIDGE_BASE_PATH': bridge_base_path,
        'LOCAL_AI_BRIDGE_TIMEOUT_SECONDS': _int_env('LOCAL_AI_BRIDGE_TIMEOUT_SECONDS', 300, 10, 600),
    }
