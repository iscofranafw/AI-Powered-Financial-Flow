import os
from functools import lru_cache
from dotenv import load_dotenv

@lru_cache(maxsize=1)
def get_settings():
    load_dotenv()
    required = ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'SUPABASE_SERVICE_ROLE_KEY']
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError('Isi konfigurasi .env: ' + ', '.join(missing))
    return {key: os.getenv(key) for key in required} | {
        'OLLAMA_HOST': os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434'),
        'OLLAMA_MODEL': os.getenv('OLLAMA_MODEL', 'llama3:8b'),
    }
