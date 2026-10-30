import json
from config.settings import get_settings

class LLMUnavailable(RuntimeError):
    pass

def _client():
    try:
        from ollama import Client
    except ImportError as exc:
        raise LLMUnavailable('Paket ollama belum diinstal. Jalankan pip install ollama') from exc
    return Client(host=get_settings()['OLLAMA_HOST'], timeout=120)

def diagnosis():
    settings = get_settings()
    try:
        names = [getattr(m, 'model', '') for m in _client().list().models]
        wanted = settings['OLLAMA_MODEL']
        matched = wanted in names or any(n.startswith(wanted + ':') or (':' in n and wanted == n.split(':')[0]) for n in names)
        return {'ok': matched, 'models': names,
                'message': 'Siap' if matched else f'Model {wanted} tidak ditemukan; jalankan ollama pull {wanted}'}
    except Exception as exc:
        return {'ok': False, 'models': [], 'message': f'Ollama tidak dapat diakses: {exc}'}

def ask_json(system, user, schema=None):
    try:
        response = _client().chat(
            model=get_settings()['OLLAMA_MODEL'],
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            format=schema or 'json', options={'temperature': 0},
        )
        result = json.loads(response.message.content)
        if not isinstance(result, dict):
            raise ValueError('Balasan model bukan objek')
        return result
    except Exception as exc:
        raise LLMUnavailable(f'Ollama gagal: {exc}') from exc

def ask_text(system, user):
    try:
        response = _client().chat(
            model=get_settings()['OLLAMA_MODEL'],
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            options={'temperature': 0.2},
        )
        return response.message.content.strip()
    except Exception as exc:
        raise LLMUnavailable(f'Ollama gagal: {exc}') from exc
