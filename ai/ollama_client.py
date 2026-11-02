import json
import time
import logging
from config.settings import get_settings

class LLMUnavailable(RuntimeError):
    pass

_CLIENT_INSTANCE = None
_CIRCUIT_OPEN_UNTIL = 0.0
CIRCUIT_COOLDOWN_SECONDS = 30.0


def _check_circuit_breaker():
    global _CIRCUIT_OPEN_UNTIL
    if time.time() < _CIRCUIT_OPEN_UNTIL:
        remaining = int(_CIRCUIT_OPEN_UNTIL - time.time())
        raise LLMUnavailable(f'Layanan Ollama sedang tidak tersedia (circuit breaker aktif, coba lagi dalam {remaining}d).')


def _record_failure():
    global _CIRCUIT_OPEN_UNTIL
    _CIRCUIT_OPEN_UNTIL = time.time() + CIRCUIT_COOLDOWN_SECONDS


def _record_success():
    global _CIRCUIT_OPEN_UNTIL
    _CIRCUIT_OPEN_UNTIL = 0.0


def _client(timeout: float = 25.0):
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is not None:
        return _CLIENT_INSTANCE
    try:
        from ollama import Client
    except ImportError as exc:
        raise LLMUnavailable('Paket ollama belum diinstal. Jalankan pip install ollama') from exc
    try:
        s = get_settings()
        _CLIENT_INSTANCE = Client(host=s['OLLAMA_HOST'], timeout=timeout)
        return _CLIENT_INSTANCE
    except Exception as exc:
        _record_failure()
        raise LLMUnavailable(f'Gagal menginisialisasi klien Ollama: {exc}') from exc


def diagnosis():
    """Lightweight availability check with short timeout to prevent UI freezes."""
    try:
        settings = get_settings()
    except Exception as exc:
        return {'ok': False, 'models': [], 'message': f'Konfigurasi belum lengkap: {exc}'}

    try:
        from ollama import Client
        fast_client = Client(host=settings['OLLAMA_HOST'], timeout=3.0)
        models_res = fast_client.list()
        names = [getattr(m, 'model', '') for m in models_res.models]
        wanted = settings['OLLAMA_MODEL']
        matched = wanted in names or any(n.startswith(wanted + ':') or (':' in n and wanted == n.split(':')[0]) for n in names)
        _record_success()
        return {
            'ok': matched,
            'models': names,
            'message': 'Siap' if matched else f'Model {wanted} tidak ditemukan; jalankan ollama pull {wanted}'
        }
    except Exception as exc:
        _record_failure()
        return {'ok': False, 'models': [], 'message': f'Ollama tidak dapat diakses (timeout/offline): {exc}'}


def ask_json(system, user, schema=None, num_predict=256):
    _check_circuit_breaker()
    try:
        response = _client().chat(
            model=get_settings()['OLLAMA_MODEL'],
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            format=schema or 'json',
            options={'temperature': 0, 'num_predict': num_predict},
            keep_alive='10m',
        )
        result = json.loads(response.message.content)
        if not isinstance(result, dict):
            raise ValueError('Balasan model bukan objek')
        _record_success()
        return result
    except Exception as exc:
        _record_failure()
        raise LLMUnavailable(f'Ollama gagal: {exc}') from exc


def ask_text(system, user, num_predict=512):
    _check_circuit_breaker()
    try:
        response = _client().chat(
            model=get_settings()['OLLAMA_MODEL'],
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            options={'temperature': 0.2, 'num_predict': num_predict},
            keep_alive='10m',
        )
        _record_success()
        return response.message.content.strip()
    except Exception as exc:
        _record_failure()
        raise LLMUnavailable(f'Ollama gagal: {exc}') from exc
