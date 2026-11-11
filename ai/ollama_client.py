import json
import time
import logging
from config.settings import get_settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


_CLIENT_INSTANCE = None
_CIRCUIT_OPEN_UNTIL = 0.0
CIRCUIT_COOLDOWN_SECONDS = 30.0


def _settings():
    return get_settings()


def _provider():
    return _settings().get('OLLAMA_PROVIDER', 'local')


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
    if _provider() == 'browser':
        raise LLMUnavailable('Ollama browser tidak dapat dipanggil langsung dari server.')
    if _CLIENT_INSTANCE is not None:
        return _CLIENT_INSTANCE
    try:
        from ollama import Client
    except ImportError as exc:
        raise LLMUnavailable('Paket ollama belum diinstal. Jalankan pip install ollama') from exc
    try:
        settings = _settings()
        _CLIENT_INSTANCE = Client(host=settings['OLLAMA_HOST'], timeout=timeout)
        return _CLIENT_INSTANCE
    except Exception as exc:
        _record_failure()
        raise LLMUnavailable(f'Gagal menginisialisasi klien Ollama: {exc}') from exc


def _browser_chat(messages, response_format, options):
    settings = _settings()
    try:
        from ai.local_ai_bridge import LocalAIBridgeError, request_chat
        return request_chat(
            messages=messages,
            model=settings['OLLAMA_MODEL'],
            response_format=response_format,
            options=options,
            timeout=settings.get('LOCAL_AI_BRIDGE_TIMEOUT_SECONDS', 240),
        )
    except LocalAIBridgeError as exc:
        raise LLMUnavailable(str(exc)) from exc


def diagnosis():
    """Check the selected provider without ever probing VPS Ollama in browser mode."""
    try:
        settings = _settings()
    except Exception as exc:
        return {'ok': False, 'state': 'error', 'models': [], 'message': f'Konfigurasi belum lengkap: {exc}'}

    if settings.get('OLLAMA_PROVIDER', 'local') == 'browser':
        try:
            from ai.local_ai_bridge import diagnosis as bridge_diagnosis
            status = bridge_diagnosis()
            status.setdefault('models', [])
            status.setdefault('state', 'error')
            status.setdefault('ok', status.get('state') == 'ready')
            return status
        except Exception:
            return {
                'ok': False,
                'state': 'extension_missing',
                'models': [],
                'message': 'Extension AI lokal belum terhubung. Ikuti petunjuk koneksi pada panel AI lokal.',
            }

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
            'state': 'ready' if matched else 'model_missing',
            'models': names,
            'message': 'Siap' if matched else f'Model {wanted} tidak ditemukan; jalankan ollama pull {wanted}'
        }
    except Exception as exc:
        _record_failure()
        return {'ok': False, 'state': 'ollama_unavailable', 'models': [], 'message': f'Ollama tidak dapat diakses (timeout/offline): {exc}'}


def ask_json(system, user, schema=None, num_predict=256):
    settings = _settings()
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    response_format = schema or 'json'
    options = {'temperature': 0, 'num_predict': num_predict}

    if settings.get('OLLAMA_PROVIDER', 'local') == 'browser':
        try:
            content = _browser_chat(messages, response_format, options)
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError('Balasan model bukan objek')
            return result
        except LLMUnavailable:
            raise
        except Exception as exc:
            raise LLMUnavailable(f'Balasan Llama lokal bukan JSON valid: {exc}') from exc

    _check_circuit_breaker()
    try:
        response = _client().chat(
            model=settings['OLLAMA_MODEL'],
            messages=messages,
            format=response_format,
            options=options,
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
    settings = _settings()
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    options = {'temperature': 0.2, 'num_predict': num_predict}

    if settings.get('OLLAMA_PROVIDER', 'local') == 'browser':
        return _browser_chat(messages, None, options).strip()

    _check_circuit_breaker()
    try:
        response = _client().chat(
            model=settings['OLLAMA_MODEL'],
            messages=messages,
            options=options,
            keep_alive='10m',
        )
        _record_success()
        return response.message.content.strip()
    except Exception as exc:
        _record_failure()
        raise LLMUnavailable(f'Ollama gagal: {exc}') from exc
