import os
import sys
from types import ModuleType
from unittest.mock import MagicMock
import pytest

try:
    import supabase
except ImportError:
    stub = ModuleType('supabase')
    stub.create_client = MagicMock()
    sys.modules['supabase'] = stub
try:
    import dotenv
except ImportError:
    stub = ModuleType('dotenv')
    stub.load_dotenv = lambda: None
    sys.modules['dotenv'] = stub


@pytest.fixture
def ollama_env(monkeypatch):
    """Ensure Ollama live tests can load configuration when unrelated Supabase settings are missing."""
    from dotenv import load_dotenv
    load_dotenv()
    dummy_keys = []
    for key in ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'SUPABASE_SERVICE_ROLE_KEY']:
        if not os.getenv(key):
            monkeypatch.setenv(key, 'http://127.0.0.1:54321' if 'URL' in key else 'placeholder-key')
            dummy_keys.append(key)
    if dummy_keys:
        from config.settings import get_settings
        get_settings.cache_clear()
    try:
        yield
    finally:
        if dummy_keys:
            from config.settings import get_settings
            get_settings.cache_clear()
