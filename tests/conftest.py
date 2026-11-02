import sys
from types import ModuleType
from unittest.mock import MagicMock
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
