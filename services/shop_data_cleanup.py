import sys
from pathlib import Path
from uuid import UUID


INDEX_ROOT = Path(__file__).resolve().parents[1] / "vector_store"


def clear_shop_local_data(shop_id):
    shop_id = str(UUID(str(shop_id)))
    root = INDEX_ROOT.resolve()
    target = (root / shop_id).resolve()
    if target.parent != root or target.name != shop_id:
        raise ValueError("Lokasi indeks toko tidak aman")
    if target.exists():
        for tipe in ("pemasukan", "pengeluaran"):
            for name in (f"{tipe}.faiss", f"{tipe}_meta.json"):
                file = target / name
                if file.resolve().parent != target:
                    raise ValueError("Lokasi berkas indeks toko tidak aman")
                file.unlink(missing_ok=True)
        target.rmdir()
    for module_name, clear_name in (
        ("ai.rag_engine", "clear_memory_cache"),
        ("ai.store_context", "clear_store_context_cache"),
        ("ai.hybrid_classifier", "clear_classification_cache"),
        ("services.forecast_service", "invalidate_forecast_cache"),
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            getattr(module, clear_name)(shop_id)
