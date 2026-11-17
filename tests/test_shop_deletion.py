"""AppTest uses newly generated disposable stores in memory; never connects to Supabase."""
import copy
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest

from services import admin_service as service
from services import shop_data_cleanup
from services.auth_service import AccessError


class Query:
    def __init__(self, db, table):
        self.db, self.table_name, self.filters = db, table, {}
        self.operation, self.values, self.one = "select", {}, False

    def select(self, *_):
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def order(self, *_, **__):
        return self

    def range(self, *_):
        return self

    def single(self):
        self.one = True
        return self

    def update(self, values):
        self.operation, self.values = "update", values
        return self

    def delete(self):
        self.operation = "delete"
        return self

    def execute(self):
        rows = self.db.tables[self.table_name]
        selected = [r for r in rows if all(r.get(k) == v for k, v in self.filters.items())]
        if self.operation == "update":
            if self.db.fail_job_update and self.table_name == "shop_deletion_jobs":
                self.db.fail_job_update = False
                raise RuntimeError("Injected job persistence failure")
            for row in selected:
                row.update(copy.deepcopy(self.values))
        elif self.operation == "delete":
            self.db.tables[self.table_name] = [r for r in rows if r not in selected]
        result = copy.deepcopy(selected)
        if self.table_name == "profiles":
            for row in result:
                row["toko"] = next((dict(nama=s["nama"]) for s in self.db.tables["toko"] if s["id"] == row["toko_id"]), None)
        return SimpleNamespace(data=(result[0] if result else None) if self.one else result)


class MemoryAdmin:
    def __init__(self):
        self.admin_id, self.owner_id, self.active_id, self.cashier_id = [str(uuid4()) for _ in range(4)]
        self.shop_id, self.active_shop = str(uuid4()), str(uuid4())
        self.email = f"delete-test-{uuid4().hex}@example.invalid"
        self.shop_name = f"Toko Uji Hapus {uuid4().hex[:8]}"
        self.tables = {
            "toko": [dict(id=self.shop_id, nama=self.shop_name), dict(id=self.active_shop, nama="Toko Uji Aktif")],
            "profiles": [
                self.profile(self.admin_id, None, "superadmin", "aktif", "admin-test@example.invalid"),
                self.profile(self.owner_id, self.shop_id, "pemilik_toko", "nonaktif", self.email),
                self.profile(self.active_id, self.active_shop, "pemilik_toko", "aktif", "active-test@example.invalid"),
                self.profile(self.cashier_id, self.shop_id, "kasir", "aktif", "cashier-test@example.invalid"),
            ],
            "shop_deletion_jobs": [],
            "audit_log": [dict(toko_id=self.shop_id, action="change_owner_status")],
        }
        for table in ("transaksi", "kategori", "rag_index_version", "import_transaction_identities", "cashier_provisioning"):
            self.tables[table] = [dict(id=str(uuid4()), toko_id=s) for s in (self.shop_id, self.active_shop)]
        self.tables["transaction_deletion_requests"] = [dict(store_id=self.shop_id)]
        self.auth_users = {r["id"] for r in self.tables["profiles"]}
        self.deleted_auth, self.fail_auth = [], False
        self.fail_rpc, self.fail_job_update = False, False
        self.auth = SimpleNamespace(admin=SimpleNamespace(delete_user=self.delete_auth, update_user_by_id=lambda *_: None))

    @staticmethod
    def profile(uid, shop, role, status, email):
        return dict(id=uid, toko_id=shop, role=role, status=status, email=email, nama=email.split("@")[0], created_at="2026-11-17T00:00:00Z")

    def table(self, table):
        return Query(self, table)

    def delete_auth(self, uid):
        if self.fail_auth:
            raise RuntimeError("Injected auth failure")
        if uid not in self.auth_users:
            error = RuntimeError("User not found")
            error.code = "user_not_found"
            raise error
        self.auth_users.remove(uid)
        self.deleted_auth.append(uid)

    def rpc(self, name, values=None):
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=self.execute_rpc(name, values or {})))

    def execute_rpc(self, name, values):
        if name == "admin_shop_summary":
            return [dict(nama_toko=s["nama"], jumlah_pengguna=1, jumlah_transaksi=1, jumlah_kategori=1) for s in self.tables["toko"]]
        if name == "mutate_finance_audited":
            owner = next(r for r in self.tables["profiles"] if r["id"] == values["p_ids"][0])
            owner.update(values["p_values"])
            return [copy.deepcopy(owner)]
        assert name == "delete_inactive_shop_atomic"
        if self.fail_rpc:
            raise RuntimeError("Injected RPC failure")
        job = next((j for j in self.tables["shop_deletion_jobs"] if j["owner_id"] == values["p_owner"]), None)
        if job:
            assert job["shop_id"] == values["p_shop"] and job["email"] == values["p_email"]
            return copy.deepcopy(job)
        owner = next(r for r in self.tables["profiles"] if r["id"] == values["p_owner"])
        if owner["role"] != "pemilik_toko" or owner["status"] != "nonaktif" or owner["toko_id"] != values["p_shop"] or owner["email"] != values["p_email"]:
            raise ValueError("Hanya pemilik Nonaktif dengan identitas toko yang sesuai dapat dihapus")
        shop_id = owner["toko_id"]
        if any(r["role"] == "pemilik_toko" and r["id"] != owner["id"] and r["toko_id"] == shop_id for r in self.tables["profiles"]):
            raise ValueError("Toko memiliki pemilik lain")
        shop = next(r for r in self.tables["toko"] if r["id"] == shop_id)
        job = dict(owner_id=owner["id"], shop_id=shop_id, email=owner["email"], owner_name=owner["nama"], shop_name=shop["nama"],
                   auth_user_ids=[r["id"] for r in self.tables["profiles"] if r["toko_id"] == shop_id])
        self.tables["shop_deletion_jobs"].append(copy.deepcopy(job))
        for table in self.tables:
            key = "id" if table == "toko" else "store_id" if table == "transaction_deletion_requests" else "toko_id"
            if table not in ("shop_deletion_jobs", "audit_log"):
                self.tables[table] = [r for r in self.tables[table] if r.get(key) != shop_id]
        return job


@pytest.fixture
def db(monkeypatch, tmp_path):
    db = MemoryAdmin()
    monkeypatch.setattr(service, "verify_actor", lambda *_: dict(id=db.admin_id, role="superadmin", status="aktif"))
    monkeypatch.setattr(service, "privileged_client", lambda: db)
    monkeypatch.setattr(shop_data_cleanup, "INDEX_ROOT", tmp_path / "vector_store")
    from pages import superadmin
    monkeypatch.setattr(superadmin, "list_deletion_requests", lambda *_, **__: [])
    return db


def app():
    from pages.superadmin import render
    render("test-only-token")


def run_app():
    at = AppTest.from_function(app).run()
    assert not at.exception
    return at


def delete(db):
    return service.delete_inactive_shop("test", db.owner_id, db.shop_id, db.email)


def test_five_required_app_states(db):
    before = copy.deepcopy(db.tables)
    at = run_app()
    assert at.button(key=f"owner_delete_{db.owner_id}").label == "Hapus"
    assert at.button(key=f"save_{db.owner_id}").label == "Simpan"
    assert not any(b.key == f"owner_delete_{db.active_id}" for b in at.button)
    at.button(key=f"owner_delete_{db.owner_id}").click().run()
    texts = "\n".join(m.value for m in at.markdown)
    assert db.shop_name in texts and db.email in texts
    assert "tidak bisa dibatalkan" in at.error[0].value
    at.button(key="cancel_shop_delete").click().run()
    assert db.tables == before and not db.deleted_auth
    at.button(key=f"owner_delete_{db.owner_id}").click().run()
    at.button(key="owner_delete_confirm").click().run()
    assert not at.exception
    assert any("berhasil dihapus" in m.value for m in at.success)
    assert not any(b.key == f"save_{db.owner_id}" for b in at.button)
    assert not any(r["id"] == db.owner_id for r in db.tables["profiles"])
    assert not any(r["id"] == db.cashier_id for r in db.tables["profiles"])
    for table, rows in db.tables.items():
        if table != "audit_log":
            assert not any(r.get("toko_id", r.get("store_id", r.get("id"))) == db.shop_id for r in rows)
    assert db.owner_id not in db.auth_users and db.cashier_id not in db.auth_users
    assert db.active_id in db.auth_users
    for table in ("toko", "profiles", "transaksi", "kategori", "rag_index_version", "import_transaction_identities", "cashier_provisioning"):
        key = "id" if table == "toko" else "toko_id"
        assert [r for r in db.tables[table] if r.get(key) == db.active_shop] == [r for r in before[table] if r.get(key) == db.active_shop]
    for frame in at.dataframe:
        assert db.email not in frame.value.to_string() and db.shop_name not in frame.value.to_string()
    at.selectbox(key=f"state_{db.active_id}").set_value("nonaktif").run()
    assert not any(b.key == f"owner_delete_{db.active_id}" for b in at.button)
    at.button(key=f"save_{db.active_id}").click().run()
    assert not at.exception
    assert next(r for r in db.tables["profiles"] if r["id"] == db.active_id)["status"] == "nonaktif"
    assert at.button(key=f"owner_delete_{db.active_id}").label == "Hapus"
    print(json.dumps({"fixture": "NEW in-memory shops only; no real database", "shop_name": db.shop_name, "owner_email": db.email,
                      "a": "PASS Nonaktif: Hapus beside Simpan", "b": "PASS Aktif: no Hapus (also before saving changed dropdown)",
                      "c": "PASS dialog name/email/permanent warning; Batal left all tables/auth unchanged", "d": "PASS confirm removed shop/data/profiles/owner+cashier auth; three displays refreshed; other test shop untouched; audit retained",
                      "e": "PASS Simpan changed other new test owner aktif -> nonaktif after deletion", "exceptions": len(at.exception)}, ensure_ascii=False))


def test_active_stale_confirmation_rejected(db):
    at = run_app()
    at.button(key=f"owner_delete_{db.owner_id}").click().run()
    owner = next(r for r in db.tables["profiles"] if r["id"] == db.owner_id)
    owner["status"] = "aktif"
    before = copy.deepcopy(db.tables)
    at.button(key="owner_delete_confirm").click().run()
    assert any("Gagal menghapus" in e.value for e in at.error)
    assert db.tables == before and not db.deleted_auth


def test_auth_failure_is_durable_and_retryable(db):
    db.fail_auth = True
    at = run_app()
    at.button(key=f"owner_delete_{db.owner_id}").click().run()
    at.button(key="owner_delete_confirm").click().run()
    assert not at.success and any("belum selesai" in w.value for w in at.warning)
    assert db.tables["shop_deletion_jobs"]
    assert at.button(key=f"save_{db.owner_id}").disabled
    db.fail_auth = False
    # A new AppTest session models restarting the application before retrying.
    at = run_app()
    at.button(key=f"owner_delete_{db.owner_id}").click().run()
    at.button(key="owner_delete_confirm").click().run()
    assert any("berhasil dihapus" in e.value for e in at.success)
    assert not db.tables["shop_deletion_jobs"] and db.owner_id not in db.auth_users


def test_retry_after_auth_succeeded_but_job_update_failed(db):
    db.fail_job_update = True
    with pytest.raises(service.ShopAuthCleanupError):
        delete(db)
    assert len(db.deleted_auth) == 1
    delete(db)
    assert len(db.deleted_auth) == 2 and not db.tables["shop_deletion_jobs"]


def test_legacy_auth_not_found_is_retryable(db):
    db.fail_job_update = True
    with pytest.raises(service.ShopAuthCleanupError):
        delete(db)
    original = db.auth.admin.delete_user
    def legacy_delete(uid):
        if uid not in db.auth_users:
            error = RuntimeError("User not found")
            error.status = 404
            raise error
        original(uid)
    db.auth.admin.delete_user = legacy_delete
    delete(db)
    assert not db.tables["shop_deletion_jobs"]


def test_other_auth_404_is_not_silently_accepted(db):
    def missing_endpoint(_):
        error = RuntimeError("Endpoint not found")
        error.status = 404
        raise error
    db.auth.admin.delete_user = missing_endpoint
    with pytest.raises(service.ShopAuthCleanupError):
        delete(db)
    assert db.tables["shop_deletion_jobs"] and db.owner_id in db.auth_users


def test_rpc_failure_does_not_call_auth_or_remove_data(db):
    db.fail_rpc = True
    before = copy.deepcopy(db.tables)
    with pytest.raises(RuntimeError, match="RPC failure"):
        delete(db)
    assert db.tables == before and not db.deleted_auth


@pytest.mark.parametrize("field,value", [("status", "aktif"), ("status", "ditolak"), ("role", "kasir"), ("toko_id", "other-shop"), ("email", "changed@example.invalid")])
def test_invalid_target_fails_closed(db, field, value):
    next(r for r in db.tables["profiles"] if r["id"] == db.owner_id)[field] = value
    before = copy.deepcopy(db.tables)
    with pytest.raises(ValueError):
        delete(db)
    assert db.tables == before and not db.deleted_auth


def test_shared_owner_blocks_delete(db):
    db.tables["profiles"].append(db.profile(str(uuid4()), db.shop_id, "pemilik_toko", "aktif", "shared-test@example.invalid"))
    before = copy.deepcopy(db.tables)
    with pytest.raises(ValueError, match="pemilik lain"):
        delete(db)
    assert db.tables == before


def test_superadmin_verification_and_self_guard(db, monkeypatch):
    def deny(*_):
        raise AccessError("Role tidak diizinkan")
    monkeypatch.setattr(service, "verify_actor", deny)
    with pytest.raises(AccessError):
        delete(db)
    monkeypatch.setattr(service, "verify_actor", lambda *_: {"id": db.owner_id})
    with pytest.raises(AccessError, match="sendiri"):
        delete(db)
    assert not db.deleted_auth


def test_local_index_cleanup_is_scoped_and_retryable(db):
    root = shop_data_cleanup.INDEX_ROOT
    target, other = root / db.shop_id, root / db.active_shop
    for folder in (target, other):
        folder.mkdir(parents=True)
        for name in ("pemasukan.faiss", "pemasukan_meta.json", "pengeluaran.faiss", "pengeluaran_meta.json"):
            (folder / name).write_text("test-only-index", encoding="utf-8")
    delete(db)
    assert not target.exists() and len(list(other.iterdir())) == 4
    with pytest.raises(ValueError):
        shop_data_cleanup.clear_shop_local_data("../outside")
