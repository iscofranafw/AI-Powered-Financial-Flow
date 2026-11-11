"""In-memory, session-bound relay for browser-local Ollama inference.

This module never loads or contacts an Ollama model on the application server.
The browser companion polls this loopback-only HTTP broker and returns results.
Prompt/response bodies are kept in memory only and are never logged.
"""
from __future__ import annotations

import atexit
import json
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, Optional
from urllib.parse import urlsplit

MAX_PROMPT_BYTES = 512 * 1024
MAX_RESULT_BYTES = 512 * 1024
MAX_RESULT_CONTENT_BYTES = 480 * 1024
MAX_STATUS_BYTES = 16 * 1024
MAX_STATUS_TEXT_BYTES = 1024
MAX_MESSAGES = 24
POLL_TIMEOUT_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 300
SESSION_IDLE_SECONDS = 600
STATUS_STALE_SECONDS = 90
JOB_TTL_SECONDS = 360
MAX_ACTIVE_JOBS_PER_SESSION = 1
MAX_ACTIVE_JOBS_GLOBAL = 8
STAGING_ALLOWED_ORIGIN = "https://shazraz.my.id"
ALLOWED_STATES = {
    "checking",
    "ready",
    "extension_missing",
    "not_paired",
    "ollama_unavailable",
    "model_missing",
    "origin_not_allowed",
    "busy",
    "timeout",
    "error",
}


class LocalAIBridgeError(RuntimeError):
    """Expected relay/companion failures safe to show in the UI."""


def _limit_utf8(value: str, maximum: int) -> str:
    return value.encode("utf-8")[:maximum].decode("utf-8", errors="ignore")


@dataclass
class _Job:
    job_id: str
    token: str
    payload: Dict[str, Any]
    created_at: float = field(default_factory=time.monotonic)
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[Dict[str, Any]] = None
    claimed: bool = False


@dataclass
class _Session:
    token: str
    status: Dict[str, Any] = field(default_factory=lambda: {
        "state": "checking",
        "message": "Memeriksa extension Financial Flow…",
        "models": [],
    })
    status_updated_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    pending: Deque[str] = field(default_factory=deque)
    active_job_id: Optional[str] = None


class BridgeBroker:
    """Thread-safe broker; each opaque token can only poll/complete its own jobs."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._sessions: Dict[str, _Session] = {}
        self._jobs: Dict[str, _Job] = {}

    def _cleanup_locked(self) -> None:
        now = time.monotonic()
        expired_jobs = [
            job_id for job_id, job in self._jobs.items()
            if now - job.created_at > JOB_TTL_SECONDS
        ]
        for job_id in expired_jobs:
            job = self._jobs.pop(job_id)
            job.response = {"ok": False, "state": "timeout", "message": "Permintaan AI lokal kedaluwarsa."}
            job.event.set()
            session = self._sessions.get(job.token)
            if session:
                session.pending = deque(value for value in session.pending if value != job_id)
                if session.active_job_id == job_id:
                    session.active_job_id = None
        expired_sessions = [
            token for token, session in self._sessions.items()
            if now - session.last_seen > SESSION_IDLE_SECONDS and not session.active_job_id
        ]
        for token in expired_sessions:
            self._sessions.pop(token, None)

    def register_session(self, token: str) -> None:
        if not isinstance(token, str) or len(token) < 32 or len(token) > 128:
            raise LocalAIBridgeError("Token sesi bridge tidak valid.")
        with self._condition:
            self._cleanup_locked()
            session = self._sessions.get(token)
            if session is None:
                self._sessions[token] = _Session(token=token)
            else:
                session.last_seen = time.monotonic()

    def _session_locked(self, token: str) -> _Session:
        self._cleanup_locked()
        session = self._sessions.get(token)
        if session is None:
            raise LocalAIBridgeError("Sesi bridge kedaluwarsa; muat ulang aplikasi.")
        session.last_seen = time.monotonic()
        return session

    def update_status(self, token: str, value: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise LocalAIBridgeError("Status extension tidak valid.")
        state = value.get("state")
        if state not in ALLOWED_STATES:
            state = "error"
        message = value.get("message", "")
        if not isinstance(message, str):
            message = ""
        models = value.get("models", [])
        if not isinstance(models, list):
            models = []
        clean_models = [_limit_utf8(item, 120) for item in models if isinstance(item, str)][:64]
        status = {
            "state": state,
            "message": _limit_utf8(message, MAX_STATUS_TEXT_BYTES),
            "models": clean_models,
        }
        with self._condition:
            session = self._session_locked(token)
            session.status = status
            session.status_updated_at = time.monotonic()
            return dict(status)

    def get_status(self, token: Optional[str]) -> Dict[str, Any]:
        if not token:
            return {
                "ok": False,
                "state": "extension_missing",
                "models": [],
                "message": "Bridge browser lokal belum terhubung.",
            }
        with self._condition:
            try:
                session = self._session_locked(token)
            except LocalAIBridgeError:
                return {
                    "ok": False,
                    "state": "extension_missing",
                    "models": [],
                    "message": "Bridge browser belum aktif. Muat ulang Financial Flow dan hubungkan extension.",
                }
            status = dict(session.status)
            age = time.monotonic() - session.status_updated_at
            if age > STATUS_STALE_SECONDS:
                status = {
                    "state": "extension_missing",
                    "message": "Extension tidak merespons. Buka Financial Flow dan hubungkan kembali tab ini.",
                    "models": [],
                }
            status["ok"] = status.get("state") == "ready"
            return status

    def poll(self, token: str, timeout: float = POLL_TIMEOUT_SECONDS) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + max(0.0, min(float(timeout), POLL_TIMEOUT_SECONDS))
        with self._condition:
            while True:
                session = self._session_locked(token)
                while session.pending:
                    job_id = session.pending.popleft()
                    job = self._jobs.get(job_id)
                    if job is None:
                        continue
                    if time.monotonic() - job.created_at > JOB_TTL_SECONDS:
                        self._jobs.pop(job_id, None)
                        job.response = {"ok": False, "state": "timeout", "message": "Permintaan AI lokal kedaluwarsa."}
                        job.event.set()
                        continue
                    job.claimed = True
                    session.active_job_id = job_id
                    session.last_seen = time.monotonic()
                    return {"id": job.job_id, "payload": job.payload}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def complete(self, token: str, job_id: str, value: Dict[str, Any]) -> None:
        if not isinstance(job_id, str) or len(job_id) > 128:
            raise LocalAIBridgeError("ID permintaan tidak valid.")
        if not isinstance(value, dict):
            raise LocalAIBridgeError("Hasil Ollama lokal tidak valid.")
        ok = value.get("ok") is True
        state = value.get("state", "ready" if ok else "error")
        if state not in ALLOWED_STATES:
            state = "error"
        content = value.get("content", "")
        message = value.get("message", "")
        if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_RESULT_CONTENT_BYTES:
            raise LocalAIBridgeError("Hasil model melebihi batas ukuran.")
        if not isinstance(message, str):
            message = ""
        response = {"ok": ok, "state": state, "content": content, "message": _limit_utf8(message, MAX_STATUS_TEXT_BYTES)}
        with self._condition:
            session = self._session_locked(token)
            job = self._jobs.get(job_id)
            if job is None or job.token != token:
                raise LocalAIBridgeError("Permintaan sudah selesai atau tidak dimiliki sesi ini.")
            if job.response is not None:
                raise LocalAIBridgeError("Hasil permintaan sudah diterima.")
            job.response = response
            session.status = {
                "state": state,
                "message": response["message"] or ("Siap" if ok else "Pemeriksaan model lokal gagal."),
                "models": session.status.get("models", []),
            }
            session.status_updated_at = time.monotonic()
            job.event.set()
            self._condition.notify_all()

    def submit_and_wait(
        self,
        token: str,
        payload: Dict[str, Any],
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_PROMPT_BYTES:
            raise LocalAIBridgeError("Permintaan AI terlalu besar; ringkas data yang dikirim.")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
            raise LocalAIBridgeError("Format pesan AI tidak valid.")
        allowed_roles = {"system", "user", "assistant"}
        for item in messages:
            if not isinstance(item, dict) or item.get("role") not in allowed_roles or not isinstance(item.get("content"), str):
                raise LocalAIBridgeError("Format pesan AI tidak valid.")
        with self._condition:
            session = self._session_locked(token)
            status = self.get_status(token)
            if not status.get("ok"):
                raise LocalAIBridgeError(status.get("message") or "Ollama lokal belum siap.")
            if session.active_job_id or session.pending:
                raise LocalAIBridgeError("Model lokal sedang memproses permintaan lain. Coba lagi sebentar.")
            active_count = sum(1 for item in self._sessions.values() if item.active_job_id or item.pending)
            if active_count >= MAX_ACTIVE_JOBS_GLOBAL:
                raise LocalAIBridgeError("Bridge sedang sibuk. Coba lagi sebentar.")
            job_id = secrets.token_urlsafe(18)
            job = _Job(job_id=job_id, token=token, payload=payload)
            self._jobs[job_id] = job
            session.pending.append(job_id)
            session.last_seen = time.monotonic()
            self._condition.notify_all()

        finished = job.event.wait(timeout=max(1.0, float(timeout)))
        with self._condition:
            session = self._sessions.get(token)
            if session and session.active_job_id == job_id:
                session.active_job_id = None
            if session:
                session.pending = deque(value for value in session.pending if value != job_id)
                session.last_seen = time.monotonic()
            self._jobs.pop(job_id, None)
            if not finished:
                raise LocalAIBridgeError("Permintaan ke Ollama lokal melewati batas waktu; periksa perangkat pengguna dan coba lagi.")
            response = job.response or {}
            if not response.get("ok"):
                raise LocalAIBridgeError(response.get("message") or "Ollama lokal gagal memproses permintaan.")
            return response.get("content", "")

    def close(self) -> None:
        with self._condition:
            for job in self._jobs.values():
                job.response = {"ok": False, "state": "error", "message": "Bridge dihentikan."}
                job.event.set()
            self._jobs.clear()
            self._sessions.clear()
            self._condition.notify_all()


BROKER = BridgeBroker()
_SERVER_LOCK = threading.Lock()
_SERVER: Optional[ThreadingHTTPServer] = None
_SERVER_THREAD: Optional[threading.Thread] = None
_SERVER_ERROR: Optional[str] = None


def _make_handler(allowed_origin: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FinancialFlowLocalAIBridge/1"
        sys_version = ""

        def log_message(self, _format: str, *_args: Any) -> None:
            # Never log headers, tokens, prompts, model output, or request bodies.
            return

        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            if origin:
                return secrets.compare_digest(origin, allowed_origin)
            return self.headers.get("Sec-Fetch-Site", "") == "same-origin"

        def _token(self) -> Optional[str]:
            authorization = self.headers.get("Authorization", "")
            if not authorization.startswith("Bearer "):
                return None
            token = authorization[7:].strip()
            return token if 32 <= len(token) <= 128 else None

        def _read_json(self, maximum: int) -> Dict[str, Any]:
            if not self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() == "application/json":
                raise ValueError("Content-Type harus application/json.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Content-Length tidak valid.") from exc
            if length <= 0 or length > maximum:
                raise ValueError("Ukuran permintaan tidak valid atau terlalu besar.")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Isi permintaan tidak lengkap.")
            value = json.loads(body.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Body harus berupa objek JSON.")
            return value

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if not self._origin_ok():
                self._send_json(403, {"ok": False, "message": "Origin ditolak."})
                return
            if urlsplit(self.path).path != "/v1/poll":
                self._send_json(404, {"ok": False, "message": "Endpoint tidak ditemukan."})
                return
            token = self._token()
            if not token:
                self._send_json(401, {"ok": False, "message": "Sesi bridge tidak valid."})
                return
            try:
                job = BROKER.poll(token, POLL_TIMEOUT_SECONDS)
                self._send_json(200, {"ok": True, "request": job})
            except LocalAIBridgeError as exc:
                self._send_json(401, {"ok": False, "message": str(exc)})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if not self._origin_ok():
                self._send_json(403, {"ok": False, "message": "Origin ditolak."})
                return
            path = urlsplit(self.path).path
            if path not in ("/v1/status", "/v1/complete"):
                self._send_json(404, {"ok": False, "message": "Endpoint tidak ditemukan."})
                return
            token = self._token()
            if not token:
                self._send_json(401, {"ok": False, "message": "Sesi bridge tidak valid."})
                return
            try:
                maximum = MAX_STATUS_BYTES if path == "/v1/status" else MAX_RESULT_BYTES
                value = self._read_json(maximum)
                if path == "/v1/status":
                    status = BROKER.update_status(token, value)
                    self._send_json(200, {"ok": True, "status": status})
                else:
                    job_id = value.get("id")
                    result = value.get("result")
                    BROKER.complete(token, job_id, result)
                    self._send_json(200, {"ok": True})
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"ok": False, "message": str(exc)[:200]})
            except LocalAIBridgeError as exc:
                self._send_json(409, {"ok": False, "message": str(exc)[:200]})
            except Exception:
                self._send_json(500, {"ok": False, "message": "Bridge interno tidak dapat menyelesaikan permintaan."})

        def do_OPTIONS(self) -> None:  # noqa: N802 - disallow cross-origin preflight
            self._send_json(405, {"ok": False, "message": "Metode tidak diizinkan."})

    return Handler


def start_bridge_server(port: int = 8511, allowed_origin: str = "https://shazraz.my.id") -> None:
    """Start the broker once, bound strictly to IPv4 loopback."""
    global _SERVER, _SERVER_THREAD, _SERVER_ERROR
    with _SERVER_LOCK:
        if _SERVER is not None and _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
            return
        if _SERVER_ERROR:
            raise LocalAIBridgeError(_SERVER_ERROR)
        if not (1 <= int(port) <= 65535):
            raise LocalAIBridgeError("Port bridge lokal tidak valid.")
        if allowed_origin != STAGING_ALLOWED_ORIGIN:
            raise LocalAIBridgeError("Origin relay hanya boleh https://shazraz.my.id untuk staging ini.")
        try:
            server = ThreadingHTTPServer(("127.0.0.1", int(port)), _make_handler(allowed_origin))
            server.daemon_threads = True
            server.block_on_close = False
            thread = threading.Thread(target=server.serve_forever, name="financial-flow-local-ai-bridge", daemon=True)
            thread.start()
        except OSError as exc:
            _SERVER_ERROR = "Relay AI lokal gagal dijalankan pada loopback VPS; periksa port bridge staging."
            raise LocalAIBridgeError(_SERVER_ERROR) from exc
        _SERVER = server
        _SERVER_THREAD = thread
        atexit.register(_shutdown_server)


def _shutdown_server() -> None:
    global _SERVER, _SERVER_THREAD
    with _SERVER_LOCK:
        server = _SERVER
        _SERVER = None
        _SERVER_THREAD = None
    if server is not None:
        server.shutdown()
        server.server_close()
    BROKER.close()


def get_session_token() -> str:
    """Return/register an opaque per-Streamlit-session capability token."""
    try:
        import streamlit as st
        token = st.session_state.get("_local_ai_bridge_token")
        if not token:
            token = secrets.token_urlsafe(36)
            st.session_state["_local_ai_bridge_token"] = token
    except Exception as exc:
        raise LocalAIBridgeError("Tidak dapat membuat token sesi browser lokal.") from exc
    BROKER.register_session(token)
    return token


def current_session_token() -> Optional[str]:
    try:
        import streamlit as st
        token = st.session_state.get("_local_ai_bridge_token")
    except Exception:
        return None
    return token if isinstance(token, str) else None


def request_chat(
    messages: list[Dict[str, str]],
    model: str,
    response_format: Any = None,
    options: Optional[Dict[str, Any]] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> str:
    token = current_session_token()
    if not token:
        raise LocalAIBridgeError("Sesi browser tidak memiliki bridge Ollama lokal. Muat ulang Financial Flow.")
    BROKER.register_session(token)
    if model != "llama3:8b":
        raise LocalAIBridgeError("Bridge staging hanya mengizinkan model llama3:8b.")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "options": options or {},
    }
    if response_format is not None:
        payload["format"] = response_format
    try:
        return BROKER.submit_and_wait(token, payload, timeout)
    except LocalAIBridgeError:
        raise


def diagnosis(token: Optional[str] = None) -> Dict[str, Any]:
    return BROKER.get_status(token or current_session_token())


def render_status_widget(settings: Dict[str, Any]) -> Optional[str]:
    """Render a small browser status component and return a UI-safe startup error."""
    try:
        port = int(settings.get("LOCAL_AI_BRIDGE_PORT", 8511))
        allowed_origin = settings.get("LOCAL_AI_ALLOWED_ORIGIN", "https://shazraz.my.id")
        start_bridge_server(port, allowed_origin)
        token = get_session_token()
        import streamlit.components.v1 as components
        component_path = Path(__file__).resolve().parent / "local_ai_frontend"
        component = components.declare_component("financial_flow_local_ai_bridge", path=str(component_path))
        component(
            session_token=token,
            api_base=settings.get("LOCAL_AI_BRIDGE_BASE_PATH", "/financial-flow/_local-ai/v1"),
            model=settings.get("OLLAMA_MODEL", "llama3:8b"),
            extension_origin="chrome-extension://jomfepbhiphidnjdmjodcadeehjnjhcl",
            key="financial_flow_local_ai_bridge_widget",
        )
        return None
    except Exception as exc:
        return str(exc)[:300]
