import json
import secrets
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from ai.local_ai_bridge import (
    BROKER,
    STAGING_ALLOWED_ORIGIN,
    BridgeBroker,
    LocalAIBridgeError,
    _make_handler,
)


class BridgeBrokerTests(unittest.TestCase):
    def ready_session(self, broker):
        token = secrets.token_urlsafe(36)
        broker.register_session(token)
        broker.update_status(token, {
            "state": "ready",
            "message": "Terhubung; model llama3:8b tersedia.",
            "models": ["llama3:8b"],
        })
        return token

    def test_browser_round_trip_returns_local_model_content(self):
        broker = BridgeBroker()
        token = self.ready_session(broker)
        payload = {
            "model": "llama3:8b",
            "messages": [{"role": "user", "content": "Ringkas arus kas."}],
            "options": {"num_predict": 64},
        }
        values = []
        errors = []

        def infer():
            try:
                values.append(broker.submit_and_wait(token, payload, timeout=3))
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=infer)
        worker.start()
        job = None
        for _ in range(40):
            job = broker.poll(token, timeout=0.05)
            if job:
                break
        self.assertIsNotNone(job)
        self.assertEqual(job["payload"], payload)
        broker.complete(token, job["id"], {
            "ok": True,
            "state": "ready",
            "content": "Arus kas positif.",
            "message": "Jawaban lokal tersedia.",
        })
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(values, ["Arus kas positif."])

    def test_request_fails_fast_when_local_model_is_not_ready(self):
        broker = BridgeBroker()
        token = secrets.token_urlsafe(36)
        broker.register_session(token)
        broker.update_status(token, {
            "state": "model_missing",
            "message": "Jalankan ollama pull llama3:8b.",
            "models": [],
        })
        with self.assertRaisesRegex(LocalAIBridgeError, "ollama pull"):
            broker.submit_and_wait(token, {
                "messages": [{"role": "user", "content": "uji"}],
            })

    def test_other_session_cannot_complete_job(self):
        broker = BridgeBroker()
        owner_token = self.ready_session(broker)
        other_token = self.ready_session(broker)
        payload = {"messages": [{"role": "user", "content": "uji"}]}
        values = []
        errors = []

        def infer():
            try:
                values.append(broker.submit_and_wait(owner_token, payload, timeout=3))
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=infer)
        worker.start()
        job = None
        for _ in range(40):
            job = broker.poll(owner_token, timeout=0.05)
            if job:
                break
        self.assertIsNotNone(job)
        with self.assertRaisesRegex(LocalAIBridgeError, "tidak dimiliki sesi"):
            broker.complete(other_token, job["id"], {"ok": True, "state": "ready", "content": "bukan sesi ini"})
        broker.complete(owner_token, job["id"], {"ok": True, "state": "ready", "content": "selesai"})
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(values, ["selesai"])

    def test_oversized_prompt_is_rejected_before_queueing(self):
        broker = BridgeBroker()
        token = self.ready_session(broker)
        payload = {"messages": [{"role": "user", "content": "x" * (512 * 1024)}]}
        with self.assertRaisesRegex(LocalAIBridgeError, "terlalu besar"):
            broker.submit_and_wait(token, payload)

    def test_http_status_endpoint_requires_exact_origin_and_session_token(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(STAGING_ALLOWED_ORIGIN))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        token = secrets.token_urlsafe(36)
        BROKER.register_session(token)
        url = f"http://127.0.0.1:{server.server_address[1]}/v1/status"
        body = json.dumps({"state": "ready", "message": "Siap", "models": ["llama3:8b"]}).encode()
        headers = {
            "Origin": STAGING_ALLOWED_ORIGIN,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=3) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(json.loads(response.read())["ok"])

            wrong_origin_headers = dict(headers, Origin="https://example.com")
            wrong_origin = urllib.request.Request(url, data=body, headers=wrong_origin_headers, method="POST")
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(wrong_origin, timeout=3)
            self.assertEqual(error.exception.code, 403)

            unknown_headers = dict(headers, Authorization="Bearer " + secrets.token_urlsafe(36))
            unknown = urllib.request.Request(url, data=body, headers=unknown_headers, method="POST")
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(unknown, timeout=3)
            self.assertEqual(error.exception.code, 409)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
