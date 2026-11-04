"""Server finto, con lo stesso protocollo di vLLM e OpenAI, per i test dei provider: registra le richieste."""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLLM:
    def __init__(self):
        self.requests = []
        self.replies = []      # coda di (status, testo o dict); vuota = "ok"
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append((self.path, body))
                status, reply = fake.replies.pop(0) if fake.replies else (200, "ok")
                if isinstance(reply, dict):
                    payload = reply
                elif self.path.endswith("/embeddings"):
                    payload = {"object": "list", "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1},
                               "data": [{"object": "embedding", "index": i, "embedding": [0.5, 0.5]}
                                        for i, _ in enumerate(body["input"])]}
                elif self.path.endswith("/messages"):
                    payload = {"id": "x", "type": "message", "role": "assistant", "model": "m", "stop_reason": "end_turn",
                               "content": [{"type": "text", "text": reply}], "usage": {"input_tokens": 1, "output_tokens": 2}}
                else:
                    payload = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": reply}}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
