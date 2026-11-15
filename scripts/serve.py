"""
Nirnay /v1/systemone API Server.
Compatible with TypeSafe AI, llama.cpp /v1/systemone, and S1MB benchmark harness.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, List

from src.infer import NirnayEngine
from src.schema import DecisionKind

engine: NirnayEngine = None

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class SystemOneRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to keep terminal clean during evaluation
        pass

    def do_GET(self):
        if self.path in ("/", "/health", "/v1/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "model": "nirnay-0.5b"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if not self.path.startswith("/v1/systemone"):
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_response(400)
            self.end_headers()
            return

        body_bytes = self.rfile.read(content_length)
        try:
            req_data = json.loads(body_bytes.decode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Invalid JSON: {str(e)}"}).encode("utf-8"))
            return

        state = req_data.get("state", "")
        # Format state if it's a dict
        if isinstance(state, dict):
            state_str = json.dumps(state, ensure_ascii=False)
        else:
            state_str = str(state)

        questions = req_data.get("questions", {})
        answers = {}

        # S1MB sends questions as a dict {question_id: {type, instructions, criteria}}
        if isinstance(questions, dict):
            items = questions.items()
        elif isinstance(questions, list):
            items = [(q.get("id", f"q_{i}"), q) for i, q in enumerate(questions)]
        else:
            items = []

        for q_id, q_data in items:
            task = q_data.get("type", "choice")
            instructions = q_data.get("instructions", "")
            if isinstance(instructions, dict):
                # May have system + instruction
                instructions = instructions.get("instruction", str(instructions))
            else:
                instructions = str(instructions)

            criteria = q_data.get("criteria", None)

            if task == "noul":
                res = engine.decide(
                    state=state_str,
                    question=instructions,
                    kind="noul",
                )
                prob_true = float(res.probabilities.get("true", 0.5))
                answers[q_id] = {
                    "type": "noul",
                    "noul": prob_true,
                }
            elif task == "score":
                # criteria is either a list of descriptions or a count
                if isinstance(criteria, list):
                    options = [str(c) for c in criteria]
                elif isinstance(criteria, dict):
                    options = [str(criteria[k]) for k in sorted(criteria.keys())]
                else:
                    options = None

                res = engine.decide(
                    state=state_str,
                    question=instructions,
                    options=options,
                    kind="score",
                )
                # Output score probabilities indexed as "0", "1", ...
                num_opts = len(options) if options else 6
                prob_dict = {}
                probs_list = list(res.probabilities.values())
                for i in range(num_opts):
                    prob_dict[str(i)] = float(probs_list[i]) if i < len(probs_list) else 0.0

                # Normalize sum
                total = sum(prob_dict.values())
                if total > 0:
                    prob_dict = {k: v / total for k, v in prob_dict.items()}

                answers[q_id] = {
                    "type": "score",
                    "probabilities": prob_dict,
                }
            else:
                # Choice
                if isinstance(criteria, dict):
                    opt_keys = list(criteria.keys())
                    options = [str(criteria[k]) for k in opt_keys]
                elif isinstance(criteria, list):
                    opt_keys = [f"option_{i}" for i in range(len(criteria))]
                    options = [str(c) for c in criteria]
                else:
                    opt_keys = ["option_0", "option_1"]
                    options = ["Option A", "Option B"]

                res = engine.decide(
                    state=state_str,
                    question=instructions,
                    options=options,
                    kind="choice",
                )
                probs_list = list(res.probabilities.values())
                prob_dict = {}
                for i, k in enumerate(opt_keys):
                    prob_dict[k] = float(probs_list[i]) if i < len(probs_list) else 0.0

                total = sum(prob_dict.values())
                if total > 0:
                    prob_dict = {k: v / total for k, v in prob_dict.items()}

                answers[q_id] = {
                    "type": "choice",
                    "probabilities": prob_dict,
                }

        response = {
            "model": "nirnay-0.5b",
            "usage": {
                "input_tokens": 128,
                "output_tokens": len(answers),
            },
            "answers": answers,
        }

        resp_bytes = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

def serve(
    model_path: str = "checkpoints/nirnay-0.5b/checkpoint-latest",
    host: str = "127.0.0.1",
    port: int = 8080,
):
    global engine
    print(f"Loading NirnayEngine from {model_path}...")
    engine = NirnayEngine.from_pretrained(model_path)
    print(f"Nirnay System One Server listening on http://{host}:{port}/v1/systemone")

    server = ThreadedHTTPServer((host, port), SystemOneRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve Nirnay System One API")
    parser.add_argument("--model_path", type=str, default="checkpoints/nirnay-0.5b/checkpoint-latest")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    serve(model_path=args.model_path, host=args.host, port=args.port)
