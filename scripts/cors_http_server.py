from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


if __name__ == "__main__":
    server = ThreadingHTTPServer(
        ("localhost", 9000),
        lambda *args, **kwargs: CORSRequestHandler(
            *args, directory=str(PROJECT_ROOT), **kwargs
        ),
    )
    print(f"Serving {PROJECT_ROOT} with CORS at http://localhost:9000")
    server.serve_forever()