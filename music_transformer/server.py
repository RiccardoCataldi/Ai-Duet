"""
Serve AMTGenerator over HTTP so the model can run on another machine (e.g. a Jetson).

    python -m music_transformer.server            # listens on 127.0.0.1:8765
    ssh -N -L 8765:localhost:8765 jetson          # on the laptop, then AMT_SERVER=http://localhost:8765

POST /respond with the prompt MIDI file as the body; the reply is the generated MIDI file,
or 204 when there is nothing to respond to.
"""

import argparse
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from music_transformer.amt import AMTGenerator


def make_handler(generator):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != '/respond':
                self.send_error(404)
                return
            body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            with tempfile.TemporaryDirectory() as tmp:
                midi_in, midi_out = os.path.join(tmp, 'in.mid'), os.path.join(tmp, 'out.mid')
                with open(midi_in, 'wb') as f:
                    f.write(body)
                start = time.time()
                try:
                    with lock:
                        ok = generator.respond(midi_in, midi_out)
                except Exception as e:
                    print(f"Error generating: {e}")
                    self.send_error(500, str(e))
                    return
                print(f"respond: {'ok' if ok else 'empty'} in {time.time() - start:.1f}s")
                if not ok:
                    self.send_response(204)
                    self.end_headers()
                    return
                with open(midi_out, 'rb') as f:
                    data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'audio/midi')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    generator = AMTGenerator()
    print(f"AMT model loaded on {generator.device}, listening on {args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), make_handler(generator)).serve_forever()
