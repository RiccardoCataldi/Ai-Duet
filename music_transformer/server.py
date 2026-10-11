"""
Serve AMTGenerator over HTTP so the model can run on another machine (e.g. a Jetson).

    python -m music_transformer.server            # listens on 127.0.0.1:8765
    ssh -N -L 8765:localhost:8765 jetson          # on the laptop, then AMT_SERVER=http://localhost:8765

POST /respond with the prompt MIDI file as the body; the reply is the generated MIDI file,
or 204 when there is nothing to respond to.

POST /stream with {"notes": [[onset, duration, pitch], ...], "start": s, "seconds": s} as JSON;
the reply streams one "onset duration pitch" line per note as soon as it is sampled.
Closing the connection stops the generation.
"""

import argparse
import json
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
            if self.path == '/stream':
                self.stream()
                return
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

        def stream(self):
            request = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            start, count = time.time(), 0
            with lock:
                answer = generator.stream(request['notes'], request['start'], request.get('seconds', 8))
                try:
                    for onset, duration, pitch in answer:
                        if count == 0:
                            print(f"stream: first note after {time.time() - start:.2f}s")
                        self.wfile.write(f"{onset:.2f} {duration:.2f} {pitch}\n".encode())
                        count += 1
                except (BrokenPipeError, ConnectionResetError):
                    print(f"stream: cut short after {count} notes")
                    return
                finally:
                    answer.close()
            print(f"stream: {count} notes in {time.time() - start:.1f}s")

        def log_message(self, *args):
            pass

    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model', default=None, help='Hugging Face name or local path of the AMT model')
    args = parser.parse_args()
    generator = AMTGenerator(model_name=args.model)
    print(f"AMT model loaded on {generator.device}, listening on {args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), make_handler(generator)).serve_forever()
