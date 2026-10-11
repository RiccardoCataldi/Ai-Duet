"""
Serve the duo over TCP so the model can run on another machine (e.g. a Jetson).

    python -m music_transformer.server            # listens on 127.0.0.1:8765
    ssh -N -L 8765:localhost:8765 jetson          # on the laptop, then AMT_SERVER=localhost:8765

One line per message, times in seconds since the connection opened:
    client -> server   "on <onset> <duration> <pitch>"   you pressed a key (the duration is a guess)
    server -> client   "note <onset> <duration> <pitch>" the model's next note
    server -> client   "cut <onset>"                     forget the model's notes from <onset> on: they are resampled
One duo at a time; a second connection waits for the first to close.
"""

import argparse
import queue
import socketserver
import threading

from anticipation.config import TIME_RESOLUTION

from music_transformer.amt import AMTGenerator
from music_transformer.duo import LOOKAHEAD, DuoSession


def make_handler(generator):
    lock = threading.Lock()

    class Handler(socketserver.StreamRequestHandler):
        disable_nagle_algorithm = True  # every note goes out at once

        def handle(self):
            with lock:
                print("duo: started")
                inbox = queue.Queue()
                threading.Thread(target=self.read, args=(inbox,), daemon=True).start()
                session, heard, played = DuoSession(generator.model, generator.device), 0, 0
                try:
                    while True:
                        # your keys come first; otherwise sample on, or wait while far enough ahead
                        wait = None if session.base is None else max(session.ahead() - LOOKAHEAD, 0) / TIME_RESOLUTION
                        try:
                            note = inbox.get(timeout=wait) if wait != 0 else inbox.get_nowait()
                        except queue.Empty:
                            note = False
                        if note is None:
                            break
                        if note:
                            heard += 1
                            cut = session.hear(*note)
                            if cut is not None:
                                self.wfile.write(f"cut {cut:.3f}\n".encode())
                        elif session.ahead() < LOOKAHEAD:
                            onset, duration, pitch = session.play()
                            self.wfile.write(f"note {onset:.3f} {duration:.3f} {pitch}\n".encode())
                            played += 1
                except (BrokenPipeError, ConnectionResetError):
                    pass
                print(f"duo: ended, heard {heard} notes, played {played}")

        def read(self, inbox):
            try:
                for line in self.rfile:
                    kind, *values = line.split()
                    if kind == b'on':
                        onset, duration, pitch = values
                        inbox.put((float(onset), float(duration), int(pitch)))
            except (ConnectionResetError, ValueError) as e:
                print(f"duo: bad input ({e})")
            finally:
                inbox.put(None)

    return Handler


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model', default=None, help='Hugging Face name or local path of the AMT model')
    args = parser.parse_args()
    generator = AMTGenerator(model_name=args.model)
    print(f"AMT model loaded on {generator.device}, listening on {args.host}:{args.port}")
    Server((args.host, args.port), make_handler(generator)).serve_forever()
