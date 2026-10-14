"""
Serve the duo over TCP so the model can run on another machine (e.g. a Jetson).

    python -m music_transformer.server            # listens on 127.0.0.1:8765
    ssh -N -L 8765:localhost:8765 jetson          # on the laptop, then AMT_SERVER=localhost:8765

One line per message, times in seconds since the connection opened:
    client -> server   "on <onset> <duration> <pitch>"   you pressed a key (the duration is a guess)
    server -> client   "note <onset> <duration> <pitch>" the model's next note
    server -> client   "cut <onset>"                     forget the model's notes from <onset> on: they are resampled
One duo at a time; a second connection waits for the first to close.

A connection that starts with "knobs" turns the duo's knobs (duo.KNOBS) instead, while a duo plays or before:
    server -> client   "knob <name> <value> <lowest> <highest> <default> <unit> <what it does>"   one per knob, then "ready"
    client -> server   "set <name> <value>"              from the model's next note on, and for the next duos
"""

import argparse
import itertools
import queue
import socketserver
import threading
import time

from anticipation.config import TIME_RESOLUTION

from music_transformer.amt import AMTGenerator
from music_transformer.duo import KNOBS, LOOKAHEAD, DuoSession, knob


def make_handler(generator):
    lock = threading.Lock()
    knobs = {name: default for name, (default, *_) in KNOBS.items()}   # as last turned, for this duo and the next
    playing = []                                                         # the session of the duo now playing

    class Handler(socketserver.StreamRequestHandler):
        disable_nagle_algorithm = True  # every note goes out at once

        def handle(self):
            opened = time.monotonic()   # times on the wire count from here
            first = self.rfile.readline()
            if first.split()[:1] == [b'knobs']:
                self.turn()
            else:
                self.duo(first, opened)

        def turn(self):
            for name, (_, low, high, unit, what) in KNOBS.items():
                self.wfile.write(f"knob {name} {knobs[name]} {low} {high} {KNOBS[name][0]} {unit or '-'} {what}\n".encode())
            self.wfile.write(b"ready\n")
            try:
                for line in self.rfile:
                    kind, *values = line.decode().split()
                    if kind == 'set' and len(values) == 2 and values[0] in KNOBS:
                        name = values[0]
                        knobs[name] = knob(name, float(values[1]))
                        for session in playing:
                            session.turn(**{name: knobs[name]})
                        print(f"duo: {name} = {knobs[name]}")
            except (ConnectionResetError, ValueError) as e:
                print(f"duo: bad knob ({e})")

        def duo(self, first, opened):
            with lock:
                print("duo: started")
                inbox = queue.Queue()
                threading.Thread(target=self.read, args=(inbox, first), daemon=True).start()
                session, heard, played, shown = DuoSession(generator.model, generator.device, started=opened, **knobs), 0, 0, None
                playing.append(session)
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
                            key, sure = session.listener.key()
                            if key != shown and sure > 0.3:
                                shown = key
                                print(f"duo: {note[0]:.1f}s you are in {session.listener.name(key)} ({sure:.0%})")
                            if cut is not None:
                                self.wfile.write(f"cut {cut:.3f}\n".encode())
                        elif session.ahead() < LOOKAHEAD:
                            onset, duration, pitch = session.play()
                            self.wfile.write(f"note {onset:.3f} {duration:.3f} {pitch}\n".encode())
                            played += 1
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    playing.remove(session)
                print(f"duo: ended, heard {heard} notes, played {played}")

        def read(self, inbox, first):
            try:
                for line in itertools.chain([first], self.rfile):
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
