import os
import signal
import socket
import statistics
import sys
import threading
import time
from collections import deque

import mido

from CK_rec.setup import Setup
from CK_rec.rec_classes import CK_rec

LATE_S = 0.15            # a note of the model that arrives later than this is skipped, not played out of time
GUESS_S = 0.3            # how long we guess your note will last before you have played any
INPUT_PROGRAM = 0        # Acoustic Grand Piano
GENERATED_PROGRAM = 4    # Electric Piano 1
# the model's sound: a warm, sustained electric piano in a big room
GENERATED_CONTROLS = ((7, 100), (91, 115), (93, 60))  # volume, reverb send, chorus send
SOFTER = 0.8             # the model plays a little softer than you: a darker, warmer electric piano
RING = 1.6               # its notes ring this much longer than the model wrote them, as if with a light pedal
RING_S = 0.4             # and at least this many seconds more


class Duo:
    """
    You and the model play at the same time. Every key you press goes to the model straight away; it sends back
    its own line a little ahead of time, and redoes the part that has not sounded yet whenever you play.
    `send` takes one protocol line (see music_transformer/server.py); `started` is when the connection opened.
    """

    def __init__(self, send, midi_rec, started):
        self.send = send
        self.midi_rec = midi_rec
        self.started = started
        self.lock = threading.Lock()
        self.notes = []                # (onset, offset, pitch, velocity, generated) in time.monotonic() seconds
        self.held = {}                 # pitch -> (onset, velocity)
        self.durations = deque(maxlen=16)
        self.velocity = 80
        self.pending = []              # the model's notes not started yet: (onset, duration, pitch)
        self.sounding = {}             # pitch -> (onset, offset) of the model's notes

    def note_on(self, pitch, velocity, now):
        with self.lock:
            self.held[pitch] = (now, velocity)
            self.velocity = velocity
            guess = statistics.median(self.durations) if self.durations else GUESS_S
        self.send(f"on {now - self.started:.3f} {guess:.3f} {pitch}\n")

    def note_off(self, pitch, now):
        with self.lock:
            if pitch not in self.held:
                return
            onset, velocity = self.held.pop(pitch)
            self.durations.append(now - onset)
            self.notes.append((onset, now, pitch, velocity, False))

    def receive(self, line):
        kind, *values = line.split()
        with self.lock:
            if kind == 'note':
                onset, duration, pitch = values
                self.pending.append((self.started + float(onset), float(duration), int(pitch)))
            elif kind == 'cut':
                cut = self.started + float(values[0])
                self.pending = [n for n in self.pending if n[0] < cut]

    def tick(self, now):
        """Starts and stops the model's notes that are due."""
        with self.lock:
            for pitch, (onset, offset) in list(self.sounding.items()):
                if offset <= now:
                    self._release(pitch, now)
            due = [n for n in self.pending if n[0] <= now]
            self.pending = [n for n in self.pending if n[0] > now]
            for onset, duration, pitch in sorted(due):
                if now - onset > LATE_S:
                    continue
                if pitch in self.sounding:
                    self._release(pitch, now)
                self.midi_rec.play_midi_message([0x90, pitch, self._velocity()], is_generated=True)
                self.sounding[pitch] = (now, now + max(duration * RING, duration + RING_S))

    def stop(self, now):
        with self.lock:
            self.pending = []
            for pitch in list(self.sounding):
                self._release(pitch, now)

    def _release(self, pitch, now):
        onset, _ = self.sounding.pop(pitch)
        self.midi_rec.play_midi_message([0x80, pitch, 0], is_generated=True)
        self.notes.append((onset, now, pitch, self._velocity(), True))

    def _velocity(self):
        return max(1, round(self.velocity * SOFTER))

    def save(self, path):
        if not self.notes:
            return
        mid = mido.MidiFile()
        ticks = lambda seconds: int(mido.second2tick(seconds, mid.ticks_per_beat, mido.bpm2tempo(120)))
        origin = min(n[0] for n in self.notes)
        for generated, channel, program in ((False, 0, INPUT_PROGRAM), (True, 1, GENERATED_PROGRAM)):
            messages = []
            for onset, offset, pitch, velocity, is_generated in self.notes:
                if is_generated == generated:
                    messages.append((onset - origin, 'note_on', pitch, velocity))
                    messages.append((offset - origin, 'note_off', pitch, 0))
            track = mido.MidiTrack([mido.Message('program_change', channel=channel, program=program)])
            if generated:
                track += [mido.Message('control_change', channel=channel, control=c, value=v) for c, v in GENERATED_CONTROLS]
            last = 0
            for at, kind, pitch, velocity in sorted(messages):
                track.append(mido.Message(kind, channel=channel, note=pitch, velocity=velocity, time=ticks(at) - last))
                last = ticks(at)
            mid.tracks.append(track)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mid.save(path)
        print(f"Duet saved as {path}")


def connect(address):
    host, port = address.rsplit(':', 1)
    sock = socket.create_connection((host, int(port)))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"Playing with the AMT model at {address}")
    return sock, time.monotonic()


if __name__ == "__main__":
    codeK = Setup()
    myPort = codeK.perform_setup()
    codeK.open_port(myPort)

    midiRec = CK_rec(myPort, 0x90, debug=False)
    midiRec.set_input_instrument(INPUT_PROGRAM)
    midiRec.set_generated_instrument(GENERATED_PROGRAM)
    for control, value in GENERATED_CONTROLS:
        midiRec.play_midi_message([0xB0 | midiRec.generated_channel, control, value])
    sock, started = connect(os.environ.get('AMT_SERVER', 'localhost:8765'))
    duo = Duo(lambda line: sock.sendall(line.encode()), midiRec, started)

    def listen():
        for line in sock.makefile('r'):
            duo.receive(line)
        print("The AMT server closed the duo")

    threading.Thread(target=listen, daemon=True).start()

    def on_midi(event, data=None):
        message, _ = event
        if len(message) < 3:
            return
        kind, pitch, velocity = message[0] & 0xF0, message[1], message[2]
        if kind == 0x90 and velocity > 0:
            midiRec.play_midi_message(message)
            duo.note_on(pitch, velocity, time.monotonic())
        elif kind in (0x80, 0x90):
            midiRec.play_midi_message(message)
            duo.note_off(pitch, time.monotonic())

    codeK.set_callback(on_midi)
    # run_duet.sh starts us in the background, where SIGINT is ignored: stop and save on SIGINT or SIGTERM anyway
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.default_int_handler)
    print("Play: the model improvises with you. Ctrl+C to stop.")

    try:
        while True:
            duo.tick(time.monotonic())
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        codeK.end()
        duo.stop(time.monotonic())
        sock.close()
        duo.save(f"generated/{int(time.time())}.mid")
        print("Recording Stopped")
        sys.exit(0)
