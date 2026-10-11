import heapq
import os
import queue
import sys
import threading
import time

import mido

from CK_rec.setup import Setup
from CK_rec.rec_classes import CK_rec

PAUSE_S = 0.5    # silence after your phrase that hands the turn to the model
ANSWER_S = 8     # longest answer, in seconds
WINDOW_S = 30    # how much of the duet the model hears
LEAD_S = 0.05    # head start given to the first answered note
INPUT_PROGRAM = 0        # Acoustic Grand Piano
GENERATED_PROGRAM = 48   # String Ensemble


def load_generator():
    server = os.environ.get('AMT_SERVER')
    if server:
        from music_transformer.remote import RemoteAMTGenerator
        print(f"Using remote AMT model at {server}")
        return RemoteAMTGenerator(server)
    from music_transformer.amt import AMTGenerator
    return AMTGenerator()


class Duet:
    """
    Call and response: when you pause, the model answers note by note as it samples them;
    when you play again, it stops and listens.
    """

    def __init__(self, generator, midi_rec):
        self.generator = generator
        self.midi_rec = midi_rec
        self.lock = threading.Lock()
        self.notes = []          # (onset, offset, pitch, velocity, generated) in time.monotonic() seconds
        self.held = {}           # pitch -> (onset, velocity)
        self.velocity = 80
        self.last_played = None  # last key press/release not yet answered
        self.answer = None       # threading.Event that stops the answer being played

    def note_on(self, pitch, velocity, now):
        with self.lock:
            if self.answer:
                self.answer.set()
                self.answer = None
            self.held[pitch] = (now, velocity)
            self.velocity = velocity
            self.last_played = now

    def note_off(self, pitch, now):
        with self.lock:
            if pitch not in self.held:
                return
            onset, velocity = self.held.pop(pitch)
            self.notes.append((onset, now, pitch, velocity, False))
            self.last_played = now

    def poll(self, now):
        with self.lock:
            if self.held or self.answer or self.last_played is None or now - self.last_played < PAUSE_S:
                return
            self.last_played = None
            heard = [n for n in self.notes if n[0] >= now - WINDOW_S]
            if not heard:
                return
            origin = min(n[0] for n in heard)
            prompt = [(onset - origin, offset - onset, pitch) for onset, offset, pitch, _, _ in heard]
            self.answer = threading.Event()
            # the model answers best right where your phrase ends; the pause is only how we notice it
            end = max(n[1] for n in heard)
            args = (self.answer, prompt, origin, end - origin, self.velocity)
        threading.Thread(target=self._play, args=args, daemon=True).start()

    def _play(self, stop, prompt, origin, start, velocity):
        incoming = queue.Queue()

        def read():
            try:
                for note in self.generator.stream(prompt, start, ANSWER_S):
                    if stop.is_set():
                        break
                    incoming.put(note)
            except Exception as e:
                print(f"Error generating music: {e}")
            finally:
                incoming.put(None)

        threading.Thread(target=read, daemon=True).start()
        pending, sounding, shift, done = [], {}, None, False
        while not stop.is_set() and not (done and not pending):
            wait = 0.01 if not pending else min(max(pending[0][0] - time.monotonic(), 0), 0.01)
            try:
                note = incoming.get(timeout=wait)
            except queue.Empty:
                note = False
            if note is None:
                done = True
            elif note:
                onset, duration, pitch = note
                at = origin + onset
                if shift is None:
                    # the first note is due when your phrase ended: play it now and keep the rest in step
                    shift = max(time.monotonic() + LEAD_S - at, 0)
                at += shift
                heapq.heappush(pending, (at, 1, pitch))
                heapq.heappush(pending, (at + duration, 0, pitch))
            now = time.monotonic()
            while pending and pending[0][0] <= now:
                _, on, pitch = heapq.heappop(pending)
                if pitch in sounding:
                    self._release(pitch, sounding.pop(pitch), now, velocity)
                if on:
                    self.midi_rec.play_midi_message([0x90, pitch, velocity], is_generated=True)
                    sounding[pitch] = now
        now = time.monotonic()
        for pitch, onset in sounding.items():
            self._release(pitch, onset, now, velocity)
        with self.lock:
            if self.answer is stop:
                self.answer = None

    def _release(self, pitch, onset, now, velocity):
        self.midi_rec.play_midi_message([0x80, pitch, 0], is_generated=True)
        with self.lock:
            self.notes.append((onset, now, pitch, velocity, True))

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
            last = 0
            for at, kind, pitch, velocity in sorted(messages):
                track.append(mido.Message(kind, channel=channel, note=pitch, velocity=velocity, time=ticks(at) - last))
                last = ticks(at)
            mid.tracks.append(track)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mid.save(path)
        print(f"Duet saved as {path}")


if __name__ == "__main__":
    codeK = Setup()
    myPort = codeK.perform_setup()
    codeK.open_port(myPort)

    midiRec = CK_rec(myPort, 0x90, debug=False)
    midiRec.set_input_instrument(INPUT_PROGRAM)
    midiRec.set_generated_instrument(GENERATED_PROGRAM)
    duet = Duet(load_generator(), midiRec)

    def on_midi(event, data=None):
        message, _ = event
        if len(message) < 3:
            return
        kind, pitch, velocity = message[0] & 0xF0, message[1], message[2]
        if kind == 0x90 and velocity > 0:
            midiRec.play_midi_message(message)
            duet.note_on(pitch, velocity, time.monotonic())
        elif kind in (0x80, 0x90):
            midiRec.play_midi_message(message)
            duet.note_off(pitch, time.monotonic())

    codeK.set_callback(on_midi)
    print("Play a phrase and pause: the model answers. Ctrl+C to stop.")

    try:
        while True:
            duet.poll(time.monotonic())
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        codeK.end()
        duet.save(f"generated/{int(time.time())}.mid")
        print("Recording Stopped")
        sys.exit(0)
