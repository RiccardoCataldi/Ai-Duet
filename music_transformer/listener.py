"""
Listens to your playing the way a duo partner does: the key you are in and when you modulate, the chord under your
hands, your pulse and your last phrase. Times are seconds since the connection opened.
"""

import cmath
import math

import numpy as np

NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]   # Krumhansl-Kessler, from C
MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
SCALES = ((0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10, 11))                      # major, minor with both sevenths
CHORDS = {'': (0, 4, 7), 'm': (0, 3, 7), '7': (0, 4, 7, 10), 'm7': (0, 3, 7, 10), 'dim': (0, 3, 6)}

STAY = 0.97      # chance per note that the key stays: lower follows a modulation sooner but flickers on passing notes
SHARPNESS = 2.0  # how much one note says about the key
FAR = 0.6        # each step round the circle of fifths makes a modulation there about 1.8 times less likely
CHORD_S = 0.4    # half-life of the notes that make the chord
PULSE_S = 6.0    # seconds of your onsets the pulse is read from
PHRASE = 1.5     # a rest of this many beats ends your phrase
KEEP_S = 20.0    # seconds of your notes kept


MEMBER = np.array([[(pc - k % 12) % 12 in SCALES[k >= 12] for pc in range(12)] for k in range(24)], float)


def _keys():
    profiles = np.array([np.roll(MAJOR, k) for k in range(12)] + [np.roll(MINOR, k) for k in range(12)])
    emit = profiles / profiles.sum(1, keepdims=True)
    signature = [7 * (k % 12 + (3 if k >= 12 else 0)) % 12 for k in range(24)]   # a minor key shares its relative's
    far = np.array([[min((a - b) % 12, (b - a) % 12) + 0.5 * ((i < 12) != (j < 12))
                     for j, b in enumerate(signature)] for i, a in enumerate(signature)])
    move = np.exp(-FAR * far)
    np.fill_diagonal(move, 0)
    move = (1 - STAY) * move / move.sum(1, keepdims=True)
    np.fill_diagonal(move, STAY)
    return np.log(emit) * SHARPNESS, move


EMIT, MOVE = _keys()


class Listener:
    def __init__(self):
        self.belief = np.full(24, 1 / 24)
        self.notes = []   # (onset, duration, pitch), oldest first

    def hear(self, onset, duration, pitch):
        belief = (self.belief @ MOVE) * np.exp(EMIT[:, pitch % 12])
        self.belief = belief / belief.sum()
        self.notes = [n for n in self.notes if n[0] > onset - KEEP_S] + [(onset, duration, pitch)]

    def key(self):
        """(key, how likely): 0-11 major from C, 12-23 minor."""
        k = int(self.belief.argmax())
        return k, float(self.belief[k])

    @staticmethod
    def name(key):
        return f"{NAMES[key % 12]} {'minor' if key >= 12 else 'major'}"

    def in_key(self):
        """For each pitch class, how likely it belongs to your key: while you modulate, both keys count."""
        return self.belief @ MEMBER

    def chord(self, now):
        """Pitch classes of the chord you are playing now, or None when nothing is sounding."""
        weight = np.zeros(12)
        for onset, duration, pitch in self.notes:
            if onset <= now:
                weight[pitch % 12] += 0.5 ** (max(now - onset - duration, 0) / CHORD_S)
        if weight.sum() < 0.5:
            return None
        scale = set(np.flatnonzero(self.in_key() > 0.5))
        best, chord = -1, None
        for root in range(12):
            for shape in CHORDS.values():
                tones = {(root + step) % 12 for step in shape}
                score = sum(weight[p] for p in tones) / math.sqrt(len(tones)) / np.linalg.norm(weight) \
                    + 0.1 * tones.issubset(scale)
                if score > best:
                    best, chord = score, tones
        return chord

    def pulse(self, now):
        """(tatum, phase) in seconds: your onsets fall on phase + k * tatum. None until your rhythm is regular."""
        onsets = sorted({round(n[0], 2) for n in self.notes if n[0] > now - PULSE_S})
        if len(onsets) < 6:
            return None
        fit = {p / 100: abs(sum(cmath.exp(2j * math.pi * t * 100 / p) for t in onsets)) / len(onsets)
               for p in range(15, 101)}
        best = max(fit.values())
        if best < 0.6:
            return None
        tatum = max(p for p, r in fit.items() if r >= 0.9 * best)
        phase = cmath.phase(sum(cmath.exp(2j * math.pi * t / tatum) for t in onsets)) / (2 * math.pi) * tatum
        return tatum, phase % tatum

    @staticmethod
    def beat(tatum):
        return tatum * math.ceil(0.4 / tatum)

    def phrase(self, now):
        """Pitches of your last phrase, once a rest has ended it; [] while you are still playing it."""
        pulse = self.pulse(now)
        rest = PHRASE * (self.beat(pulse[0]) if pulse else 0.4)
        if not self.notes or now - self.notes[-1][0] < rest:
            return []
        start = len(self.notes) - 1
        while start > 0 and self.notes[start][0] - self.notes[start - 1][0] < rest:
            start -= 1
        return [n[2] for n in self.notes[start:]]
