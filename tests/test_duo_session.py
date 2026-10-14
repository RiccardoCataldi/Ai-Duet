"""Needs torch and anticipation: run it where the model runs (python -m unittest tests.test_duo_session)."""
import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anticipation.config import CONTEXT_SIZE, MAX_PITCH
from anticipation.vocab import NOTE_OFFSET, TIME_OFFSET, VOCAB_SIZE
from music_transformer import duo
from music_transformer.duo import DuoSession


class Cache:
    def __init__(self):
        self.length = 0

    def crop(self, length):
        self.length = length


class UniformModel:
    """Every token equally likely; checks the cache stays in step with the tokens fed."""

    def __call__(self, inp, past_key_values=None, use_cache=True):
        past = past_key_values or Cache()
        past.length += inp.shape[1]
        self.fed = past.length
        return SimpleNamespace(logits=torch.zeros(1, inp.shape[1], VOCAB_SIZE), past_key_values=past)


class StuckModel(UniformModel):
    """Always wants the earliest time and pitch 61: the loop AMT falls into once a doubled note is in its context."""

    def __call__(self, inp, past_key_values=None, use_cache=True):
        out = super().__call__(inp, past_key_values, use_cache)
        out.logits[0, -1, TIME_OFFSET:TIME_OFFSET + 10000] = -torch.arange(10000).float()
        out.logits[0, -1, NOTE_OFFSET + 4 * MAX_PITCH + 61] = 50
        return out


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class DuoSessionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.clock, self.model = Clock(), UniformModel()
        self.session = DuoSession(self.model, 'cpu', clock=self.clock)

    def fill(self):
        while self.session.ahead() < duo.LOOKAHEAD:
            yield self.session.play()

    def test_the_model_plays_its_own_instrument_after_now_without_long_rests(self):
        self.clock.t = 3.0
        self.session.hear(3.0, 0.3, 60)
        notes = list(self.fill())
        self.assertGreater(len(notes), 0)
        onsets = [onset for onset, _, _ in notes]
        self.assertEqual(onsets, sorted(onsets))
        self.assertGreaterEqual(onsets[0], 3.0 + duo.MARGIN / 100)
        self.assertLessEqual(max(b - a for a, b in zip([3.0] + onsets, onsets)), duo.MAX_REST / 100 + 1e-9)
        instruments = {(n - NOTE_OFFSET) // MAX_PITCH for n in self.session.seq[3::3][1:]}
        self.assertEqual(instruments, {self.session.me})

    def test_your_note_drops_only_what_the_model_has_not_played_yet(self):
        self.session.hear(0.0, 0.3, 60)
        notes = list(self.fill())
        self.clock.t = 0.5
        cut = self.session.hear(0.5, 0.3, 62)
        kept = [n for n in notes if n[0] <= 0.5 + duo.MARGIN / 100]
        dropped = [n for n in notes if n[0] > 0.5 + duo.MARGIN / 100]
        self.assertEqual(cut, dropped[0][0] if dropped else None)
        self.assertEqual(len(self.session.seq), 1 + 3 * (1 + len(kept) + 1))
        self.assertEqual(self.session.seq[-1], NOTE_OFFSET + 62)
        self.session.play()
        self.assertEqual(self.model.fed, len(self.session.seq) - 1)

    def test_a_full_context_is_trimmed_and_restarts_at_time_zero(self):
        self.session.hear(0.0, 0.3, 60)
        longest = 0
        for _ in range(2000):
            self.clock.t += 0.05
            list(self.fill())
            longest = max(longest, len(self.session.seq))
            if self.session.base > 0:
                break
        self.assertGreater(self.session.base, 0)
        self.assertLessEqual(longest, CONTEXT_SIZE)
        self.assertLessEqual(len(self.session.seq), duo.KEEP + 30)
        self.assertEqual(self.session.seq[1], TIME_OFFSET)


    def test_the_model_stays_a_soloist_even_when_it_wants_to_loop(self):
        session = DuoSession(StuckModel(), 'cpu', clock=self.clock)
        session.hear(0.0, 0.3, 60)
        notes = [session.play() for _ in range(40)]
        onsets = {}
        for onset, _, pitch in notes:
            onsets.setdefault(onset, []).append(pitch)
        for pitches in onsets.values():
            self.assertEqual(len(pitches), len(set(pitches)))
            self.assertLessEqual(len(pitches), duo.MAX_CHORD)
        times = sorted(onsets)
        self.assertGreaterEqual(min(b - a for a, b in zip(times, times[1:])), duo.MIN_STEP / 100 - 1e-9)
        pitches = [pitch for _, _, pitch in notes]
        runs = [tuple(pitches[i:i + duo.NO_REPEAT]) for i in range(len(pitches) - duo.NO_REPEAT + 1)]
        self.assertEqual(len(runs), len(set(runs)))

    def test_a_knob_turned_while_it_plays_changes_its_next_notes(self):
        session = DuoSession(StuckModel(), 'cpu', clock=self.clock)
        session.hear(0.0, 0.3, 60)
        for _ in range(10):
            session.play()
        session.turn(min_step=50, top_p=2.0)
        self.assertEqual(session.knobs()['top_p'], 1.0)   # kept within its range
        times = sorted({session.play()[0] for _ in range(20)})
        self.assertGreaterEqual(min(b - a for a, b in zip(times, times[1:])), 0.5 - 1e-9)

    def test_knobs_given_at_the_start_hold_from_the_first_note(self):
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, memory=60.4, stay=0.9)
        self.assertEqual(session.memory, 60)
        self.assertEqual(session.knobs()['stay'], 0.9)

    def test_the_model_remembers_all_of_you_but_only_the_last_moments_of_itself(self):
        self.session.hear(0.0, 0.3, 60)
        while self.clock.t < 5:
            self.clock.t += 0.05
            list(self.fill())
        self.session.hear(self.clock.t, 0.3, 62)
        mine = [e[0] - TIME_OFFSET for e in self.session._events() if self.session._instrument(e[2]) == self.session.me]
        yours = [e[2] - NOTE_OFFSET for e in self.session._events() if self.session._instrument(e[2]) == self.session.you]
        self.assertEqual(yours, [60, 62])
        self.assertEqual(self.session.seq[1], TIME_OFFSET)
        self.assertGreaterEqual(min(mine), self.session.heard_at - duo.MEMORY)
        self.session.play()
        self.assertEqual(self.model.fed, len(self.session.seq) - 1)

    def test_the_model_prefers_the_pitch_classes_you_are_playing(self):
        for i, pitch in enumerate((60, 64, 67, 72)):   # C major triad
            self.session.hear(i * 0.2, 0.2, pitch)
        self.clock.t = 0.6
        notes = []
        for _ in range(60):
            notes.append(self.session.play())
            self.clock.t = notes[-1][0]
            self.session.hear(self.clock.t, 0.2, (60, 64, 67)[len(notes) % 3])
        in_key = sum(pitch % 12 in (0, 4, 7) for _, _, pitch in notes) / len(notes)
        self.assertGreater(in_key, 0.35)   # 3 pitch classes of 12: 25% without ears

    def play_scale(self, pitches, start):
        for i, pitch in enumerate(pitches):
            self.clock.t = start + i * 0.3
            self.session.hear(self.clock.t, 0.2, pitch)
        self.clock.t += 1.5   # you rest: the model answers and leads

    def test_after_you_modulate_it_plays_in_the_new_key(self):
        c_major = [60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60, 62, 65, 69, 71, 67]
        self.play_scale(c_major, 0.1)
        self.play_scale([p + 3 for p in c_major], self.clock.t)
        pitches = [pitch % 12 for _ in range(4) for _, _, pitch in self.fill()] + \
                  [self.session.play()[2] % 12 for _ in range(60)]
        self.assertGreater(sum(p in (3, 5, 7, 8, 10, 0, 2) for p in pitches) / len(pitches), 0.75)   # 58% by chance
        self.assertLess(sum(p in (4, 11) for p in pitches) / len(pitches), 0.1)   # only in C: A stays possible, as in Bb

    def test_its_onsets_fall_on_your_pulse(self):
        self.play_scale([60, 62, 64, 65, 67, 69, 71, 72], 0.1)
        onsets = [self.session.play()[0] for _ in range(30)]
        for onset in onsets:
            off = (onset - 0.1) % 0.3
            self.assertLessEqual(min(off, 0.3 - off), duo.GRID / 100 + 1e-9)

if __name__ == '__main__':
    unittest.main()
