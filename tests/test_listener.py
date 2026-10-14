import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from music_transformer.listener import Listener

C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60, 62, 65, 69, 71, 67]
EB_MAJOR = [p + 3 for p in C_MAJOR]


def play(listener, pitches, start=0.0, step=0.25):
    for i, pitch in enumerate(pitches):
        listener.hear(start + i * step, 0.2, pitch)
    return start + len(pitches) * step


class ListenerTest(unittest.TestCase):
    def test_it_finds_your_key_and_follows_a_modulation_within_a_bar_or_two(self):
        listener = Listener()
        end = play(listener, C_MAJOR)
        self.assertEqual(listener.name(listener.key()[0]), 'C major')
        keys = []
        for i, pitch in enumerate(EB_MAJOR):
            listener.hear(end + i * 0.25, 0.2, pitch)
            keys.append(listener.name(listener.key()[0]))
        self.assertEqual(keys[-1], 'Eb major')
        self.assertLessEqual(keys.index('Eb major'), 10)
        self.assertEqual(keys[keys.index('Eb major'):], ['Eb major'] * (len(keys) - keys.index('Eb major')))

    def test_a_lower_stay_follows_a_modulation_sooner(self):
        heard = []
        for stay in (0.97, 0.85):
            listener = Listener()
            listener.stay(stay)
            end = play(listener, C_MAJOR)
            for i, pitch in enumerate(EB_MAJOR):
                listener.hear(end + i * 0.25, 0.2, pitch)
                if listener.name(listener.key()[0]) == 'Eb major':
                    heard.append(i)
                    break
        self.assertLess(heard[1], heard[0])

    def test_a_passing_note_does_not_change_the_key(self):
        listener = Listener()
        end = play(listener, C_MAJOR)
        listener.hear(end, 0.2, 66)
        self.assertEqual(listener.name(listener.key()[0]), 'C major')

    def test_it_names_the_chord_you_hold(self):
        listener = Listener()
        play(listener, C_MAJOR)
        for pitch in (55, 59, 62, 65):   # G7
            listener.hear(10.0, 1.0, pitch)
        self.assertEqual(listener.chord(10.3), {7, 11, 2, 5})

    def test_it_finds_your_pulse_and_hears_where_your_phrase_ends(self):
        listener = Listener()
        end = play(listener, C_MAJOR, start=0.1, step=0.3)
        tatum, phase = listener.pulse(end)
        self.assertAlmostEqual(tatum, 0.3)
        self.assertAlmostEqual(phase, 0.1, delta=0.02)
        self.assertEqual(listener.phrase(end - 0.2), [])
        self.assertEqual(listener.phrase(end + 1.0), C_MAJOR)


if __name__ == '__main__':
    unittest.main()
