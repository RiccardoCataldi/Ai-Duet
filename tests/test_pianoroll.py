import os
import tempfile
import unittest

import mido

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import pianoroll


class PianoRollTest(unittest.TestCase):
    def test_a_saved_duo_becomes_your_notes_and_the_models_in_seconds(self):
        mid = mido.MidiFile()
        for channel, pitch in ((0, 60), (1, 67)):
            mid.tracks.append(mido.MidiTrack([
                mido.Message('note_on', channel=channel, note=pitch, velocity=90, time=0),
                mido.Message('note_off', channel=channel, note=pitch, velocity=0, time=mid.ticks_per_beat)]))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'duo.mid')
            mid.save(path)
            notes = pianoroll.notes_from_midi(path)
        self.assertEqual(sorted((p, c) for _, _, p, c in notes), [(60, 0), (67, 1)])
        for start, end, _, _ in notes:
            self.assertAlmostEqual(start, 0)
            self.assertAlmostEqual(end, 0.5)  # a beat at 120 bpm

    def test_live_notes_are_held_until_released_and_dropped_once_off_screen(self):
        live = pianoroll.Live()
        live.message([0x91, 64, 80], 10.0)
        self.assertEqual(live.visible(10.5, 600), [[10.0, None, 64, 1]])
        live.message([0x91, 64, 0], 11.0)
        self.assertEqual(live.visible(11.0, 600), [[10.0, 11.0, 64, 1]])
        self.assertEqual(live.visible(11.1 + 600 / pianoroll.SPEED, 600), [])

    def test_the_keyboard_spans_whole_octaves_around_the_notes(self):
        self.assertEqual(pianoroll.octaves([(0, 1, 50, 0), (0, 1, 73, 1)]), (48, 84))


if __name__ == '__main__':
    unittest.main()
