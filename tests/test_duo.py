import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main


class Out:
    def __init__(self):
        self.messages = []

    def play_midi_message(self, message, is_generated=False):
        self.messages.append((list(message), is_generated))

    def generated_on(self):
        return [m[1] for m, generated in self.messages if generated and m[0] == 0x90]


class DuoTest(unittest.TestCase):
    def setUp(self):
        self.sent, self.out = [], Out()
        self.duo = main.Duo(self.sent.append, self.out, started=100.0)

    def test_your_key_goes_to_the_model_at_once_with_a_guessed_duration(self):
        self.duo.note_on(60, 90, 101.5)
        self.assertEqual(self.sent, [f"on 1.500 {main.GUESS_S:.3f} 60\n"])

    def test_the_guess_follows_how_long_you_hold_your_notes(self):
        for i, pitch in enumerate((60, 62, 64)):
            self.duo.note_on(pitch, 90, 101 + i)
            self.duo.note_off(pitch, 101 + i + 0.1)
        self.duo.note_on(65, 90, 105)
        self.assertEqual(self.sent[-1], "on 5.000 0.100 65\n")

    def test_the_model_plays_its_notes_when_they_are_due_and_releases_them(self):
        self.duo.receive("note 2.000 0.500 67\n")
        self.duo.tick(101.9)
        self.assertEqual(self.out.generated_on(), [])
        self.duo.tick(102.0)
        self.assertEqual(self.out.generated_on(), [67])
        self.duo.tick(102.6)
        self.assertIn(([0x80, 67, 0], True), self.out.messages)

    def test_a_cut_drops_the_notes_that_have_not_sounded(self):
        for line in ("note 2.000 0.200 60\n", "note 2.500 0.200 62\n", "note 3.000 0.200 64\n"):
            self.duo.receive(line)
        self.duo.receive("cut 2.500\n")
        for now in (102.0, 102.5, 103.0):
            self.duo.tick(now)
        self.assertEqual(self.out.generated_on(), [60])

    def test_a_note_that_arrives_too_late_is_skipped(self):
        self.duo.receive("note 2.000 0.200 60\n")
        self.duo.tick(102.0 + main.LATE_S + 0.05)
        self.assertEqual(self.out.generated_on(), [])


if __name__ == '__main__':
    unittest.main()
