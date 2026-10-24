import os
import tempfile
import unittest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import knobs


class MindTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'mind.json')
        self.mind = knobs.Mind(None, self.path)   # no server: we give it the knobs ourselves
        self.mind.sock = DummySocket()
        self.sent = []

        def send(knob):
            self.sent.append(knob)
            self.mind.send(knob)

        for name, (low, high, default) in {'min_step': (5, 60, 20), 'space': (10, 150, 50),
                                           'apart': (0.0, 6.0, 2.0), 'lead': (0.0, 6.0, 3.0),
                                           'echo': (0.0, 6.0, 3.0)}.items():
            self.mind.knobs[name] = knobs.Knob(name, default, low, high, default, '', '', isinstance(low, int),
                                               send)

    def tearDown(self):
        self.tmp.cleanup()

    def macro(self, label):
        return next(m for m in self.mind.macros if m.name == label)

    def test_a_musical_control_turns_its_knobs_along_their_line_and_its_middle_is_the_default(self):
        self.assertEqual(knobs.along((60, 20, 8), 0.5), 20)
        self.assertEqual(knobs.along((60, 20, 8), 0.0), 60)
        self.assertEqual(knobs.along((60, 20, 8), 0.75), 14)
        self.macro('density').set(1.0)
        self.assertEqual((self.mind.knobs['min_step'].value, self.mind.knobs['space'].value), (8, 15))
        self.assertEqual(self.macro('density').shown(), 'dense')
        self.assertEqual({k.name for k in self.sent}, {'min_step', 'space'})

    def test_a_knob_is_dimmed_while_what_it_shapes_is_not_happening(self):
        self.mind.state = {'mode': 'accompanying', 'key': 'C major', 'chord': 'G7', 'bpm': 92,
                           'active': ['out_of_key', 'space', 'apart']}
        self.assertTrue(self.mind.active(self.macro('role')))
        self.assertFalse(self.mind.active(self.macro('dialogue')))
        self.assertTrue(self.mind.active(self.mind.knobs['space']))
        self.assertFalse(self.mind.active(self.mind.knobs['echo']))
        self.assertEqual(self.mind.status(), 'C major · G7 · 92 bpm · accompanying you')

    def test_what_you_turn_is_kept_for_next_time(self):
        self.macro('role').set(0.0)
        again = knobs.Mind(None, self.path)
        self.assertEqual((again.saved['knobs']['apart'], again.saved['knobs']['lead']), (4.0, 0.0))
        self.assertEqual(next(m for m in again.macros if m.name == 'role').value, 0.0)

    def test_a_knob_you_never_turned_starts_where_its_control_stands(self):
        self.macro('role').set(1.0)                 # leads
        again = knobs.Mind(None, self.path)
        again.sock = DummySocket()
        again.saved['knobs'] = {'space': 30}        # as if only space had ever been turned
        served = {name: knobs.Knob(name, default, 0, 150, default, '', '', False, again.send)
                  for name, default in (('space', 50.0), ('apart', 2.0), ('lead', 3.0))}
        again._restore(served)
        self.assertEqual({name: knob.value for name, knob in served.items()}, {'space': 30, 'apart': 0.0, 'lead': 6.0})
        self.assertEqual(again.saved['knobs'], {'space': 30})   # it goes on following its control


class DummySocket:
    def sendall(self, data):
        pass


if __name__ == '__main__':
    unittest.main()
