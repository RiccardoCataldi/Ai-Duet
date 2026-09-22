import unittest

from accompaniment import tick


def note(pitch, velocity, source, released, age):
    return {
        "pitch": pitch,
        "velocity": velocity,
        "source": source,
        "released": released,
        "age": age,
    }


class TickTest(unittest.TestCase):
    def test_empty_picture_does_not_call_predict(self):
        def predict(state, question):
            raise AssertionError("predict should not be called")

        events, sounding, state = tick([], predict)

        self.assertEqual(events, [])
        self.assertEqual(sounding, [])
        self.assertIsNone(state)

    def test_held_notes_are_the_reference_over_just_released(self):
        seen = {}

        def predict(state, question):
            seen["state"] = state
            return "wait", 1.0

        picture = [
            note(67, 40, "human", True, 30),
            note(60, 80, "human", False, 100),
            note(64, 90, "human", False, 100),
        ]
        tick(picture, predict)

        self.assertEqual(seen["state"]["reference"], 64)

    def test_just_released_notes_are_the_reference_when_nothing_is_held(self):
        seen = {}

        def predict(state, question):
            seen["state"] = state
            return "wait", 1.0

        picture = [
            note(62, 50, "human", True, 15),
            note(58, 40, "human", True, 15),
            note(70, 90, "ai", False, 200),
        ]
        tick(picture, predict)

        self.assertEqual(seen["state"]["reference"], 62)

    def test_accompaniment_notes_are_the_reference_when_the_player_has_nothing(self):
        seen = {}

        def predict(state, question):
            seen["state"] = state
            return "wait", 1.0

        picture = [
            note(48, 60, "ai", False, 200),
            note(55, 40, "ai", False, 180),
        ]
        tick(picture, predict)

        self.assertEqual(seen["state"]["reference"], 55)

    def test_wait_emits_nothing_and_keeps_sounding_notes(self):
        def predict(state, question):
            return "wait", 0.9

        picture = [
            note(60, 80, "human", False, 50),
            note(72, 70, "ai", False, 400),
        ]
        events, sounding, state = tick(picture, predict)

        self.assertEqual(events, [])
        self.assertEqual(sounding, [{"pitch": 72, "velocity": 70}])
        self.assertEqual(state["reference"], 60)

    def test_minor_third_above_uses_the_highest_reference_note(self):
        def predict(state, question):
            return "min3_up", 0.8

        picture = [
            note(60, 100, "human", False, 10),
            note(67, 40, "human", False, 10),
        ]
        events, sounding, _state = tick(picture, predict)

        self.assertEqual(
            events,
            [{"kind": "note_on", "channel": 1, "pitch": 70, "velocity": 40}],
        )
        self.assertEqual(sounding, [{"pitch": 70, "velocity": 40}])

    def test_probability_below_one_half_emits_nothing(self):
        def predict(state, question):
            return "min3_up", 0.49

        picture = [
            note(60, 80, "human", False, 10),
            note(72, 70, "ai", False, 400),
        ]
        events, sounding, _state = tick(picture, predict)

        self.assertEqual(events, [])
        self.assertEqual(sounding, [{"pitch": 72, "velocity": 70}])

    def test_probability_of_one_half_plays(self):
        def predict(state, question):
            return "p5_up", 0.5

        events, sounding, _state = tick([note(60, 64, "human", False, 5)], predict)

        self.assertEqual(
            events,
            [{"kind": "note_on", "channel": 1, "pitch": 67, "velocity": 64}],
        )
        self.assertEqual(sounding, [{"pitch": 67, "velocity": 64}])

    def test_pitch_outside_midi_range_emits_nothing(self):
        def predict(state, question):
            return "oct_up", 0.9

        picture = [
            note(120, 50, "human", False, 5),
            note(48, 30, "ai", False, 100),
        ]
        events, sounding, _state = tick(picture, predict)

        self.assertEqual(events, [])
        self.assertEqual(sounding, [{"pitch": 48, "velocity": 30}])

        def predict_down(state, question):
            return "oct_down", 0.9

        events, sounding, _state = tick([note(4, 50, "human", False, 5)], predict_down)
        self.assertEqual(events, [])
        self.assertEqual(sounding, [])

    def test_predict_exception_emits_nothing_and_keeps_sounding_notes(self):
        def predict(state, question):
            raise RuntimeError("model failed")

        picture = [
            note(60, 80, "human", False, 10),
            note(72, 70, "ai", False, 400),
        ]
        events, sounding, state = tick(picture, predict)

        self.assertEqual(events, [])
        self.assertEqual(sounding, [{"pitch": 72, "velocity": 70}])
        self.assertEqual(state["reference"], 60)

    def test_state_notes_are_ordered_held_then_released_then_accompaniment(self):
        seen = {}

        def predict(state, question):
            seen["state"] = state
            seen["question"] = question
            return "wait", 1.0

        picture = [
            note(72, 10, "ai", False, 400),
            note(67, 30, "human", True, 50),
            note(64, 80, "human", False, 100),
            note(60, 40, "human", False, 200),
            note(62, 20, "human", True, 15),
        ]
        tick(picture, predict)

        self.assertEqual(
            seen["state"],
            {
                "reference": 64,
                "notes": [
                    note(60, 40, "human", False, 200),
                    note(64, 80, "human", False, 100),
                    note(62, 20, "human", True, 15),
                    note(67, 30, "human", True, 50),
                    note(72, 10, "ai", False, 400),
                ],
            },
        )
        self.assertEqual(
            seen["question"],
            {
                "gesture": {
                    "type": "choice",
                    "instructions": (
                        "Choose the accompaniment gesture for this instant. "
                        "Prefer wait when the accompaniment already sounding still fits."
                    ),
                    "criteria": {
                        "wait": "add no new accompaniment note",
                        "double": "restate the reference notes",
                        "min3_up": "one note a minor third above the reference",
                        "maj3_up": "one note a major third above the reference",
                        "p5_up": "one note a perfect fifth above the reference",
                        "oct_up": "one note an octave above the reference",
                        "oct_down": "one note an octave below the reference",
                    },
                }
            },
        )

    def test_interval_gestures_add_one_note_from_the_highest_reference(self):
        cases = [
            ("maj3_up", 64),
            ("p5_up", 67),
            ("oct_up", 72),
            ("oct_down", 48),
        ]
        for gesture, pitch in cases:
            with self.subTest(gesture=gesture):
                def predict(state, question, gesture=gesture):
                    return gesture, 0.9

                events, sounding, _state = tick([note(60, 88, "human", False, 1)], predict)
                self.assertEqual(
                    events,
                    [{"kind": "note_on", "channel": 1, "pitch": pitch, "velocity": 88}],
                )
                self.assertEqual(sounding, [{"pitch": pitch, "velocity": 88}])

    def test_double_copies_every_reference_note_with_its_own_velocity(self):
        def predict(state, question):
            return "double", 0.9

        picture = [
            note(64, 70, "human", False, 10),
            note(60, 40, "human", False, 10),
            note(67, 90, "human", True, 5),
            note(72, 20, "ai", False, 300),
        ]
        events, sounding, _state = tick(picture, predict)

        self.assertEqual(
            events,
            [
                {"kind": "note_off", "channel": 1, "pitch": 72, "velocity": 0},
                {"kind": "note_on", "channel": 1, "pitch": 60, "velocity": 40},
                {"kind": "note_on", "channel": 1, "pitch": 64, "velocity": 70},
            ],
        )
        self.assertEqual(
            sounding,
            [{"pitch": 60, "velocity": 40}, {"pitch": 64, "velocity": 70}],
        )

    def test_same_pitches_are_restruck_and_a_new_gesture_replaces_sounding_notes(self):
        def predict(state, question):
            return "oct_up", 1.0

        picture = [
            note(60, 80, "human", False, 10),
            note(72, 80, "ai", False, 250),
            note(48, 10, "ai", False, 250),
        ]
        events, sounding, _state = tick(picture, predict)

        self.assertEqual(
            events,
            [
                {"kind": "note_off", "channel": 1, "pitch": 48, "velocity": 0},
                {"kind": "note_off", "channel": 1, "pitch": 72, "velocity": 0},
                {"kind": "note_on", "channel": 1, "pitch": 72, "velocity": 80},
            ],
        )
        self.assertEqual(sounding, [{"pitch": 72, "velocity": 80}])


if __name__ == "__main__":
    unittest.main()
