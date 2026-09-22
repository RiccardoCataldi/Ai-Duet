import os
import threading

GESTURE_QUESTION = {
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
}

_OFFSETS = {
    "min3_up": 3,
    "maj3_up": 4,
    "p5_up": 7,
    "oct_up": 12,
    "oct_down": -12,
}


def _by_pitch(notes):
    return sorted(notes, key=lambda n: n["pitch"])


def _sounding(notes):
    return [{"pitch": n["pitch"], "velocity": n["velocity"]} for n in _by_pitch(notes)]


def _events(sounding, new_notes):
    events = [
        {"kind": "note_off", "channel": 1, "pitch": n["pitch"], "velocity": 0}
        for n in sounding
    ]
    events += [
        {"kind": "note_on", "channel": 1, "pitch": n["pitch"], "velocity": n["velocity"]}
        for n in new_notes
    ]
    return events


def tick(picture, predict):
    held = _by_pitch(n for n in picture if n["source"] == "human" and not n["released"])
    released = _by_pitch(n for n in picture if n["source"] == "human" and n["released"])
    sounding_notes = _by_pitch(n for n in picture if n["source"] == "ai" and not n["released"])
    sounding = _sounding(sounding_notes)
    if held:
        reference_set = held
    elif released:
        reference_set = released
    elif sounding_notes:
        reference_set = sounding_notes
    else:
        return [], sounding, None
    state = {
        "reference": max(n["pitch"] for n in reference_set),
        "notes": held + released + sounding_notes,
    }
    try:
        gesture, probability = predict(state, GESTURE_QUESTION)
    except Exception:
        return [], sounding, state
    if gesture == "wait" or probability < 0.5:
        return [], sounding, state
    top = max(reference_set, key=lambda n: n["pitch"])
    if gesture == "double":
        new_notes = _sounding(reference_set)
    elif gesture in _OFFSETS:
        pitch = top["pitch"] + _OFFSETS[gesture]
        if not 0 <= pitch <= 127:
            return [], sounding, state
        new_notes = [{"pitch": pitch, "velocity": top["velocity"]}]
    else:
        return [], sounding, state
    return _events(sounding, new_notes), new_notes, state


def _age_ms(now, attacked_at):
    return int(round((now - attacked_at) * 1000))


class Collector:
    def __init__(self):
        self._held = {}
        self._released = {}
        self._lock = threading.Lock()

    def note_on(self, pitch, velocity, now):
        with self._lock:
            self._released.pop(pitch, None)
            self._held[pitch] = (velocity, now)

    def note_off(self, pitch):
        with self._lock:
            held = self._held.pop(pitch, None)
            if held is None:
                return
            self._released[pitch] = held

    def take(self, sounding, now):
        with self._lock:
            picture = [
                {
                    "pitch": pitch,
                    "velocity": velocity,
                    "source": "human",
                    "released": False,
                    "age": _age_ms(now, attacked_at),
                }
                for pitch, (velocity, attacked_at) in self._held.items()
            ]
            picture += [
                {
                    "pitch": pitch,
                    "velocity": velocity,
                    "source": "human",
                    "released": True,
                    "age": _age_ms(now, attacked_at),
                }
                for pitch, (velocity, attacked_at) in self._released.items()
            ]
            self._released.clear()
        picture += [
            {
                "pitch": note["pitch"],
                "velocity": note["velocity"],
                "source": "ai",
                "released": False,
                "age": _age_ms(now, note["attacked_at"]),
            }
            for note in sounding
        ]
        return picture


class LayaPort:
    def __init__(self):
        self._agent = None
        try:
            os.environ["USE_TF"] = "0"
            import laya

            self._agent = laya.load("convaiinnovations/laya")
        except Exception as exc:
            print(f"Error loading accompaniment: {exc}")

    def predict(self, state, question):
        if self._agent is None:
            raise RuntimeError("accompaniment model not loaded")
        result = self._agent.predict(state, question)
        answer = result["answers"]["gesture"]
        choice = answer["choice"]
        return choice, answer["probabilities"][choice]
