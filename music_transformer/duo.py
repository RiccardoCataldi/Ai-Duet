"""
A real-time duo with the Anticipatory Music Transformer: you play one instrument, the model improvises on another
at the same time. Your notes enter the context the moment you press them; the part of the model's line that has
not sounded yet is thrown away and sampled again, now hearing you.

Times on the wire are seconds since the connection opened; inside the context they are ticks since `base`.
"""

import math
import time

import torch
from anticipation.config import CONTEXT_SIZE, MAX_DUR, MAX_PITCH, MAX_TIME, TIME_RESOLUTION
from anticipation.vocab import AUTOREGRESS, DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET
from anticipation.sample import safe_logits, nucleus

from music_transformer.listener import STAY, Listener

LOOKAHEAD = 100   # ticks the model plays ahead of now (1 s)
MARGIN = 15       # ticks: a new note of the model sounds no earlier than this, to cover sampling and the network
MAX_REST = 200    # ticks: the longest silence the model may leave; it reads long silences as a sparse piece and stops
# left free, the model plays like a full Lakh MIDI track: dense chords and the same arpeggio over and over.
# These keep it a soloist.
MAX_CHORD = 2     # most notes the model may start at once
MIN_STEP = 20     # ticks between two of its onsets that are not a chord
NO_REPEAT = 3     # it may not play again a run of this many pitches it played in its last LINE notes
LINE = 32
# it listens to you more than to itself:
MEMORY = 200      # ticks of its own playing it remembers; all of yours within WINDOW stays
SPACE = 50        # ticks between its onsets while you are playing: it leaves you room
PLAYING = 100     # ticks after your last note that you still count as playing
# it listens like a partner (listener.py): your key, your chord, your pulse, your phrases.
CHORD_TONE = 1.0  # logit added to the tones of the chord you are playing
OUT_OF_KEY = 2.0  # logit taken from a pitch class outside your key; in between while you are modulating
UNDER = 1.0       # while you play it accompanies: logit taken from pitches at or above your lowest recent note
ANSWER = 400      # ticks after your phrase ends in which it answers it, following its intervals
ECHO = 3.0        # logit added to the pitch that continues your phrase's intervals
GRID = 2          # ticks off your pulse that its onsets may land (your beat while you play, your tatum otherwise)
WINDOW = 2000     # ticks of the duo the model keeps hearing when the context fills up or the clock runs out
KEEP = 600        # tokens kept at most in that case

# the knobs you can turn while you play (the piano roll shows them): default, lowest, highest, unit, what it does
KNOBS = {
    'min_step': (MIN_STEP, 5, 60, 'ticks', "time between its onsets when you rest"),
    'space': (SPACE, 10, 150, 'ticks', "time between its onsets while you play"),
    'out_of_key': (OUT_OF_KEY, 0.0, 6.0, '', "how strongly it keeps to your key"),
    'chord_tone': (CHORD_TONE, 0.0, 4.0, '', "how much it leans on the chord you play"),
    'under': (UNDER, 0.0, 4.0, '', "how firmly it stays below you while you play"),
    'answer': (ANSWER, 0, 1000, 'ticks', "how long after your phrase it answers it"),
    'echo': (ECHO, 0.0, 6.0, '', "how closely its answer follows your intervals"),
    'grid': (GRID, 0, 20, 'ticks', "how far off your pulse its onsets may land"),
    'stay': (STAY, 0.80, 0.995, '', "how sure it must be you modulated (lower = sooner)"),
    'no_repeat': (NO_REPEAT, 2, 8, 'notes', "the length of runs it may not repeat"),
    'memory': (MEMORY, 50, 2000, 'ticks', "how much of its own playing it remembers"),
    'temperature': (1.0, 0.5, 1.5, '', "how adventurous its choices are"),
    'top_p': (0.98, 0.80, 1.0, '', "how many unlikely notes it still considers"),
    'lowest': (0, 0, 8, 'octave_from', "the lowest octave it may play"),
    'highest': (8, 0, 8, 'octave_to', "the highest octave it may play"),
}
# what it is doing, and the knobs that shape its notes then; the others wait for their moment
IN_EFFECT = {
    'listening': set(),
    'accompanying': {'space', 'under'},     # you are playing
    'answering': {'min_step', 'answer', 'echo'},   # your phrase just ended
    'alone': {'min_step', 'answer'},        # you rest, and it has answered
}
ALWAYS = {'out_of_key', 'chord_tone', 'stay', 'no_repeat', 'memory', 'temperature', 'top_p', 'lowest', 'highest'}


def knob(name, value):
    """`value` for the knob `name`, kept within its range."""
    default, low, high, *_ = KNOBS[name]
    value = min(max(value, low), high)
    return round(value) if isinstance(default, int) else float(value)


class DuoSession:
    def __init__(self, model, device, you=0, me=4, clock=time.monotonic, started=None, **knobs):
        self.model, self.device = model, device
        self.you, self.me = you, me    # General MIDI programs: piano, electric piano
        self.listener = Listener()
        for name, (default, *_) in KNOBS.items():
            setattr(self, name, default)
        self.turn(**knobs)
        self.heard_at = None           # context time of your last note
        self.clock, self.started = clock, clock() if started is None else started
        self.base = None               # session tick of time 0 in the context: your first note
        self.seq = [AUTOREGRESS]       # complete events in time order, yours and the model's
        self.past, self.fed = None, 0  # the KV cache covers seq[:fed]

    def turn(self, **knobs):
        """Sets knobs (see KNOBS), kept within their range; they take effect from the model's next note."""
        for name, value in knobs.items():
            setattr(self, name, knob(name, value))
        if 'stay' in knobs:
            self.listener.stay(self.stay)

    def state(self):
        """What it hears and what it is doing: {'mode', 'key', 'chord', 'bpm', 'active': the knobs in effect}."""
        if self.base is None or self.heard_at is None:
            return {'mode': 'listening', 'key': None, 'chord': None, 'bpm': None, 'active': sorted(ALWAYS)}
        now = self.now()
        at = self._seconds(now)
        if now - self.heard_at < PLAYING:
            mode = 'accompanying'
        elif now - self.heard_at < self.answer and len(self.listener.phrase(at)) > 1:
            mode = 'answering'
        else:
            mode = 'alone'
        key, sure = self.listener.key()
        chord = self.listener.chord(at)
        pulse = self.listener.pulse(at)
        active = ALWAYS | IN_EFFECT[mode] | ({'grid'} if pulse else set())
        return {'mode': mode, 'key': self.listener.name(key) if sure > 0.3 else None,
                'chord': self.listener.chord_name(chord) if chord else None,
                'bpm': round(60 / self.listener.beat(pulse[0])) if pulse else None, 'active': sorted(active)}

    def octaves(self):
        """The lowest and highest pitch it may play, from C of its lowest octave to B of its highest (C4 = 60)."""
        low, high = sorted((self.lowest, self.highest))
        return 12 * (low + 1), min(12 * (high + 1) + 11, MAX_PITCH - 1)

    def knobs(self):
        return {name: getattr(self, name) for name in KNOBS}

    def now(self):
        return round((self.clock() - self.started) * TIME_RESOLUTION) - self.base

    def ahead(self):
        """Ticks of music already sampled beyond now; the caller samples more while this is under LOOKAHEAD."""
        return self._last_time() - self.now()

    def hear(self, onset, duration, pitch):
        """Your note, pressed at `onset`. Returns the time from which the model's unplayed notes are dropped, or None."""
        self.listener.hear(onset, duration, pitch)
        tick = round(onset * TIME_RESOLUTION)
        if self.base is None:
            self.base = tick
        cut = self._forget_after(self.now() + MARGIN)
        self.heard_at = max(tick - self.base, self._last_time(), 0)
        self.seq += [TIME_OFFSET + self.heard_at,
                     DUR_OFFSET + min(max(round(duration * TIME_RESOLUTION), 1), MAX_DUR - 1),
                     NOTE_OFFSET + self.you * MAX_PITCH + pitch]
        # otherwise its own line, two or three times denser than yours, drowns you out
        events = self._events()
        kept = [e for e in events if self._instrument(e[2]) != self.me or e[0] - TIME_OFFSET >= self.heard_at - self.memory]
        if len(kept) < len(events):
            self._restart(kept)
        return cut

    @torch.no_grad()
    def play(self):
        """Samples the model's next note: (onset, duration, pitch) in session seconds."""
        self._fit()
        now, last, line = self.now(), self._last_time(), self._my_line()
        you_play = self.heard_at is not None and now - self.heard_at < PLAYING
        step = self.space if you_play else self.min_step
        earliest = max(last, now + MARGIN)
        latest = min(max(earliest, last + MAX_REST), MAX_TIME - 1)
        if line:
            mine = line[-1][0]
            chord = [note for time_, note in line if time_ == mine]
            earliest = max(earliest, mine + (step if len(chord) >= MAX_CHORD else 0))
            latest = min(max(latest, mine + step), MAX_TIME - 1)
        for slot in range(3):
            logits = self._logits()
            mask = torch.full_like(logits, -float('inf'))
            if slot == 0:
                mask[TIME_OFFSET + earliest:TIME_OFFSET + latest + 1] = 0
                if line:
                    mask[TIME_OFFSET + mine + 1:TIME_OFFSET + mine + step] = -float('inf')
                mask = self._on_your_pulse(mask, earliest, latest, you_play)
            elif slot == 1:
                mask[DUR_OFFSET:DUR_OFFSET + MAX_DUR] = 0
            else:
                low, high = self.octaves()
                mask[NOTE_OFFSET + self.me * MAX_PITCH + low:NOTE_OFFSET + self.me * MAX_PITCH + high + 1] = 0
                for note in self._banned(line, self.seq[-2] - TIME_OFFSET):
                    mask[note] = -float('inf')
                if self.heard_at is not None:
                    mask[NOTE_OFFSET + self.me * MAX_PITCH:NOTE_OFFSET + (self.me + 1) * MAX_PITCH] += \
                        self._partner(self.seq[-2] - TIME_OFFSET, line)
            token = int(torch.multinomial(torch.softmax(nucleus(logits + mask, self.top_p), -1), 1))
            self.seq.append(token)
        time_, duration, note = self.seq[-3:]
        return ((time_ - TIME_OFFSET + self.base) / TIME_RESOLUTION,
                (duration - DUR_OFFSET) / TIME_RESOLUTION,
                (note - NOTE_OFFSET) % MAX_PITCH)

    def _seconds(self, tick):
        return (tick + self.base) / TIME_RESOLUTION

    def _on_your_pulse(self, mask, earliest, latest, you_play):
        """Keeps only the onsets on your beat (while you play) or your tatum; unchanged while your rhythm is unclear."""
        pulse = self.listener.pulse(self._seconds(self.now()))
        if pulse is None:
            return mask
        unit = self.listener.beat(pulse[0]) if you_play else pulse[0]
        first = math.ceil((self._seconds(earliest) - pulse[1]) / unit - self.grid / (unit * TIME_RESOLUTION))
        grid = torch.full_like(mask, -float('inf'))
        for k in range(first, first + math.ceil((latest - earliest) / (unit * TIME_RESOLUTION)) + 2):
            tick = round((pulse[1] + k * unit) * TIME_RESOLUTION) - self.base
            grid[TIME_OFFSET + max(tick - self.grid, earliest):TIME_OFFSET + min(tick + self.grid, latest) + 1] = 0
        on = mask + grid
        return on if torch.isfinite(on).any() else mask

    def _partner(self, tick, line):
        """Logits added to its pitches at `tick`: your chord and key; under you while you play; your phrase after it."""
        at, you_play = self._seconds(tick), tick - self.heard_at < PLAYING
        in_key, chord = self.listener.in_key(), self.listener.chord(at) or set()
        bias = torch.tensor([self.chord_tone * (p % 12 in chord) - self.out_of_key * (1 - in_key[p % 12])
                             for p in range(MAX_PITCH)], dtype=torch.float)
        if you_play:
            recent = at - PLAYING / TIME_RESOLUTION - 1
            low = min((n[2] for n in self.listener.notes if n[0] >= recent), default=MAX_PITCH)
            bias[low:] -= self.under
        elif line and tick - self.heard_at < self.answer:
            phrase = self.listener.phrase(at)
            answered = sum(time_ > self.heard_at for time_, _ in line)
            if 0 < answered < len(phrase):
                echo = (line[-1][1] - NOTE_OFFSET) % MAX_PITCH + phrase[answered] - phrase[answered - 1]
                if in_key[echo % 12] < 0.5:   # your interval, bent into your key as a sequence would be
                    echo += 1 if in_key[(echo + 1) % 12] >= in_key[(echo - 1) % 12] else -1
                if 0 <= echo < MAX_PITCH:
                    bias[echo] += self.echo
        return bias.to(self.device)

    def _logits(self):
        inp = torch.tensor([self.seq[self.fed:]], device=self.device)
        out = self.model(inp, past_key_values=self.past, use_cache=True)
        self.past, self.fed = out.past_key_values, len(self.seq)
        return safe_logits(out.logits[0, -1].float() / self.temperature, len(self.seq) - 1)

    def _last_time(self):
        return self.seq[-3] - TIME_OFFSET if len(self.seq) > 1 else 0

    def _my_line(self):
        """(time, note token) of the model's last LINE notes, oldest first."""
        line, complete = [], len(self.seq) - (len(self.seq) - 1) % 3   # skip an event still being sampled
        for i in range(complete - 3, 0, -3):
            if self._instrument(self.seq[i + 2]) == self.me:
                line.append((self.seq[i] - TIME_OFFSET, self.seq[i + 2]))
                if len(line) == LINE:
                    break
        return line[::-1]

    def _banned(self, line, tick):
        """Notes the model may not play at `tick`: one it is already playing there, or one that repeats a run."""
        banned = {note for time_, note in line if time_ == tick}   # Lakh MIDI is full of doubled notes
        notes = [note for _, note in line]
        run = notes[len(notes) - self.no_repeat + 1:]
        for i in range(len(notes) - self.no_repeat + 1):
            if notes[i:i + self.no_repeat - 1] == run:
                banned.add(notes[i + self.no_repeat - 1])
        return banned

    def _forget_after(self, tick):
        """Drops the model's trailing notes that start after `tick`; they have not sounded yet."""
        cut = None
        while len(self.seq) > 1 and self._instrument(self.seq[-1]) == self.me and self._last_time() > tick:
            cut = self._last_time()
            del self.seq[-3:]
        if self.fed > len(self.seq):
            self.past.crop(len(self.seq))
            self.fed = len(self.seq)
        return None if cut is None else (cut + self.base) / TIME_RESOLUTION

    def _fit(self):
        """Keeps the context under CONTEXT_SIZE tokens and its times under MAX_TIME."""
        now = self.now()
        if len(self.seq) + 3 <= CONTEXT_SIZE and now + LOOKAHEAD + MAX_REST < MAX_TIME:
            return
        self._restart([e for e in self._events() if e[0] - TIME_OFFSET >= now - WINDOW][-(KEEP // 3):], now)

    def _events(self):
        """Complete events in the context, leaving out one still being sampled."""
        return [self.seq[i:i + 3] for i in range(1, len(self.seq) - 2, 3)]

    def _restart(self, events, now=None):
        """A new context from `events`, moved to start at time 0 as the model was trained; the cache is rebuilt."""
        shift = events[0][0] - TIME_OFFSET if events else (now if now is not None else self.now())
        self.base += shift
        if self.heard_at is not None:
            self.heard_at -= shift
        self.seq = [AUTOREGRESS] + [token for t, d, n in events for token in (t - shift, d, n)]
        self.past, self.fed = None, 0

    @staticmethod
    def _instrument(note):
        return (note - NOTE_OFFSET) // MAX_PITCH
