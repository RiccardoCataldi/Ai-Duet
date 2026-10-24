"""
Knobs on screen next to the piano roll, in four tabs:
- AI mind: presets, ways to play together (it comps while you solo, it solos while you comp...), then a few musical controls (density, harmony, role...), each turning some of the model's knobs on the Jetson
  (see music_transformer/server.py, "knobs"), the octaves it may play, the scale it keeps to (yours, or one you force
  it into) and its tempo (yours, or one you fix, with a metronome)
- Your sound, AI sound: yours and the model's in FluidSynth on this laptop (sound.py)
- Advanced: every knob of the model on its own
A knob is dimmed while what it shapes is not happening (the server says what the model is doing). Drag a knob up or
down (Shift: finely) or turn the wheel over it; a double click puts it back.
"""
import json
import math
import os
import socket
import threading
import time

import pygame
import pygame.gfxdraw

import sound

PANEL_W = 360
COLUMNS = 3
DRAG = 220                # pixels of drag across a knob's whole range
STEPS = 50                # wheel steps across it
DOUBLE_CLICK_S = 0.35
SWEEP = 270               # degrees a knob turns, its gap at the bottom
MIND_FILE = os.path.expanduser('~/.config/ai-duet/mind.json')   # the model's knobs as last turned

# the musical controls: label, its two ends, what it does, and the model's knobs it turns, each at (0, middle, 1);
# the middle is the duo's default
MACROS = [
    ('density', ('sparse', 'dense'), "how many notes it plays",
     {'min_step': (60, 20, 8), 'space': (150, 50, 15), 'max_rest': (500, 200, 100)}),
    ('harmony', ('inside', 'outside'), "how far it strays from your key, and how it colours the chord you play",
     {'out_of_key': (5.0, 2.0, 0.0), 'chord_tone': (3.0, 1.0, 0.0), 'tension': (0.0, 1.0, 3.0)}),
    ('role', ('accompanies', 'leads'), "while you play: it keeps apart from your hands and under your melody, or goes "
                                       "anywhere; when you rest: it comps on, or takes the melody, up where yours was",
     {'apart': (4.0, 2.0, 0.0), 'lead': (0.0, 3.0, 6.0)}),
    ('dialogue', ('ignores you', 'answers you'), "when your phrase ends: how soon and how closely it answers it",
     {'answer': (0, 400, 1000), 'echo': (0.0, 3.0, 6.0)}),
    ('groove', ('loose', 'tight'), "how closely its notes sit on your pulse",
     {'grid': (20, 2, 0)}),
    ('risk', ('safe', 'adventurous'), "how surprising its choices are",
     {'temperature': (0.7, 1.0, 1.4), 'top_p': (0.85, 0.98, 1.0)}),
    ('focus', ('on you', 'on itself'), "whose playing it builds on: yours, or its own line too",
     {'memory': (50, 200, 2000)}),
    ('key follow', ('steady', 'quick'), "how soon it follows you into a new key",
     {'stay': (0.995, 0.97, 0.85)}),
]
# ways to play together: name, what it does, its musical controls (0 to 1; one left out stands in the middle) and the
# model's knobs it sets (one any other preset sets and it leaves out goes back to its default)
PRESETS = [
    ('duet', "the duo's defaults: it comps while you play, answers your phrases and plays on when you rest", {}, {}),
    ('call & response', "it keeps quiet while you play, then answers each phrase soon, close to your intervals, and "
                        "takes the melody",
     {'dialogue': 1.0, 'role': 0.75, 'density': 0.6, 'focus': 0.2}, {'space': 150, 'chord': 1, 'hands': 1}),
    ('it comps', "it plays the harmony, low and steady with both hands, and keeps comping when you rest: you improvise",
     {'role': 0.0, 'dialogue': 0.0, 'density': 0.4, 'harmony': 0.4, 'groove': 1.0, 'risk': 0.3, 'key follow': 0.3},
     {'chord': 4, 'solo_chord': 4, 'hands': 2, 'lowest': 2, 'highest': 4}),
    ('it solos', "you play the harmony, it improvises a line over it, up high",
     {'role': 0.85, 'dialogue': 0.3, 'density': 0.8, 'harmony': 0.6, 'groove': 0.7, 'risk': 0.75, 'focus': 0.8},
     {'chord': 1, 'solo_chord': 1, 'hands': 1, 'lowest': 5, 'highest': 7}),
    ('free', "outside your key and off your pulse, surprising and dense: free improvisation together",
     {'harmony': 1.0, 'risk': 1.0, 'groove': 0.0, 'density': 0.75, 'focus': 0.8, 'key follow': 1.0}, {}),
]
PRESET_KNOBS = {name for *_, turns in PRESETS for name in turns}
MODES = {'listening': 'waiting for you to play', 'accompanying': 'accompanying you', 'answering': 'answering you',
         'alone': 'playing on its own'}

PANEL = (17, 20, 31)
EDGE = (44, 49, 68)
TRACK = (46, 52, 72)
BODY = (30, 34, 48)
YOU = (255, 176, 84)      # the colours of the piano roll
AI = (94, 214, 230)
TEXT = (210, 214, 226)
DIM = (120, 126, 146)
FAINT = (70, 76, 96)


class Knob:
    """`turned(knob)` is called whenever what it shows changes (a musical control: whenever it moves)."""

    def __init__(self, name, value, low, high, default, unit, what, whole, turned, ends=None):
        self.name, self.unit, self.what, self.whole, self.turned, self.ends = name, unit, what, whole, turned, ends
        self.value, self.low, self.high, self.default = float(value), float(low), float(high), float(default)

    def set(self, value):
        before = self.shown()
        self.value = min(max(value, self.low), self.high)
        if self.shown() != before or self.unit == 'macro':
            self.turned(self)

    def step(self):
        """How far one turn of the wheel moves it."""
        if self.unit == 'program':
            return 1
        span = (self.high - self.low) / STEPS
        return max(1, round(span)) if self.whole else span

    def shown(self):
        value = round(self.value) if self.whole else self.value
        if self.unit == 'macro':
            return self.ends[0] if value < 0.4 else self.ends[1] if value > 0.6 else 'balanced'
        if self.unit == 'ticks':
            return f"{value / 100:.2f} s"
        if self.unit == 'notes':
            return f"{value} notes"
        if self.unit == 'program':
            return sound.GM[value]
        if self.unit == 'bpm':
            return f"{value} bpm" if value else 'yours'
        if self.unit.startswith('choice:'):
            return self.unit[7:].split('/')[value].replace('_', ' ')
        if self.unit in ('octave_from', 'octave_to'):
            return f"{'C' if self.unit == 'octave_from' else 'B'}{value}"
        if self.unit == 'pan':
            side = round((value - 64) / 63 * 100)
            return 'centre' if side == 0 else f"{'L' if side < 0 else 'R'} {abs(side)}%"
        if self.unit == 'hands':
            return 'one hand' if value == 1 else 'two hands'
        if self.unit == 'ms':
            return f"{value} ms"
        if self.unit == 'times':
            return f"x{2 ** (value / 1200):.2f}"
        if self.unit == 'level':   # a MIDI level out of 127, or a share
            return f"{value / 127:.0%}" if self.whole else f"{value:.0%}"
        if self.unit == 'amount':
            return f"{(value - self.low) / (self.high - self.low):.0%}"
        return f"{value:.3f}" if self.high - self.low < 0.5 else f"{value:.2f}"

    def label(self):
        return self.name.replace('_', ' ')

    def share(self):
        return (self.value - self.low) / (self.high - self.low)


def along(anchors, share):
    """The value at `share` (0 to 1) of a line through anchors at 0, 0.5 and 1."""
    a, b, c = anchors
    return a + (b - a) * share * 2 if share <= 0.5 else b + (c - b) * (share - 0.5) * 2


class Mind:
    """The model's knobs over the knobs connection to the AMT server, and what it says it is doing.
    They are kept in MIND_FILE and given back to the server each time it connects; it reconnects whenever lost
    (with no `address`, it does not connect: its knobs are given to it)."""

    def __init__(self, address, path=MIND_FILE):
        self.address, self.path = address, path
        self.knobs = {}           # name -> Knob, once connected
        self.state = None         # the last DuoSession.state() heard
        self.sock = None
        self.lock = threading.Lock()
        try:
            with open(path) as f:
                self.saved = json.load(f)
        except (OSError, ValueError):
            self.saved = {}
        self.saved.setdefault('knobs', {})
        self.saved.setdefault('macros', {})
        self.macros = [Knob(label, self.saved['macros'].get(label, 0.5), 0, 1, 0.5, 'macro', what, False,
                            self._macro, ends) for label, ends, what, _ in MACROS]
        self.turns = {label: turns for label, _, _, turns in MACROS}
        if address:
            threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        host, port = self.address.rsplit(':', 1)
        while True:
            try:
                sock = socket.create_connection((host, int(port)), timeout=3)
                sock.sendall(b"knobs\n")
                lines = sock.makefile('r')
                knobs = {}
                for line in lines:
                    if line.strip() == 'ready':
                        break
                    _, name, value, low, high, default, unit, what = line.split(maxsplit=7)
                    knobs[name] = Knob(name, value, low, high, default, unit.strip('-'), what.strip(),
                                       '.' not in default, self.send)
                sock.settimeout(None)
                with self.lock:
                    self.knobs, self.sock = knobs, sock
                self._restore(knobs)
                for line in lines:
                    if line.startswith('state '):
                        self.state = json.loads(line[6:])
            except (OSError, ValueError):
                pass
            with self.lock:
                self.knobs, self.sock, self.state = {}, None, None
            time.sleep(2)

    def _restore(self, knobs):
        """Gives the server back its knobs as you left them; one you never turned starts where its control stands."""
        saved = self.saved['knobs']
        for name, knob in knobs.items():
            if name in saved:
                knob.value = saved[name]
                self.send(knob, keep=False)
        for macro in self.macros:
            for name, anchors in self.turns[macro.name].items():
                if name in knobs and name not in saved:   # not kept: it follows its control until you turn it
                    knobs[name].value = along(anchors, macro.value)
                    self.send(knobs[name], keep=False)

    def send(self, knob, keep=True):
        """Turns `knob` on the server and, unless `keep` is False, keeps it for next time."""
        with self.lock:
            sock = self.sock
        if sock is None:
            return
        try:
            sock.sendall(f"set {knob.name} {knob.value:.4f}\n".encode())
        except OSError:
            return
        if keep:
            self.saved['knobs'][knob.name] = knob.value
            self._save()

    def _macro(self, macro):
        for name, anchors in self.turns[macro.name].items():
            if name in self.knobs:
                self.knobs[name].set(along(anchors, macro.value))
        self.saved['macros'][macro.name] = macro.value
        self._save()

    def presets(self):
        """(name, what, its musical controls, its knobs) of each preset, every control and knob it sets filled in: a
        knob it leaves out stands where its control puts it, or at its default."""
        presets = []
        for name, what, macros, turns in PRESETS:
            macros = {m.name: macros.get(m.name, 0.5) for m in self.macros}
            free = {n: along(anchors, macros[m]) for m, knobs in self.turns.items() for n, anchors in knobs.items()}
            presets.append((name, what, macros, {n: turns.get(n, free.get(n, self.knobs[n].default))
                                                 for n in PRESET_KNOBS if n in self.knobs}))
        return presets

    def play(self, preset):
        _, _, macros, turns = preset
        for macro in self.macros:
            macro.set(macros[macro.name])
        for name, value in turns.items():
            self.knobs[name].set(value)

    def playing(self, preset):
        """Whether every control and knob stands where `preset` sets them."""
        _, _, macros, turns = preset
        return (all(abs(m.value - macros[m.name]) < 0.01 for m in self.macros) and
                all(abs(self.knobs[n].value - v) < 0.01 for n, v in turns.items()))

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.saved, f, indent=1)

    def active(self, knob):
        """Whether `knob` shapes the model's notes now."""
        if self.state is None:
            return True
        names = self.turns[knob.name] if knob.unit == 'macro' else {knob.name}
        return bool(set(names) & set(self.state['active']))

    def status(self):
        """What it hears and does, e.g. 'C major · G7 · 92 bpm · accompanying you'."""
        if self.state is None:
            return None
        heard = [self.state[k] for k in ('key', 'chord') if self.state[k]]
        if self.state['bpm']:
            heard.append(f"{self.state['bpm']} bpm")
        return ' · '.join(heard + [MODES[self.state['mode']]])


def sound_knobs(engine, player):
    """The knobs of `player`'s sound in `engine`."""
    knobs = []
    for name, (*defaults, low, high, unit, what) in sound.SOUND.items():
        default = defaults[player]
        knobs.append(Knob(name, engine.sound[player][name], low, high, default, unit, what, isinstance(default, int),
                          lambda knob: engine.set(player, knob.name, knob.value)))
    return knobs


class Panel:
    def __init__(self, address, engine, mind_file=MIND_FILE):
        self.mind = Mind(address, mind_file)
        waiting = f"waiting for the model at {address}"
        lit = lambda: 'lit knobs shape what it plays now' if self.mind.knobs else waiting
        register = lambda: [self.mind.knobs[n] for n in ('lowest', 'highest', 'root', 'scale', 'bpm', 'metronome')
                              if n in self.mind.knobs]
        yours, its = sound_knobs(engine, sound.YOU), sound_knobs(engine, sound.AI)
        # title, colour, knobs, status line, whether its knobs wait for their moment
        self.tabs = [('AI mind', AI, lambda: (self.mind.macros if self.mind.knobs else []) + register(), lit, True),
                     ('Your sound', YOU, lambda: yours, lambda: 'in FluidSynth, on this laptop', False),
                     ('AI sound', AI, lambda: its, lambda: 'in FluidSynth, on this laptop', False),
                     ('Advanced', AI, lambda: list(self.mind.knobs.values()), lit, True)]
        self.tab = 0
        self.font = pygame.font.Font(None, 24)
        self.tab_font = pygame.font.Font(None, 22)
        self.small = pygame.font.Font(None, 19)
        self.held = None          # (knob, y where the drag started, value then)
        self.hover = None
        self.last_click = (None, 0)
        self.places = {}          # knob name -> (centre, radius), from the last draw
        self.tab_places = []      # rects of the tabs, from the last draw
        self.preset_places = []   # (preset, rect) of the presets, from the last draw
        self.preset_hover = None

    def knobs(self):
        return self.tabs[self.tab][2]()

    def handle(self, event, origin):
        """Mouse events, at `origin` the panel's top left corner on screen."""
        if event.type == pygame.MOUSEWHEEL:
            knob = self._at(pygame.mouse.get_pos(), origin)
            if knob:
                knob.set(knob.value + event.y * knob.step())
            return
        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            return
        pos = event.pos
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, rect in enumerate(self.tab_places):
                if rect.collidepoint(pos[0] - origin[0], pos[1] - origin[1]):
                    self.tab, self.hover, self.held = i, None, None
                    return
            preset = self._preset_at(pos, origin)
            if preset:
                self.mind.play(preset)
                return
            knob = self._at(pos, origin)
            if knob is None:
                return
            name, at = self.last_click
            if name == knob.name and time.monotonic() - at < DOUBLE_CLICK_S:
                knob.set(knob.default)
                self.last_click = (None, 0)
                return
            self.last_click = (knob.name, time.monotonic())
            self.held = (knob, pos[1], knob.value)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.held = None
        elif event.type == pygame.MOUSEMOTION:
            self.hover = self._at(pos, origin)
            self.preset_hover = self._preset_at(pos, origin)
            if self.held:
                knob, y, start = self.held
                fine = 5 if pygame.key.get_mods() & pygame.KMOD_SHIFT else 1
                knob.set(start + (y - pos[1]) / (DRAG * fine) * (knob.high - knob.low))

    def _preset_at(self, pos, origin):
        return next((preset for preset, rect in self.preset_places
                     if rect.collidepoint(pos[0] - origin[0], pos[1] - origin[1])), None)

    def _at(self, pos, origin):
        for knob in self.knobs():
            place = self.places.get(knob.name)
            if place:
                (x, y), r = place
                if math.hypot(pos[0] - origin[0] - x, pos[1] - origin[1] - y) <= r + 14:
                    return knob
        return None

    def draw(self, surface):
        width, height = surface.get_size()
        surface.fill(PANEL)
        pygame.draw.line(surface, EDGE, (0, 0), (0, height))
        _, accent, _, status, waits = self.tabs[self.tab]
        self._tabs(surface)
        surface.blit(self.small.render(status(), True, DIM), (24, 58))
        self.preset_places = []
        top = self._presets(surface, accent, width) + 46 if self.tab == 0 and self.mind.knobs else 116
        knobs = self.knobs()
        cell_h = 104
        rows = max(1, (height - top - 84) // cell_h + 1)   # the rows that fit above the caption
        columns = max(COLUMNS, math.ceil(len(knobs) / rows))
        cell_w = (width - 24) / columns
        radius = 24 if columns == COLUMNS else 20
        self.places = {}
        for i, knob in enumerate(knobs):
            col, row = i % columns, i // columns
            centre = (round(12 + cell_w * (col + 0.5)), round(top + row * cell_h))
            self.places[knob.name] = (centre, radius)
            lit = knob is self.hover or (self.held and self.held[0] is knob)
            active = not waits or self.mind.active(knob)
            self._knob(surface, knob, centre, radius, lit, accent if active else FAINT, cell_w - 6, active)
        shown = (self.held[0] if self.held else None) or self.hover
        if shown:
            self._caption(surface, self._explain(shown), height - 84, width - 48)
        elif self.preset_hover and self.preset_places:
            name, what, *_ = self.preset_hover
            self._caption(surface, f"{name}: {what}", height - 84, width - 48)

    def _presets(self, surface, accent, width):
        """Draws the presets as a row of buttons, wrapping, the one playing lit; returns the bottom of the last row."""
        x, y = 24, 80
        for preset in self.mind.presets():
            rect = pygame.Rect(x, y, self.small.size(preset[0])[0] + 20, 24)
            if rect.right > width - 24 and x > 24:
                x, y = 24, y + 30
                rect.topleft = (x, y)
            playing = self.mind.playing(preset)
            pygame.draw.rect(surface, accent if playing else BODY, rect, border_radius=12)
            pygame.draw.rect(surface, accent if playing or preset == self.preset_hover else EDGE, rect, 1, border_radius=12)
            surface.blit(self.small.render(preset[0], True, PANEL if playing else TEXT), (rect.x + 10, rect.y + 6))
            self.preset_places.append((preset, rect))
            x = rect.right + 6
        return y + 24

    def _explain(self, knob):
        if knob.unit != 'macro':
            return knob.what
        turns = ', '.join(name.replace('_', ' ') for name in self.mind.turns[knob.name])
        return f"{knob.ends[0]} to {knob.ends[1]}: {knob.what} ({turns})"

    def _tabs(self, surface):
        x, self.tab_places = 24, []
        for i, (title, accent, *_) in enumerate(self.tabs):
            label = self.tab_font.render(title, True, TEXT if i == self.tab else DIM)
            rect = pygame.Rect(x - 6, 18, label.get_width() + 12, 30)
            self.tab_places.append(rect)
            surface.blit(label, (x, 26))
            if i == self.tab:
                pygame.draw.line(surface, accent, (x, 46), (x + label.get_width(), 46), 2)
            x += label.get_width() + 16

    def _knob(self, surface, knob, centre, radius, lit, accent, room, active):
        start = 90 + SWEEP / 2                       # degrees, counter-clockwise from 3 o'clock
        arc(surface, centre, radius + 7, 4, start, start - SWEEP, TRACK)
        if knob.share() > 0.002:
            arc(surface, centre, radius + 7, 4, start, start - SWEEP * knob.share(), accent)
        x, y = centre
        pygame.gfxdraw.filled_circle(surface, x, y, radius, BODY)
        pygame.gfxdraw.aacircle(surface, x, y, radius, accent if lit else EDGE)
        angle = math.radians(start - SWEEP * knob.share())
        tip = (x + math.cos(angle) * (radius - 6), y - math.sin(angle) * (radius - 6))
        pygame.draw.line(surface, TEXT if active else DIM, (x + math.cos(angle) * 6, y - math.sin(angle) * 6), tip, 3)
        moved = abs(knob.value - knob.default) > (knob.high - knob.low) / 200
        label = self.small.render(knob.label(), True, TEXT if lit or active else FAINT)
        surface.blit(label, (x - label.get_width() // 2, y + radius + 10))
        text = knob.shown()
        while self.small.size(text)[0] > room and len(text) > 4:
            text = text[:-2] + '.'
        value = self.small.render(text, True, accent if moved and active else DIM if active else FAINT)
        surface.blit(value, (x - value.get_width() // 2, y + radius + 26))

    def _caption(self, surface, text, y, width):
        line = ''
        for word in text.split():
            if self.font.size(f"{line} {word}".strip())[0] > width:
                surface.blit(self.font.render(line, True, TEXT), (24, y))
                y, line = y + 22, word
            else:
                line = f"{line} {word}".strip()
        surface.blit(self.font.render(line, True, TEXT), (24, y))


def arc(surface, centre, radius, width, start, end, colour):
    """A smooth thick arc from `start` to `end` degrees (counter-clockwise from 3 o'clock, `end` < `start`)."""
    steps = max(2, int(abs(start - end) / 3))
    points = lambda r: [(centre[0] + r * math.cos(math.radians(start + (end - start) * i / steps)),
                         centre[1] - r * math.sin(math.radians(start + (end - start) * i / steps)))
                        for i in range(steps + 1)]
    outline = points(radius + width / 2) + points(radius - width / 2)[::-1]
    pygame.gfxdraw.aapolygon(surface, outline, colour)
    pygame.gfxdraw.filled_polygon(surface, outline, colour)
