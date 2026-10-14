"""
Knobs on screen next to the piano roll, in three tabs: the model's mind (its knobs on the Jetson, see
music_transformer/server.py, "knobs"), your sound and the model's (sound.py, in FluidSynth on this laptop).
Drag a knob up or down (Shift: finely) or turn the wheel over it; a double click puts it back.
"""
import math
import socket
import threading
import time

import pygame
import pygame.gfxdraw

import sound

PANEL_W = 330
COLUMNS = 3
DRAG = 220                # pixels of drag across a knob's whole range
STEPS = 50                # wheel steps across it
DOUBLE_CLICK_S = 0.35
SWEEP = 270               # degrees a knob turns, its gap at the bottom

PANEL = (17, 20, 31)
EDGE = (44, 49, 68)
TRACK = (46, 52, 72)
BODY = (30, 34, 48)
YOU = (255, 176, 84)      # the colours of the piano roll
AI = (94, 214, 230)
TEXT = (210, 214, 226)
DIM = (120, 126, 146)


class Knob:
    """`turned(knob)` is called whenever what it shows changes."""

    def __init__(self, name, value, low, high, default, unit, what, whole, turned):
        self.name, self.unit, self.what, self.whole, self.turned = name, unit, what, whole, turned
        self.value, self.low, self.high, self.default = float(value), float(low), float(high), float(default)

    def set(self, value):
        before = self.shown()
        self.value = min(max(value, self.low), self.high)
        if self.shown() != before:
            self.turned(self)

    def step(self):
        """How far one turn of the wheel moves it."""
        if self.unit == 'program':
            return 1
        span = (self.high - self.low) / STEPS
        return max(1, round(span)) if self.whole else span

    def shown(self):
        value = round(self.value) if self.whole else self.value
        if self.unit == 'ticks':
            return f"{value / 100:.2f} s"
        if self.unit == 'notes':
            return f"{value} notes"
        if self.unit == 'program':
            return sound.GM[value]
        if self.unit in ('octave_from', 'octave_to'):
            return f"{'C' if self.unit == 'octave_from' else 'B'}{value}"
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


class Server:
    """The knobs connection to the AMT server; it connects again whenever it is lost."""

    def __init__(self, address):
        self.address = address
        self.knobs = []
        self.sock = None
        self.lock = threading.Lock()
        threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        host, port = self.address.rsplit(':', 1)
        while True:
            if self.sock is None:
                try:
                    sock = socket.create_connection((host, int(port)), timeout=3)
                    sock.sendall(b"knobs\n")
                    knobs = []
                    for line in sock.makefile('r'):
                        if line.strip() == 'ready':
                            break
                        _, name, value, low, high, default, unit, what = line.split(maxsplit=7)
                        knobs.append(Knob(name, value, low, high, default, unit.strip('-'), what.strip(),
                                          '.' not in default, self.send))
                    sock.settimeout(None)
                    with self.lock:
                        self.knobs, self.sock = knobs, sock
                except (OSError, ValueError):
                    pass
            time.sleep(2)

    def send(self, knob):
        with self.lock:
            if self.sock is None:
                return
            try:
                self.sock.sendall(f"set {knob.name} {knob.value:.4f}\n".encode())
            except OSError:
                self.sock, self.knobs = None, []


def sound_knobs(engine, player):
    """The knobs of `player`'s sound in `engine`."""
    knobs = []
    for name, (*defaults, low, high, unit, what) in sound.SOUND.items():
        default = defaults[player]
        knobs.append(Knob(name, engine.sound[player][name], low, high, default, unit, what, isinstance(default, int),
                          lambda knob: engine.set(player, knob.name, knob.value)))
    return knobs


class Panel:
    def __init__(self, address, engine):
        self.server = Server(address)
        self.tabs = [('AI mind', AI, lambda: self.server.knobs,
                      lambda: 'turn them while you play' if self.server.knobs
                      else f"waiting for the model at {address}"),
                     ('Your sound', YOU, lambda knobs=sound_knobs(engine, sound.YOU): knobs,
                      lambda: 'in FluidSynth, on this laptop'),
                     ('AI sound', AI, lambda knobs=sound_knobs(engine, sound.AI): knobs,
                      lambda: 'in FluidSynth, on this laptop')]
        self.tab = 0
        self.font = pygame.font.Font(None, 24)
        self.small = pygame.font.Font(None, 19)
        self.held = None          # (knob, y where the drag started, value then)
        self.hover = None
        self.last_click = (None, 0)
        self.places = {}          # knob name -> (centre, radius), from the last draw
        self.tab_places = []      # rects of the tabs, from the last draw

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
            if self.held:
                knob, y, start = self.held
                fine = 5 if pygame.key.get_mods() & pygame.KMOD_SHIFT else 1
                knob.set(start + (y - pos[1]) / (DRAG * fine) * (knob.high - knob.low))

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
        _, accent, _, status = self.tabs[self.tab]
        self._tabs(surface)
        surface.blit(self.small.render(status(), True, DIM), (24, 58))
        cell_w = (width - 24) / COLUMNS
        cell_h = 104
        radius = 24
        self.places = {}
        for i, knob in enumerate(self.knobs()):
            col, row = i % COLUMNS, i // COLUMNS
            centre = (round(12 + cell_w * (col + 0.5)), round(116 + row * cell_h))
            self.places[knob.name] = (centre, radius)
            lit = knob is self.hover or (self.held and self.held[0] is knob)
            self._knob(surface, knob, centre, radius, lit, accent, cell_w - 6)
        shown = (self.held[0] if self.held else None) or self.hover
        if shown:
            self._caption(surface, shown.what, height - 62, width - 48)

    def _tabs(self, surface):
        x, self.tab_places = 24, []
        for i, (title, accent, _, _) in enumerate(self.tabs):
            label = self.font.render(title, True, TEXT if i == self.tab else DIM)
            rect = pygame.Rect(x - 6, 18, label.get_width() + 12, 30)
            self.tab_places.append(rect)
            surface.blit(label, (x, 25))
            if i == self.tab:
                pygame.draw.line(surface, accent, (x, 47), (x + label.get_width(), 47), 2)
            x += label.get_width() + 22

    def _knob(self, surface, knob, centre, radius, lit, accent, room):
        start = 90 + SWEEP / 2                       # degrees, counter-clockwise from 3 o'clock
        arc(surface, centre, radius + 7, 4, start, start - SWEEP, TRACK)
        if knob.share() > 0.002:
            arc(surface, centre, radius + 7, 4, start, start - SWEEP * knob.share(), accent)
        x, y = centre
        pygame.gfxdraw.filled_circle(surface, x, y, radius, BODY)
        pygame.gfxdraw.aacircle(surface, x, y, radius, accent if lit else EDGE)
        angle = math.radians(start - SWEEP * knob.share())
        tip = (x + math.cos(angle) * (radius - 6), y - math.sin(angle) * (radius - 6))
        pygame.draw.line(surface, TEXT, (x + math.cos(angle) * 6, y - math.sin(angle) * 6), tip, 3)
        moved = abs(knob.value - knob.default) > (knob.high - knob.low) / 200
        label = self.small.render(knob.label(), True, TEXT if lit else DIM)
        surface.blit(label, (x - label.get_width() // 2, y + radius + 10))
        text = knob.shown()
        while self.small.size(text)[0] > room and len(text) > 4:
            text = text[:-2] + '.'
        value = self.small.render(text, True, accent if moved else DIM)
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
