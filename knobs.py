"""
The duo's knobs, on screen next to the piano roll: drag one up or down (Shift: finely) or turn the wheel over it;
a double click puts it back. Each turn reaches the model on the Jetson from its next note on (see
music_transformer/server.py, "knobs").
"""
import math
import socket
import threading
import time

import pygame
import pygame.gfxdraw

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
ACCENT = (94, 214, 230)   # the model's colour: these are its knobs
TEXT = (210, 214, 226)
DIM = (120, 126, 146)


class Knob:
    def __init__(self, name, value, low, high, default, unit, what):
        self.name, self.unit, self.what = name, unit, what
        self.whole = '.' not in default    # a knob in ticks or notes
        self.value, self.low, self.high, self.default = float(value), float(low), float(high), float(default)

    def set(self, value):
        value = min(max(value, self.low), self.high)
        changed = round(value, 4) != round(self.value, 4)
        self.value = value
        return changed

    def shown(self):
        value = round(self.value) if self.whole else self.value
        if self.unit == 'ticks':
            return f"{value / 100:.2f} s"
        if self.unit == 'notes':
            return f"{value} notes"
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
                        kind, name, value, low, high, default, unit, what = line.split(maxsplit=7)
                        knobs.append(Knob(name, value, low, high, default, unit.strip('-'), what.strip()))
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


class Panel:
    def __init__(self, address):
        self.server = Server(address)
        self.font = pygame.font.Font(None, 24)
        self.small = pygame.font.Font(None, 19)
        self.held = None          # (knob, y where the drag started, value then)
        self.hover = None
        self.last_click = (None, 0)
        self.places = {}          # knob name -> (centre, radius), from the last draw

    def handle(self, event, origin):
        """Mouse events, at `origin` the panel's top left corner on screen."""
        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL):
            return
        if event.type == pygame.MOUSEWHEEL:
            knob = self._at(pygame.mouse.get_pos(), origin)
            if knob and knob.set(knob.value + event.y * (knob.high - knob.low) / STEPS):
                self.server.send(knob)
            return
        pos = event.pos
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            knob = self._at(pos, origin)
            if knob is None:
                return
            name, at = self.last_click
            if name == knob.name and time.monotonic() - at < DOUBLE_CLICK_S:
                if knob.set(knob.default):
                    self.server.send(knob)
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
                if knob.set(start + (y - pos[1]) / (DRAG * fine) * (knob.high - knob.low)):
                    self.server.send(knob)

    def _at(self, pos, origin):
        for knob in self.server.knobs:
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
        title = self.font.render('Knobs', True, TEXT)
        surface.blit(title, (24, 25))
        knobs = self.server.knobs
        status = 'turn them while you play' if knobs else f"waiting for the model at {self.server.address}"
        surface.blit(self.small.render(status, True, DIM), (24, 50))
        cell_w = (width - 24) / COLUMNS
        cell_h = 104
        radius = 24
        self.places = {}
        for i, knob in enumerate(knobs):
            col, row = i % COLUMNS, i // COLUMNS
            centre = (round(12 + cell_w * (col + 0.5)), round(110 + row * cell_h))
            self.places[knob.name] = (centre, radius)
            self._knob(surface, knob, centre, radius, knob is self.hover or (self.held and self.held[0] is knob))
        shown = (self.held[0] if self.held else None) or self.hover
        if shown:
            self._caption(surface, shown.what, height - 62, width - 48)

    def _knob(self, surface, knob, centre, radius, lit):
        start = 90 + SWEEP / 2                       # degrees, counter-clockwise from 3 o'clock
        arc(surface, centre, radius + 7, 4, start, start - SWEEP, TRACK)
        if knob.share() > 0.002:
            arc(surface, centre, radius + 7, 4, start, start - SWEEP * knob.share(), ACCENT)
        x, y = centre
        pygame.gfxdraw.filled_circle(surface, x, y, radius, BODY)
        pygame.gfxdraw.aacircle(surface, x, y, radius, EDGE if not lit else ACCENT)
        angle = math.radians(start - SWEEP * knob.share())
        tip = (x + math.cos(angle) * (radius - 6), y - math.sin(angle) * (radius - 6))
        pygame.draw.line(surface, TEXT, (x + math.cos(angle) * 6, y - math.sin(angle) * 6), tip, 3)
        moved = abs(knob.value - knob.default) > (knob.high - knob.low) / 200
        label = self.small.render(knob.label(), True, TEXT if lit else DIM)
        surface.blit(label, (x - label.get_width() // 2, y + radius + 10))
        value = self.small.render(knob.shown(), True, ACCENT if moved else DIM)
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
