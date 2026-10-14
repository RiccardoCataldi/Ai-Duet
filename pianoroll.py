"""
A piano roll of the duo: your notes and the model's rise from the keyboard as you play.

    python pianoroll.py                                   # live, next to main.py (F: fullscreen, Esc: quit)
    python pianoroll.py video duo.mid --audio duo.ogg -o duo.mp4

Live, it listens to what main.py sends to FluidSynth on Midi Through: channel 0 is you, channel 1 the model.
"""
import argparse
import subprocess
import sys
import time

import mido
import pygame

SIZE = (1280, 720)
FPS = 60                  # live
VIDEO_FPS = 30
RANGE = (36, 96)          # live: C2 to C7; a video fits the octaves its notes use
KEYBOARD_H = 120
SPEED = 140               # pixels a second the notes rise
TAIL_S = 5                # seconds the video goes on after the last note, while the reverb dies away

BACKGROUND = ((12, 14, 22), (22, 26, 40))  # top, bottom
YOU = (255, 176, 84)      # amber
AI = (94, 214, 230)       # teal
WHITE_KEY = (232, 234, 240)
BLACK_KEY = (28, 30, 38)
GRID = (255, 255, 255, 14)
TEXT = (210, 214, 226)
DIM = (120, 126, 146)

PLAYERS = {0: ('You', 'piano', YOU), 1: ('AI', 'electric piano', AI)}


def is_black(pitch):
    return pitch % 12 in (1, 3, 6, 8, 10)


class Keyboard:
    """Where each key is, and the keyboard drawn once."""

    def __init__(self, width, top, low, high):
        self.top = top
        whites = [p for p in range(low, high + 1) if not is_black(p)]
        self.white_w = width / len(whites)
        self.black_w = self.white_w * 0.6
        self.x = {}
        for i, p in enumerate(whites):
            self.x[p] = (i * self.white_w, self.white_w)
        for p in range(low, high + 1):
            if is_black(p):
                left, _ = self.x[p - 1]
                self.x[p] = (left + self.white_w - self.black_w / 2, self.black_w)
        self.surface = pygame.Surface((width, KEYBOARD_H), pygame.SRCALPHA)
        self._draw(self.surface, {})

    def _draw(self, surface, lit):
        black_h = KEYBOARD_H * 0.62
        for p, (x, w) in self.x.items():
            if not is_black(p):
                rect = pygame.Rect(round(x) + 1, 0, round(w) - 1, KEYBOARD_H - 2)
                pygame.draw.rect(surface, lit.get(p, WHITE_KEY), rect, border_bottom_left_radius=4,
                                 border_bottom_right_radius=4)
        for p, (x, w) in self.x.items():
            if is_black(p):
                rect = pygame.Rect(round(x), 0, round(w), black_h)
                pygame.draw.rect(surface, lit.get(p, BLACK_KEY), rect, border_bottom_left_radius=3,
                                 border_bottom_right_radius=3)

    def draw(self, screen, lit):
        """`lit` maps the pitches sounding now to the colour of who plays them."""
        if not lit:
            screen.blit(self.surface, (0, self.top))
            return
        keys = self.surface.copy()
        self._draw(keys, {p: mix(colour, WHITE_KEY if not is_black(p) else BLACK_KEY, 0.15)
                          for p, colour in lit.items()})
        screen.blit(keys, (0, self.top))


def mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


class Roll:
    """Draws the notes, `(start, end or None, pitch, channel)` in seconds, as they are at `now`."""

    def __init__(self, size=SIZE, keys=RANGE):
        self.size = size
        width, height = size
        self.keyboard = Keyboard(width, height - KEYBOARD_H, *keys)
        self.background = vertical_gradient(size, *BACKGROUND)
        self.grid = pygame.Surface((width, height - KEYBOARD_H), pygame.SRCALPHA)
        for p, (x, _) in self.keyboard.x.items():
            if p % 12 == 0:
                pygame.draw.line(self.grid, GRID, (round(x), 0), (round(x), height - KEYBOARD_H))
        # the notes fade out as they rise to the top
        self.fade = pygame.Surface((width, height // 3), pygame.SRCALPHA)
        for y in range(height // 3):
            alpha = round(255 * (1 - y / (height // 3)) ** 2)
            self.fade.fill((*BACKGROUND[0], alpha), pygame.Rect(0, y, width, 1))
        self.glow = {c: glow_sprite(colour) for c, (_, _, colour) in PLAYERS.items()}
        self.font = pygame.font.Font(None, 26)
        self.small = pygame.font.Font(None, 22)

    def draw(self, screen, notes, now, clock=None):
        screen.blit(self.background, (0, 0))
        screen.blit(self.grid, (0, 0))
        top = self.keyboard.top
        lit = {}
        for start, end, pitch, channel in notes:
            if pitch not in self.keyboard.x or channel not in PLAYERS:
                continue
            colour = PLAYERS[channel][2]
            ended = now if end is None or end > now else end
            if start > now:
                continue
            y_top = top - (now - start) * SPEED
            y_bottom = top - (now - ended) * SPEED
            if y_bottom < 0:
                continue
            x, w = self.keyboard.x[pitch]
            inset = 2 if not is_black(pitch) else 1
            rect = pygame.Rect(round(x) + inset, round(y_top), max(2, round(w) - 2 * inset),
                               max(3, round(y_bottom - y_top)))
            pygame.draw.rect(screen, colour, rect, border_radius=4)
            pygame.draw.rect(screen, mix(colour, (255, 255, 255), 0.45), (rect.x, rect.y, rect.w, min(rect.h, 3)),
                             border_top_left_radius=4, border_top_right_radius=4)
            if ended == now:
                lit[pitch] = colour
                sprite = self.glow[channel]
                screen.blit(sprite, (round(x + w / 2 - sprite.get_width() / 2), top - sprite.get_height() // 2),
                            special_flags=pygame.BLEND_ADD)
        screen.blit(self.fade, (0, 0))
        pygame.draw.line(screen, (60, 66, 88), (0, top - 1), (self.size[0], top - 1), 2)
        self.keyboard.draw(screen, lit)
        self._legend(screen, clock)

    def _legend(self, screen, clock):
        x = 28
        for name, instrument, colour in PLAYERS.values():
            pygame.draw.circle(screen, colour, (x + 6, 34), 6)
            label = self.font.render(name, True, TEXT)
            screen.blit(label, (x + 20, 25))
            sub = self.small.render(instrument, True, DIM)
            screen.blit(sub, (x + 26 + label.get_width(), 28))
            x += 50 + label.get_width() + sub.get_width()
        title = self.font.render('AI Duet', True, DIM)
        screen.blit(title, (self.size[0] - title.get_width() - 28, 25))
        if clock is not None:
            stamp = self.small.render(f"{int(clock) // 60}:{int(clock) % 60:02d}", True, DIM)
            screen.blit(stamp, (self.size[0] - stamp.get_width() - 28, 50))


def vertical_gradient(size, top, bottom):
    surface = pygame.Surface(size, pygame.SRCALPHA)
    for y in range(size[1]):
        surface.fill(mix(top, bottom, y / max(1, size[1] - 1)), pygame.Rect(0, y, size[0], 1))
    return surface


def glow_sprite(colour, radius=34):
    sprite = pygame.Surface((radius * 2, radius * 2))
    for r in range(radius, 0, -1):
        t = (1 - r / radius) ** 2 * 0.5
        pygame.draw.circle(sprite, tuple(round(c * t) for c in colour), (radius, radius), r)
    return sprite


def octaves(notes):
    """The keys from the C below the lowest note to the C above the highest."""
    pitches = [n[2] for n in notes]
    return min(pitches) // 12 * 12, max(pitches) // 12 * 12 + 12


def notes_from_midi(path):
    """The notes of a saved duo, `(start, end, pitch, channel)` in seconds."""
    notes, held, now = [], {}, 0.0
    for message in mido.MidiFile(path):
        now += message.time
        if message.type == 'note_on' and message.velocity > 0:
            held[(message.channel, message.note)] = now
        elif message.type in ('note_on', 'note_off'):
            start = held.pop((message.channel, message.note), None)
            if start is not None:
                notes.append((start, now, message.note, message.channel))
    return notes


class Live:
    """The notes heard on a MIDI port, kept while they are on screen."""

    def __init__(self):
        self.notes = []          # [start, end or None, pitch, channel]
        self.held = {}           # (channel, pitch) -> note

    def message(self, message, now):
        if len(message) < 3:
            return
        kind, channel, pitch, velocity = message[0] & 0xF0, message[0] & 0x0F, message[1], message[2]
        if kind == 0x90 and velocity > 0:
            self.off(channel, pitch, now)
            note = [now, None, pitch, channel]
            self.held[(channel, pitch)] = note
            self.notes.append(note)
        elif kind in (0x80, 0x90):
            self.off(channel, pitch, now)

    def off(self, channel, pitch, now):
        note = self.held.pop((channel, pitch), None)
        if note is not None:
            note[1] = now

    def visible(self, now, height):
        self.notes = [n for n in self.notes if n[1] is None or (now - n[1]) * SPEED < height]
        return self.notes


def live(port_name):
    import rtmidi
    pygame.init()
    pygame.display.set_caption('AI Duet')
    screen = pygame.display.set_mode(SIZE, pygame.RESIZABLE)
    midi = rtmidi.MidiIn()
    ports = midi.get_ports()
    port = next((i for i, p in enumerate(ports) if port_name in p), None)
    if port is None:
        sys.exit(f"No MIDI port named {port_name!r} in {ports}")
    midi.open_port(port)
    heard = Live()
    midi.set_callback(lambda event, data=None: heard.message(event[0], time.monotonic()))
    roll, clock, started = Roll(screen.get_size()), pygame.time.Clock(), time.monotonic()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                pygame.display.toggle_fullscreen()
            if event.type == pygame.VIDEORESIZE:
                roll = Roll(screen.get_size())
        now = time.monotonic()
        roll.draw(screen, [tuple(n) for n in heard.visible(now, screen.get_height())], now, now - started)
        pygame.display.flip()
        clock.tick(FPS)


def video(midi_path, audio, out):
    import imageio_ffmpeg
    pygame.init()
    notes = notes_from_midi(midi_path)
    length = max(n[1] for n in notes) + TAIL_S
    screen = pygame.Surface(SIZE)
    roll = Roll(SIZE, octaves(notes))
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(VIDEO_FPS), '-i', '-']
    if audio:
        command += ['-i', audio, '-c:a', 'aac', '-b:a', '160k', '-shortest']
    command += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'slow', '-movflags', '+faststart',
                out]
    ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(int(length * VIDEO_FPS)):
        now = frame / VIDEO_FPS
        roll.draw(screen, notes, now, now)
        ffmpeg.stdin.write(pygame.image.tobytes(screen, 'RGB'))
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"Video saved as {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', nargs='?', choices=('live', 'video'), default='live')
    parser.add_argument('midi', nargs='?', help='video: the saved duo (generated/<time>.mid)')
    parser.add_argument('--audio', help='video: its rendering, to put under the picture')
    parser.add_argument('-o', '--out', default='duo.mp4')
    parser.add_argument('--port', default='Midi Through', help='live: the MIDI port to listen to')
    args = parser.parse_args()
    if args.mode == 'video':
        video(args.midi, args.audio, args.out)
    else:
        live(args.port)
