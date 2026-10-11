"""
Client for music_transformer.server: same respond() and stream() as AMTGenerator, no torch needed.
"""

import json
import urllib.request


class RemoteAMTGenerator:
    def __init__(self, url, timeout=60):
        self.base = url.rstrip('/')
        self.url = self.base + '/respond'
        self.timeout = timeout

    def respond(self, midi_in, midi_out):
        with open(midi_in, 'rb') as f:
            body = f.read()
        request = urllib.request.Request(self.url, data=body, headers={'Content-Type': 'audio/midi'})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if response.status == 204:
                return False
            data = response.read()
        with open(midi_out, 'wb') as f:
            f.write(data)
        return True

    def stream(self, notes, start, seconds=8):
        body = json.dumps({'notes': notes, 'start': start, 'seconds': seconds}).encode()
        request = urllib.request.Request(self.base + '/stream', data=body, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for line in response:
                onset, duration, pitch = line.split()
                yield float(onset), float(duration), int(pitch)
