"""
Client for music_transformer.server: same respond() interface as AMTGenerator, no torch needed.
"""

import urllib.request


class RemoteAMTGenerator:
    def __init__(self, url, timeout=60):
        self.url = url.rstrip('/') + '/respond'
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
