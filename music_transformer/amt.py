"""
Call-and-response generation with the Anticipatory Music Transformer (Stanford CRFM, trained on Lakh MIDI).
https://github.com/jthickstun/anticipation
"""

import torch
from transformers import AutoModelForCausalLM
from anticipation import ops
from anticipation.config import CONTEXT_SIZE, MAX_DUR, MAX_PITCH, TIME_RESOLUTION
from anticipation.vocab import AUTOREGRESS, DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.sample import safe_logits, future_logits, nucleus


class AMTGenerator:
    def __init__(self, model_name=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available()
                                 else 'mps' if torch.backends.mps.is_available() else 'cpu')
        model_name = model_name or ('stanford-crfm/music-medium-800k' if self.device != 'cpu'
                                    else 'stanford-crfm/music-small-800k')
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()

    def respond(self, midi_in, midi_out, seconds=8, instrument=0, max_prompt_events=150, **sampling):
        events = midi_to_events(midi_in)[-3 * max_prompt_events:]
        if not events:
            return False
        # play everything as one instrument and start the prompt at time 0
        events[2::3] = [NOTE_OFFSET + instrument * MAX_PITCH + (n - NOTE_OFFSET) % MAX_PITCH for n in events[2::3]]
        events = ops.translate(events, -ops.min_time(events, seconds=False))
        start = ops.max_time(events, seconds=False)
        generated = [token for event in self._sample(events, start, seconds, instrument, **sampling) for token in event]
        if not generated:
            return False
        events_to_midi(ops.translate(generated, -start)).save(midi_out)
        return True

    def stream(self, notes, start, seconds=8, instrument=0, max_prompt_events=150, **sampling):
        """
        notes: (onset, duration, pitch) in seconds; start: the instant the answer may begin.
        Yields the answer one (onset, duration, pitch) at a time, as soon as each note is sampled.
        """
        events = []
        for onset, duration, pitch in sorted(notes)[-max_prompt_events:]:
            events += [TIME_OFFSET + round(onset * TIME_RESOLUTION),
                       DUR_OFFSET + min(max(round(duration * TIME_RESOLUTION), 1), MAX_DUR - 1),
                       NOTE_OFFSET + instrument * MAX_PITCH + pitch]
        for time, duration, note in self._sample(events, round(start * TIME_RESOLUTION), seconds, instrument, **sampling):
            yield ((time - TIME_OFFSET) / TIME_RESOLUTION,
                   (duration - DUR_OFFSET) / TIME_RESOLUTION,
                   (note - NOTE_OFFSET) % MAX_PITCH)

    @torch.no_grad()
    def _sample(self, events, start, seconds, instrument, top_p=0.95, temperature=0.9):
        end = start + int(seconds * TIME_RESOLUTION)
        allowed = slice(NOTE_OFFSET + instrument * MAX_PITCH, NOTE_OFFSET + (instrument + 1) * MAX_PITCH)

        seq, past, event = [AUTOREGRESS] + events, None, []
        inp, now = seq, start
        while len(seq) < CONTEXT_SIZE:
            out = self.model(torch.tensor([inp], device=self.device), past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = safe_logits(out.logits[0, -1].float() / temperature, len(seq) - 1)
            if len(event) == 0:
                logits = future_logits(logits, now)
            elif len(event) == 2:
                mask = torch.full_like(logits, -float('inf'))
                mask[allowed] = 0
                logits = logits + mask
            token = int(torch.multinomial(torch.softmax(nucleus(logits, top_p), -1), 1))
            if len(event) == 0:
                now = token - TIME_OFFSET
                if now >= end:
                    break
            event.append(token)
            seq.append(token)
            inp = [token]
            if len(event) == 3:
                yield event
                event = []
