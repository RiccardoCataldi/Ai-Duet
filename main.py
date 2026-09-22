import sys
import threading
import time

from CK_rec.rec_classes import CK_rec
from CK_rec.setup import Setup
from accompaniment import Collector, LayaPort, tick

GRID_S = 0.25


def _send(midi_rec, event):
    status = 0x80 if event["kind"] == "note_off" else 0x90
    midi_rec.play_midi_message([status, event["pitch"], event["velocity"]], is_generated=True)


def main():
    code_k = Setup()
    my_port = code_k.perform_setup()
    code_k.open_port(my_port)
    on_id = 151
    print("your note on id is: ", on_id)

    midi_rec = CK_rec(my_port, on_id, debug=False)
    collector = Collector()
    sounding = []
    busy = False
    gate = threading.Lock()

    def on_midi(event, data=None):
        midi_rec(event, data)
        message, _deltatime = event
        status = message[0]
        if status in (254, 176):
            return
        if status == on_id:
            collector.note_on(message[1], message[2], time.monotonic())
        else:
            collector.note_off(message[1])

    code_k.set_callback(on_midi)
    port = LayaPort()

    def run_tick(picture):
        nonlocal sounding, busy
        try:
            events, next_sounding, _state = tick(picture, port.predict)
        except Exception as exc:
            print(f"Error in accompaniment: {exc}")
        else:
            for event in events:
                _send(midi_rec, event)
            if events:
                attacked_at = time.monotonic()
                with gate:
                    sounding = [
                        {
                            "pitch": note["pitch"],
                            "velocity": note["velocity"],
                            "attacked_at": attacked_at,
                        }
                        for note in next_sounding
                    ]
        finally:
            with gate:
                busy = False

    next_grid = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if now >= next_grid:
                with gate:
                    if not busy:
                        busy = True
                        picture = collector.take(sounding, now)
                        threading.Thread(target=run_tick, args=(picture,), daemon=True).start()
                now = time.monotonic()
                while next_grid <= now:
                    next_grid += GRID_S
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        code_k.end()
        print("Recording Stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
