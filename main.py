import mido
from mido import MidiFile, MidiTrack
import threading
import os
import rtmidi

# import from parrentdir
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from CK_rec.setup import Setup
from CK_rec.rec_classes import CK_rec

import time
from music_transformer.amt import AMTGenerator


class MusicGeneratorApp:
    def __init__(self, midi_recorder):
        try:
            self.generator = AMTGenerator()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.generator = None
        self.generated = None
        self.midi_recorder = midi_recorder
        self.midi_file_path = "output.mid"

        self.midi_recorder.set_input_instrument(0)
        self.midi_recorder.set_generated_instrument(48)

    def generate(self):
        if self.generator is None:
            print("Cannot generate music: model not loaded")
            return

        try:
            temp_midi = f"temp_generated_{int(time.time())}.mid"
            if not self.generator.respond(self.midi_file_path, temp_midi):
                print("No valid MIDI data to generate from")
                return
            self.generated = temp_midi
            self.midi_recorder.play_external_midi(temp_midi)
        except Exception as e:
            print(f"Error generating music: {e}")

    def save_to_file(self):
        if self.generated is not None:
            os.makedirs("generated", exist_ok=True)
            os.replace(self.generated, f"./generated/{int(time.time())}.mid")


if __name__ == "__main__":
    # Start the Device
    codeK = Setup()
    myPort = codeK.perform_setup()
    codeK.open_port(myPort)
    on_id = 151
    print("your note on id is: ", on_id)

    # Initialize MIDI recorder
    midiRec = CK_rec(myPort, on_id, debug=False)
    codeK.set_callback(midiRec)

    # Initialize app with MIDI recorder
    app = MusicGeneratorApp(midiRec)

    def generate_and_play():
        while True:
            try:
                midiRec.saveTrack("output")
                app.generate()
                midiRec.clearTrack()
                time.sleep(5)  # Wait 5 seconds before next generation
            except Exception as e:
                print(f"Error in generation thread: {e}")

    # Start generation thread
    generation_thread = threading.Thread(target=generate_and_play, daemon=True)
    generation_thread.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        codeK.end()
        app.save_to_file()
        print("Recording Stopped")
        sys.exit(0)
