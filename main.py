import mido
from mido import MidiFile, MidiTrack
from music21 import *
import threading
import os
import rtmidi
#import from parrentdir
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from CK_rec.setup import Setup
from CK_rec.rec_classes import CK_rec

from pyo import *

import time
import music21.midi.realtime
import numpy as np
import tensorflow as tf
from music_transformer.convert import midi2idxenc, idxenc2stream
from music_transformer.transformer import MusicGenerator
from music_transformer.vocab import MusicVocab


class MusicGeneratorApp:
    def __init__(self, midi_recorder):
        try:
            model = tf.saved_model.load('trained_models/decoder_only_smaller_1024_mega_ds')
            self.generator = MusicGenerator(model)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please make sure the model file exists in the trained_models directory")
            self.generator = None
        self.vocab = MusicVocab.create()
        self.generated = None
        self.sequence = None
        self.player = None
        self.is_playing = False
        self.midi_recorder = midi_recorder
        self.midi_file_path = "output.mid"
    
    def generate(self):
        if self.generator is None:
            print("Cannot generate music: model not loaded")
            return
            
        try:
            # Load MIDI file and convert to sequence
            inp = midi2idxenc(self.midi_file_path, self.vocab, add_eos=False)
            if len(inp) == 0:
                print("No valid MIDI data to generate from")
                return
        except Exception as e:
            print(f"Error converting MIDI: {e}")
            return

        try:
            generated = self.generator.extend_sequence(inp, max_generate_len=64, search="greedy",
                                                       top_k_notes=128, top_k_durations=128,
                                                       top_k_offset=0, beam_width=3,
                                                       creativity=100)
            self.sequence = generated.numpy()
            self.generated = idxenc2stream(self.sequence, vocab=self.vocab)
            
            # Save the generated music to a temporary MIDI file
            temp_midi = f"temp_generated_{int(time.time())}.mid"
            self.generated.write('midi', fp=temp_midi)
            
            # Play the generated music through MIDI output
            self.midi_recorder.play_external_midi(temp_midi)
            self.is_playing = True
            
        except Exception as e:
            print(f"Error generating music: {e}")

    def save_to_file(self):
        if self.generated is not None:
            try:
                if not os.path.isdir('./generated'):
                    os.mkdir('generated')
                self.generated.write('midi', fp=f'./generated/{int(time.time())}.mid')
            except Exception as e:
                print(e)


if __name__ == "__main__":
    # Start the Device
    codeK = Setup()
    myPort = codeK.perform_setup()
    codeK.open_port(myPort)
    on_id = 151
    print('your note on id is: ', on_id)

    # Initialize MIDI recorder
    midiRec = CK_rec(myPort, on_id, debug=False)
    codeK.set_callback(midiRec)
    
    # Initialize app with MIDI recorder
    app = MusicGeneratorApp(midiRec)
    
    def generate_and_play():
        while True:
            try:
                midiRec.saveTrack('output')
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
            time.sleep(0.1)  # Keep main thread alive
                
    except KeyboardInterrupt:   
        codeK.end()
        app.save_to_file()
        print('Recording Stopped')
        sys.exit(0)
        

           