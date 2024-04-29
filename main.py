import mido
from mido import MidiFile, MidiTrack
from music21 import *

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
    def __init__(self):
        model = tf.saved_model.load('trained_models/decoder_only_smaller_1024_mega_ds')
        self.generator = MusicGenerator(model)
        self.vocab = MusicVocab.create()
        self.generated = None
        self.sequence = None
        self.player = None
        self.is_playing = False

        # Imposta il percorso del file MIDI        
        self.midi_file_path = "output.mid"

    def generate(self):
        try:
            # Carica il file MIDI e lo converte in una sequenza
            inp = midi2idxenc(self.midi_file_path, self.vocab, add_eos=False)
        except Exception as e:
            print(e)
            return

        try:
            generated = self.generator.extend_sequence(inp, max_generate_len=64, search="greedy",
                                                       top_k_notes=128, top_k_durations=128,
                                                       top_k_offset=0, beam_width=3,
                                                       creativity=0)
            self.sequence = generated.numpy()
            self.generated = idxenc2stream(self.sequence, vocab=self.vocab)
            self.player = music21.midi.realtime.StreamPlayer(self.generated)
            self.player.play()
            self.is_playing = True
            
        except Exception as e:
            print(e)

    def save_to_file(self):
        if self.generated is not None:
            try:
                if not os.path.isdir('./generated'):
                    os.mkdir('generated')
                self.generated.write('midi', fp=f'./generated/{int(time.time())}.mid')
            except Exception as e:
                print(e)

if __name__ == "__main__":
    app = MusicGeneratorApp()
    
    input_port_name = mido.get_input_names()[0]

    # Start the Device
    codeK = Setup()
    myPort = codeK.perform_setup()
    codeK.open_port(myPort)
    #on_id = codeK.get_device_id()
    on_id = 151
    print('your note on id is: ', on_id)

    # record
    midiRec = CK_rec(myPort, on_id, debug=False)
    codeK.set_callback(midiRec)

    # set timer of 5 seconds
    timer = time.time()
    try:
        while True:
            if time.time() - timer > 2:
                midiRec.saveTrack('output')
                app.generate()
                timer = time.time()
                #del output.mid
                midiRec.clearTrack()
                
    except KeyboardInterrupt:   
        
        codeK.end()
        app.save_to_file()
        print('Recording Stopped')
        sys.exit(0)
        

           
