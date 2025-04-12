import mido
from mido import MidiFile, MidiTrack
from music21 import *
import threading
import os
import rtmidi
#import from parrentdir
import sys
import inspect
import msvcrt
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
        
        # Instrument settings
        self.input_instrument = 0  # Piano
        self.generated_instrument = 48  # Strings
        self.setup_instruments()
    
    def setup_instruments(self):
        """Set up initial instruments"""
        self.midi_recorder.set_input_instrument(self.input_instrument)
        self.midi_recorder.set_generated_instrument(self.generated_instrument)
    
    def set_input_instrument(self, program):
        """Change the input instrument"""
        self.input_instrument = program
        self.midi_recorder.set_input_instrument(program)
        print(f"Input instrument changed to: {self.get_instrument_name(program)}")
    
    def set_generated_instrument(self, program):
        """Change the generated music instrument"""
        self.generated_instrument = program
        self.midi_recorder.set_generated_instrument(program)
        print(f"Generated instrument changed to: {self.get_instrument_name(program)}")
    
    def get_instrument_name(self, program):
        """Get the name of an instrument from its program number"""
        instruments = {
            0: "Piano",
            1: "Bright Piano",
            2: "Electric Grand",
            3: "Honky-tonk Piano",
            4: "Electric Piano 1",
            5: "Electric Piano 2",
            6: "Harpsichord",
            7: "Clavinet",
            8: "Celesta",
            9: "Glockenspiel",
            10: "Music Box",
            11: "Vibraphone",
            12: "Marimba",
            13: "Xylophone",
            14: "Tubular Bells",
            15: "Dulcimer",
            16: "Drawbar Organ",
            17: "Percussive Organ",
            18: "Rock Organ",
            19: "Church Organ",
            20: "Reed Organ",
            21: "Accordion",
            22: "Harmonica",
            23: "Tango Accordion",
            24: "Acoustic Guitar (nylon)",
            25: "Acoustic Guitar (steel)",
            26: "Electric Guitar (jazz)",
            27: "Electric Guitar (clean)",
            28: "Electric Guitar (muted)",
            29: "Overdriven Guitar",
            30: "Distortion Guitar",
            31: "Guitar Harmonics",
            32: "Acoustic Bass",
            33: "Electric Bass (finger)",
            34: "Electric Bass (pick)",
            35: "Fretless Bass",
            36: "Slap Bass 1",
            37: "Slap Bass 2",
            38: "Synth Bass 1",
            39: "Synth Bass 2",
            40: "Violin",
            41: "Viola",
            42: "Cello",
            43: "Contrabass",
            44: "Tremolo Strings",
            45: "Pizzicato Strings",
            46: "Orchestral Harp",
            47: "Timpani",
            48: "String Ensemble 1",
            49: "String Ensemble 2",
            50: "Synth Strings 1",
            51: "Synth Strings 2",
            52: "Choir Aahs",
            53: "Voice Oohs",
            54: "Synth Voice",
            55: "Orchestra Hit",
            56: "Trumpet",
            57: "Trombone",
            58: "Tuba",
            59: "Muted Trumpet",
            60: "French Horn",
            61: "Brass Section",
            62: "Synth Brass 1",
            63: "Synth Brass 2",
            64: "Soprano Sax",
            65: "Alto Sax",
            66: "Tenor Sax",
            67: "Baritone Sax",
            68: "Oboe",
            69: "English Horn",
            70: "Bassoon",
            71: "Clarinet",
            72: "Piccolo",
            73: "Flute",
            74: "Recorder",
            75: "Pan Flute",
            76: "Blown Bottle",
            77: "Shakuhachi",
            78: "Whistle",
            79: "Ocarina",
            80: "Lead 1 (square)",
            81: "Lead 2 (sawtooth)",
            82: "Lead 3 (calliope)",
            83: "Lead 4 (chiff)",
            84: "Lead 5 (charang)",
            85: "Lead 6 (voice)",
            86: "Lead 7 (fifths)",
            87: "Lead 8 (bass + lead)",
            88: "Pad 1 (new age)",
            89: "Pad 2 (warm)",
            90: "Pad 3 (polysynth)",
            91: "Pad 4 (choir)",
            92: "Pad 5 (bowed)",
            93: "Pad 6 (metallic)",
            94: "Pad 7 (halo)",
            95: "Pad 8 (sweep)",
            96: "FX 1 (rain)",
            97: "FX 2 (soundtrack)",
            98: "FX 3 (crystal)",
            99: "FX 4 (atmosphere)",
            100: "FX 5 (brightness)",
            101: "FX 6 (goblins)",
            102: "FX 7 (echoes)",
            103: "FX 8 (sci-fi)",
            104: "Sitar",
            105: "Banjo",
            106: "Shamisen",
            107: "Koto",
            108: "Kalimba",
            109: "Bagpipe",
            110: "Fiddle",
            111: "Shanai",
            112: "Tinkle Bell",
            113: "Agogo",
            114: "Steel Drums",
            115: "Woodblock",
            116: "Taiko Drum",
            117: "Melodic Tom",
            118: "Synth Drum",
            119: "Reverse Cymbal",
            120: "Guitar Fret Noise",
            121: "Breath Noise",
            122: "Seashore",
            123: "Bird Tweet",
            124: "Telephone Ring",
            125: "Helicopter",
            126: "Applause",
            127: "Gunshot"
        }
        return instruments.get(program, f"Unknown ({program})")

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
    
    print("\nControls:")
    print("i - Change input instrument")
    print("g - Change generated instrument")
    print("q - Quit")
    
    try:
        while True:
            if msvcrt.kbhit():  # Check if a key was pressed
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 'i':
                    try:
                        print("\nEnter input instrument number (0-127):")
                        program = int(input())
                        if 0 <= program <= 127:
                            app.set_input_instrument(program)
                        else:
                            print("Invalid program number")
                    except ValueError:
                        print("Please enter a valid number")
                elif key == 'g':
                    try:
                        print("\nEnter generated instrument number (0-127):")
                        program = int(input())
                        if 0 <= program <= 127:
                            app.set_generated_instrument(program)
                        else:
                            print("Invalid program number")
                    except ValueError:
                        print("Please enter a valid number")
                elif key == 'q':
                    break
            time.sleep(0.1)  # Keep main thread alive
                
    except KeyboardInterrupt:   
        pass
    finally:
        codeK.end()
        app.save_to_file()
        print('Recording Stopped')
        sys.exit(0)
        

           