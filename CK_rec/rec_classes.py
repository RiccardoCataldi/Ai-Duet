import mido
from mido import Message, MidiFile, MidiTrack
import rtmidi

class CK_rec(object):
    def __init__(self, port, device_id, tempo=120, debug=True):
        self.port = port
        self.tempo = tempo
        self.debug = debug
        self.on_id = device_id
        self.__mid = MidiFile()
        self.__track = MidiTrack()
        self.prepareTrack()
        self.__activesense = 0
        self.setup_output()
    

    def setup_output(self):
        self.__midiout = rtmidi.MidiOut()
        self.__ports_out = self.__midiout.get_ports()
        if self.__ports_out:
            self.__midiout.open_port(0)  # Open the first available MIDI output device
            print(f"MIDI output opened on: {self.__ports_out[0]}")
    
    def play_midi_message(self, message):
        if self.__midiout and self.__midiout.is_port_open():
            try:
                self.__midiout.send_message(message)
            except Exception as e:
                print(f"Error sending MIDI message: {e}")

    def prepareTrack(self):
        print("Recording started")
        self.__mid.tracks.append(self.__track)

    def __call__(self, event, data=None):
        message, deltatime = event
        self.__activesense += deltatime
        if message[0] != 254: #ignore active sense
            miditime = int(round(mido.second2tick(self.__activesense, self.__mid.ticks_per_beat, mido.bpm2tempo(self.tempo))))
            if self.debug:
                print('deltatime: ', deltatime, 'msg: ', message, 'activecomp: ', self.__activesense)
            else:
                if message[0] == 144:  # Stampa solo note-on
                    print("Playing MIDI note:", message[1])
            if message[0] == self.on_id:
                self.__track.append(Message('note_on', note=message[1], velocity=message[2], time=miditime))
                self.play_midi_message(message)  # Invia il messaggio MIDI
                self.__activesense = 0
                
            elif message[0] == 176:
                self.__track.append(Message('control_change', channel=1, control=message[1], value=message[2], time=miditime))
            else:
                self.__track.append(Message('note_off', note=message[1], velocity=message[2], time=miditime))
                self.play_midi_message(message)  # Invia il messaggio MIDI
                self.__activesense = 0
                
    def closePort(self):
        self.__midiout.close_port()

    def saveTrack(self, name):
        self.__mid.save(name+'.mid')
        print("\nRecording saved as "+name+".mid in the Recordings folder\n")
        
    def clearTrack(self):
        self.__mid.tracks.remove(self.__track)
        self.__track = MidiTrack()
        self.prepareTrack()

    def play_external_midi(self, midi_file):
        try:
            mid = MidiFile(midi_file)
            for msg in mid.play():
                self.play_midi_message(msg.bytes())
        except Exception as e:
            print(f"Error playing external MIDI: {e}")
