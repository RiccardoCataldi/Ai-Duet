# AI Music Generator

An interactive AI music generation system that responds to MIDI keyboard input in real-time. The system uses a transformer-based model to generate music based on your playing, creating a collaborative experience between human and AI.

## Features

- Real-time MIDI input processing
- AI-powered music generation based on your playing
- A real-time duo: the model improvises on electric piano while you play, and follows your key
- Different instruments for input and generated music
- MIDI file saving and playback capabilities

## Requirements

- Python 3.8+
- MIDI keyboard/controller
- MIDI output device (virtual or physical)
- TensorFlow (for the AI model)

## Dependencies

The project requires several Python packages listed in `requirements.txt`. See the Setup section for installation instructions.

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Connect your MIDI keyboard/controller to your computer
4. Make sure you have a MIDI output device configured
5. The AI uses the pretrained [Anticipatory Music Transformer](https://github.com/jthickstun/anticipation) (trained on the Lakh MIDI dataset). It is downloaded from Hugging Face on first run: `music-medium-800k` on GPU/Apple Silicon, `music-small-800k` on CPU

## Usage

1. Start the model server (`python -m music_transformer.server`, or on a Jetson: see below)
2. Run the program (`AMT_SERVER` defaults to `localhost:8765`):
```bash
python main.py
```

3. Select your MIDI input device when prompted
4. Start playing on your MIDI keyboard: you play piano, the model plays electric piano at the same time
5. Press Ctrl+C to stop the program and save the duo to `generated/`

## Duo on a Jetson

The model runs on a Jetson Orin Nano (JetPack 6) and the laptop handles MIDI and sound; `run_duet.sh` connects the two.

**On the Jetson, once:**
- The repo in `~/Ai-Duet` with a venv `.venv` made with `--system-site-packages` (it uses NVIDIA's torch)
- The model converted to safetensors in `~/models/music-medium-800k` (transformers 5 refuses `.bin` files with torch < 2.6)
- Power mode MAXN_SUPER (it persists across reboots)
- `ssh jetson` works from the laptop with a key

**On the laptop, once:** FluidSynth with the `FluidR3_GM` soundfont, and the venv `.venv` of this repo.

**Every time the Jetson boots:**
```bash
ssh -t jetson sudo jetson_clocks   # the clocks go back to defaults after every reboot
./run_duet.sh                      # from this repo, on the laptop
```
`run_duet.sh` restarts everything: the server on the Jetson, the SSH tunnel on port 8765, FluidSynth and `main.py` on the KeyStep.

**Logs:** `~/amt_server.log` on the Jetson (`duo: started`, then how many notes it heard and played);
`duet.log` and `fluidsynth.log` on the laptop in `$LOG_DIR` (default `/tmp`).

**Changing the code on the Jetson:** it does not pull from git; copy the files over, e.g.
`scp music_transformer/duo.py jetson:~/Ai-Duet/music_transformer/`, then run `./run_duet.sh` again.
The session tests need torch, so run them there: `python -m unittest tests.test_duo_session`.

**Tuning by ear** (top of `music_transformer/duo.py`):
- `MIN_STEP`, `SPACE`: time between the model's onsets, when you rest and while you play
- `OFF_KEY`: how strongly it keeps to the pitch classes you just played (higher = more in your key)
- `NO_REPEAT`: the length of runs it may not repeat
- `MEMORY`: how much of its own playing it remembers (shorter = listens to you more)
- `temperature` and `top_p` in `DuoSession`: how adventurous its choices are

## Instrument Selection

When changing instruments, enter a number between 0-127. Some interesting combinations to try:

- Keyboard: Piano (0), Generated: Strings (48)
- Keyboard: Electric Piano (4), Generated: Synth Pad (88)
- Keyboard: Acoustic Guitar (24), Generated: Choir (52)
- Keyboard: Trumpet (56), Generated: Synth Brass (62)

The program supports all General MIDI instruments (0-127), including:
- Pianos and Keyboards (0-7)
- Chromatic Percussion (8-15)
- Organs (16-23)
- Guitars (24-31)
- Basses (32-39)
- Strings (40-47)
- Ensembles (48-55)
- Brass (56-63)
- Reeds (64-71)
- Pipes (72-79)
- Synth Leads (80-87)
- Synth Pads (88-95)
- Synth Effects (96-103)
- Ethnic (104-111)
- Percussive (112-119)
- Sound Effects (120-127)

## Project Structure

```
ai-music/
├── CK_rec/              # MIDI recording and playback module
├── music_transformer/   # Music generation model
├── trained_models/      # Place for your trained models
├── generated/           # Directory for saved generated music
├── main.py             # Main application
└── requirements.txt    # Project dependencies
```

## MIDI Configuration

The program supports:
- Multiple MIDI input devices
- MIDI output for playback
- Real-time MIDI message processing
- MIDI file saving and loading
- Different instruments on separate MIDI channels
- Program change messages for instrument selection

## Troubleshooting

If you encounter issues:
1. Check your MIDI device connections
2. Verify the model file exists in the correct location
3. Ensure all dependencies are installed
4. Check the console output for error messages
5. Make sure your MIDI output device is properly configured

## License

This project is licensed under the terms included in the LICENSE file.

## Contributing

Feel free to submit issues and enhancement requests!
