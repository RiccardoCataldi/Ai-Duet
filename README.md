# AI Music Generator

An interactive AI music generation system that responds to MIDI keyboard input in real-time. The system uses a transformer-based model to generate music based on your playing, creating a collaborative experience between human and AI.

## Features

- Real-time MIDI input processing
- AI-powered music generation based on your playing
- Parallel playback of your performance and AI-generated music
- Different instruments for input and generated music
- Automatic generation every 5 seconds
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
5. Place your trained model in the `trained_models/decoder_only_smaller_1024_mega_ds` directory

## Usage

1. Run the program:
```bash
python main.py
```

2. Select your MIDI input device when prompted
3. Start playing on your MIDI keyboard
4. The system will:
   - Record your performance on one instrument (default: Piano)
   - Generate new music on a different instrument (default: Strings)
   - Play both parts in parallel

5. Use the following controls while playing:
   - Press 'i' to change your keyboard's instrument
   - Press 'g' to change the generated music's instrument
   - Press 'q' to quit

6. Press Ctrl+C to stop the program and save the generated music

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
