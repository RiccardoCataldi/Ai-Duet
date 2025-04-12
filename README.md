# AI Music Generator

An interactive AI music generation system that responds to MIDI keyboard input in real-time. The system uses a transformer-based model to generate music based on your playing, creating a collaborative experience between human and AI.

## Features

- Real-time MIDI input processing
- AI-powered music generation based on your playing
- Parallel playback of your performance and AI-generated music
- Automatic generation every 5 seconds
- MIDI file saving and playback capabilities

## Requirements

- Python 3.8+
- MIDI keyboard/controller
- MIDI output device (virtual or physical)

## Dependencies

The project requires several Python packages. Install them using:

```bash
pip install -r requirements.txt
```

## Setup

1. Connect your MIDI keyboard/controller to your computer
2. Make sure you have a MIDI output device configured
3. Place your trained model in the `trained_models/decoder_only_smaller_1024_mega_ds` directory

## Usage

1. Run the program:
```bash
python main.py
```

2. Select your MIDI input device when prompted
3. Start playing on your MIDI keyboard
4. The system will:
   - Record your performance
   - Generate new music based on your playing every 5 seconds
   - Play both your performance and the generated music in parallel

5. Press Ctrl+C to stop the program and save the generated music

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

## Troubleshooting

If you encounter issues:
1. Check your MIDI device connections
2. Verify the model file exists in the correct location
3. Ensure all dependencies are installed
4. Check the console output for error messages

## License

This project is licensed under the terms included in the LICENSE file.

## Contributing

Feel free to submit issues and enhancement requests!
