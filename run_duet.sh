#!/bin/bash
# Restart the duet: AMT server on the Jetson, SSH tunnel, timidity synth and main.py on the laptop.
cd "$(dirname "$0")"
LOG=${LOG_DIR:-/tmp}

pkill -INT -f "^\.venv/bin/python main\.py"; sleep 2; pkill -f "^\.venv/bin/python main\.py"
pkill -x timidity
pkill -f "^ssh -N .*-L 8765:localhost:8765"
sleep 1

ssh jetson 'pkill -f "^python -u -m music_transformer\.server"; sleep 2
  cd ~/Ai-Duet || exit 1
  . .venv/bin/activate
  HF_HUB_OFFLINE=1 setsid nohup python -u -m music_transformer.server --model ~/models/music-medium-800k > ~/amt_server.log 2>&1 < /dev/null &
  for i in $(seq 60); do grep -q -E "listening|Traceback" ~/amt_server.log && break; sleep 2; done
  grep -E "listening|Traceback" ~/amt_server.log'

setsid nohup ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 8765:localhost:8765 jetson > /dev/null 2>&1 < /dev/null &
setsid nohup timidity -iA -OO > "$LOG/timidity.log" 2>&1 < /dev/null &
sleep 3
TIMIDITY=$(aconnect -o | awk -F'[ :]' '/TiMidity/ {print $2; exit}')
aconnect 14:0 "$TIMIDITY:0" && echo "Midi Through -> TiMidity ($TIMIDITY:0)"

KEYSTEP=$(.venv/bin/python -c "import rtmidi; print(next(i for i, p in enumerate(rtmidi.MidiIn().get_ports()) if 'KeyStep' in p))")
echo "$KEYSTEP" | AMT_SERVER=http://localhost:8765 PYTHONUNBUFFERED=1 setsid nohup .venv/bin/python main.py > "$LOG/duet.log" 2>&1 &
sleep 4
grep -E "chosen|output opened|remote" "$LOG/duet.log"
