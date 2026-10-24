#!/bin/bash
# Readies the Jetson for the duo, once: ssh -t jetson sudo bash Ai-Duet/jetson_setup.sh
# - run_duet.sh may max the clocks (jetson_clocks) without a password
# - about 1.2 GB more for the model: it runs without a desktop, and the services the duo does not use stay off.
#   The model alone is at the edge of the Orin Nano's 8 GB; with them on, its GPU allocations fail (NvMap error 12).
# Undo: sudo systemctl set-default graphical.target; sudo systemctl enable --now <service>; sudo rm /etc/sudoers.d/jetson_clocks
set -e
WHO=${SUDO_USER:?run it with sudo}
SERVICES="jupyterlab ollama docker docker.socket containerd"   # those this Jetson had on

echo "$WHO ALL=(root) NOPASSWD: /usr/bin/jetson_clocks" > /etc/sudoers.d/jetson_clocks
chmod 440 /etc/sudoers.d/jetson_clocks
visudo -cf /etc/sudoers.d/jetson_clocks

for service in $SERVICES; do
    systemctl list-unit-files "$service*" | grep -q "^$service" && systemctl disable --now "$service" && echo "off: $service"
done
# the desktop's own FluidSynth: the duo's plays on the laptop
sudo -u "$WHO" XDG_RUNTIME_DIR="/run/user/$(id -u "$WHO")" systemctl --user disable --now fluidsynth 2>/dev/null \
    && echo "off: fluidsynth (user)"

systemctl set-default multi-user.target
systemctl isolate multi-user.target   # the desktop stops now: ssh is all the duo needs
/usr/bin/jetson_clocks
free -h | head -2
