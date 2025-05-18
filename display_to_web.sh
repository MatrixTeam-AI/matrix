# How to forward display to web when running on Ubuntu server

# 0. Installation
# xvfb: for virtual display
# x11vnc: display to VNC
# noVNC: web client for VNC
# sudo apt-get update
# sudo apt-get install xvfb libgl1-mesa-glx
# sudo apt-get install x11vnc
# git clone https://github.com/novnc/noVNC.git # v1.6.0

# 1. Start virtual display
Xvfb :1 -screen 0 720x480x24 &
export DISPLAY=:1

# 2. forward display to web
x11vnc -display :1 -forever -shared -rfbport 5901 &
sleep 5
cd noVNC
./utils/novnc_proxy --vnc localhost:5901 --listen localhost:6080 &

# 3. open browser
# forward 6080 port in VSCode and open the web.
# double click the file `vnc.html`

# 4. Export environment variables in the terminal to enable display access
export DISPLAY=:1
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LIBGL_ALWAYS_SOFTWARE=1