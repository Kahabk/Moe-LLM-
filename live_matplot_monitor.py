#!/usr/bin/env python3
# live_matplot_monitor.py
# Real-time Matplotlib monitor for your MoE training

import socket
import json
import matplotlib.pyplot as plt
import threading
import time
import collections

UDP_IP = "127.0.0.1"
UDP_PORT = 9999

# buffers
MAX_POINTS = 4000
steps = collections.deque(maxlen=MAX_POINTS)
losses = collections.deque(maxlen=MAX_POINTS)
lrs    = collections.deque(maxlen=MAX_POINTS)

# UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)


def udp_listener():
    """Background thread: receives training metrics."""
    while True:
        try:
            data, _ = sock.recvfrom(65536)
            msg = json.loads(data.decode())

            steps.append(msg.get("step", 0))
            losses.append(msg.get("ce", 0.0))
            lrs.append(msg.get("lr", 0.0))

        except BlockingIOError:
            time.sleep(0.01)
        except Exception as e:
            print("Error:", e)


# start listening thread
threading.Thread(target=udp_listener, daemon=True).start()

# OPTIONAL: Enable Seaborn theme
# import seaborn as sns
# sns.set_theme(style="darkgrid")

plt.ion()
fig, ax1 = plt.subplots(figsize=(10,5))
ax2 = ax1.twinx()

line_loss, = ax1.plot([], [], "r-", label="CE Loss", linewidth=1.8)
line_lr, = ax2.plot([], [], "y-", label="LR", linewidth=1)

ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")
ax2.set_ylabel("LR")

fig.suptitle("MoE Training Live Monitor (Matplotlib)", fontsize=14)
fig.legend(loc="upper right")

print("Live monitor running. Waiting for training data...")

# real-time update loop
while True:
    if len(steps) > 2:
        line_loss.set_data(steps, losses)
        line_lr.set_data(steps, lrs)

        ax1.relim()
        ax1.autoscale_view()

        ax2.relim()
        ax2.autoscale_view()

    plt.pause(0.05)
