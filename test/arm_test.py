"""WASD debug teleop for the arm on port B, usable over plain SSH.

Type W/A/S/D in the terminal. Each key press is echoed, then the leader arm
moves by a fixed step. Run inside the lerobot env:  python wasd_teleop.py

The leader arm normally has torque off (you move it by hand). This script turns
torque ON so the arm can move itself, and turns it OFF again on exit.
"""
import select
import sys
import termios
import time
import tty

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ARM_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128613-if00"  # B

STEP = 40            # raw servo ticks per key press (4096 ticks = 360 deg, so 40 is about 3.5 deg)
MAX_EXCURSION = 600  # never go further than this many ticks from the starting pose
TORQUE_LIMIT = 400   # 0..1000, kept low so the arm is gentle and easy to stop by hand

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTORS = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(JOINTS, start=1)}

# key -> (joint, direction). Flip the sign if a key moves the wrong way.
KEYMAP = {
    "w": ("shoulder_lift", +1),
    "s": ("shoulder_lift", -1),
    "a": ("shoulder_pan", +1),
    "d": ("shoulder_pan", -1),
    "j": ("gripper", +1),  # open
    "k": ("gripper", -1),  # close
}


def main():
    bus = FeetechMotorsBus(port=ARM_PORT, motors=MOTORS)
    bus.connect()

    start = bus.sync_read("Present_Position", normalize=False)
    target = dict(start)
    print("Start positions (ticks):", start)

    # Hold the current pose first, then enable torque, so nothing jumps.
    for m in JOINTS:
        bus.write("Goal_Position", m, start[m], normalize=False)
        bus.write("Torque_Limit", m, TORQUE_LIMIT, normalize=False)
    bus.enable_torque()

    print("Ready. W/S = shoulder_lift, A/D = shoulder_pan, J = open claw, K = close claw, Space = hold, Q = quit.")
    print("Support the arm: it will go limp when this script exits.\n")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # read single keys without waiting for Enter
        while True:
            if not select.select([sys.stdin], [], [], 0.01)[0]:
                continue
            key = sys.stdin.read(1).lower()

            if key == "q":
                print("Q pressed: quitting")
                break
            if key == " ":
                target = bus.sync_read("Present_Position", normalize=False)
                for m in JOINTS:
                    bus.write("Goal_Position", m, target[m], normalize=False)
                print("SPACE pressed: holding current position")
                continue
            if key not in KEYMAP:
                print(f"Unmapped key pressed: {key!r}")
                continue

            joint, direction = KEYMAP[key]
            new = target[joint] + direction * STEP
            lo, hi = start[joint] - MAX_EXCURSION, start[joint] + MAX_EXCURSION
            new = max(max(lo, 0), min(min(hi, 4095), new))
            if new == target[joint]:
                print(f"{key.upper()} pressed: {joint} at limit, not moving")
                continue
            target[joint] = new
            bus.write("Goal_Position", joint, new, normalize=False)
            print(f"{key.upper()} pressed: {joint} {direction * STEP:+d} -> goal {new}")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        bus.disable_torque()
        bus.disconnect()
        print("Torque off, port closed.")


if __name__ == "__main__":
    main()