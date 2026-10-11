#!/usr/bin/env bash
# Start the robot: tracking, height/distance, gestures, voice, Go to, Ambient and the app's API.
# Usage: ./robot            (extra tracker options can be added, e.g. ./robot --dry-run)
# Output also goes to tracker/robot.log. If the tracker crashes it restarts; Ctrl+C stops it.
cd "$(dirname "$0")"

VOICES="Arnav Jaisree"   # people whose voice makes the arm look for their face
LOG=tracker/robot.log

if [[ " $* " == *" -h "* || " $* " == *" --help "* ]]; then
    exec .venv/bin/python tracker/track.py --help
fi

# Only one program can drive the arm: stop an older copy first and let it free the gripper.
if pkill -INT -f 'tracker/track\.py'; then
    echo "Stopping the tracker that was already running..."
    for _ in $(seq 20); do pgrep -f 'tracker/track\.py' >/dev/null || break; sleep 0.5; done
fi

# Say when the app can actually reach the tracker.
announce() {
    for _ in $(seq 60); do
        if curl -s -m 1 -o /dev/null http://127.0.0.1:8000/status; then
            echo ">>> Tracker online: the app can reach it now (http://$(hostname -I | awk '{print $1}'):8000)"
            return
        fi
        sleep 1
    done
    echo ">>> Tracker still not reachable after 60 s; see $LOG"
}

stopping=0
trap 'stopping=1' INT TERM
while (( ! stopping )); do
    # After a crash, don't fight a tracker someone else started: wait until it stops.
    other=$(pgrep -f 'tracker/track\.py' | head -1)
    if [[ -n "$other" ]]; then
        echo ">>> Another tracker (pid $other, $(ps -o lstart= -p "$other" | xargs)) is using the arm and camera."
        echo ">>> Waiting for it to stop; Ctrl+C here to quit, or stop it and this one takes over."
        while (( ! stopping )) && kill -0 "$other" 2>/dev/null; do sleep 2; done
        (( stopping )) && break
    fi
    echo "=== $(date '+%F %T') starting the tracker" | tee -a "$LOG"
    announce &
    announcer=$!
    .venv/bin/python -u tracker/track.py \
        --tilt --height --gestures \
        --distance 1.0 --distance-reference 1.0 --distance-face-width 100 \
        --depth-forward 0.05 --depth-back 0.05 --depth-rate 0.02 \
        --voice $VOICES --follow-mode speaker \
        "$@" 2>&1 | tee -i -a "$LOG"   # -i: Ctrl+C must not cut the log while the gripper is being freed
    status=${PIPESTATUS[0]}
    kill "$announcer" 2>/dev/null
    if (( stopping )) || (( status == 0 )) || (( status == 130 )); then
        echo "Tracker stopped."
        break
    fi
    echo ">>> Tracker exited with an error (code $status); restarting in 3 s. Ctrl+C to stop. Details: $LOG"
    sleep 3
done
