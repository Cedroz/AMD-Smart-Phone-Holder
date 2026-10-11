"""Send live speaker identification to the tracker: one POST /voice per hop.

tracker/track.py --voice NAME starts this automatically with the speakerid Python. It can also be
run by hand while the tracker is up:

    /home/aup/miniforge3/envs/speakerid/bin/python voice_bridge.py --tracker-url http://127.0.0.1:8000
"""

import argparse
import json
import time
import urllib.error
import urllib.request

import config
from live import speaker_stream


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1.0) as response:
        response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker-url", default="http://127.0.0.1:8000")
    parser.add_argument("--device", type=int, default=config.MIC_DEVICE_INDEX)
    mic_group = parser.add_mutually_exclusive_group()
    mic_group.add_argument("--local-mic", dest="local_mic", action="store_true", help="capture from this host")
    mic_group.add_argument("--forwarded-mic", dest="local_mic", action="store_false", help="capture from the forwarded laptop mic")
    parser.set_defaults(local_mic=config.MIC_INPUT_MODE == "local")
    args = parser.parse_args()

    url = args.tracker_url.rstrip("/") + "/voice"
    last_error, last_print = None, None
    try:
        for speaker, confidence, _, level in speaker_stream(args.device, local_mic=args.local_mic, levels=True):
            try:
                post(url, {"speaker": speaker, "confidence": round(float(confidence), 3), "level": round(level, 1)})
                last_error = None
            except (urllib.error.URLError, OSError) as error:
                if str(error) != last_error:  # the tracker may be restarting; say so once
                    print(f"Tracker not reachable at {url}: {error}", flush=True)
                    last_error = str(error)
            if speaker != last_print:
                print(f"{time.strftime('%H:%M:%S')} {speaker} ({confidence:.2f})", flush=True)
                last_print = speaker
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
