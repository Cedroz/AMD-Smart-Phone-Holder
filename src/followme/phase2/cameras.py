"""Single-owner capture processes with bounded latest-frame IPC and stall cleanup."""
import multiprocessing as mp
from queue import Empty, Full
import time


def capture(path, role, settings, frames, events, stopped):
    import cv2
    proxy = path.startswith(('http://','https://'))
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG if proxy else cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            events.put({'error': 'Camera cannot be opened'})
            return
        if not proxy:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['capture_width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['capture_height'])
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FPS, settings['capture_fps'])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        events.put({'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    'reported_fps': cap.get(cv2.CAP_PROP_FPS), 'source':path,
                    'timestamp_kind':'proxy_delivery' if proxy else 'device_acquisition',
                    'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4,'little').decode(errors='replace')})
        seq = 0
        while not stopped.is_set():
            ok, image = cap.read()
            stamp = time.monotonic_ns()
            if not ok:
                time.sleep(.02)
                continue
            expected=(settings['capture_height'],settings['capture_width'])
            if not proxy and image.shape[:2]!=expected:
                events.put({'error':'Camera returned unexpected dimensions; zone metadata would be invalid'})
                return
            angle = settings['rotation'][role]
            if angle:
                image = cv2.rotate(image, {90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
            ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY,80])
            if not ok:
                continue
            seq += 1
            item = {'sequence': seq, 'capture_monotonic_ns': stamp, 'jpeg': jpeg.tobytes()}
            try:
                frames.put_nowait(item)
            except Full:
                try:
                    frames.get_nowait()
                except Empty:
                    pass
                try:
                    frames.put_nowait(item)
                except Full:
                    pass
    finally:
        cap.release()


class Capture:
    def __init__(self, path, role, settings):
        ctx = mp.get_context('spawn')
        self.role, self.settings = role, settings
        self.frames, self.events = ctx.Queue(maxsize=1), ctx.Queue(maxsize=4)
        self.stopped = ctx.Event()
        self.process = ctx.Process(target=capture,args=(path,role,settings,self.frames,self.events,self.stopped),daemon=True)
        self.latest = None
        self.metadata = {}
        self.error = None
        self.started = time.monotonic_ns()
        self.process.start()

    def poll(self, now):
        try:
            while True:
                self.metadata.update(self.events.get_nowait())
        except Empty:
            pass
        try:
            while True:
                self.latest = self.frames.get_nowait()
        except Empty:
            pass
        if self.error:
            return None
        reference = self.latest['capture_monotonic_ns'] if self.latest else self.started
        # Startup allows process spawn/driver initialization; steady-state is 1s.
        limit = self.settings['capture_stall_s'] if self.latest else 5
        if not self.process.is_alive() or now-reference > limit*1e9 or self.metadata.get('error'):
            self.error = self.metadata.get('error', 'Capture stopped or stalled; restart perception explicitly')
            self.stopped.set()
            if self.process.is_alive(): self.process.terminate()
            return None
        return self.latest

    def close(self):
        self.stopped.set()
        self.process.join(timeout=.1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=.5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=.5)
        for queue in (self.frames, self.events):
            queue.cancel_join_thread()

