"""Phase 5 single-camera ownership; preserves other phases' configuration."""
import threading

from followme.environment import preflight
from followme.phase2.cameras import Capture
from followme.phase2.runtime import Runtime


class SingleCameraPerception(Runtime):
    def start(self):
        device = preflight(self.root)['devices']['external']
        if device['owners'] and not self.preview_sources:
            raise RuntimeError('Tracking camera busy; use diagnostic proxy mode')
        if not device['present'] and not self.preview_sources:
            raise RuntimeError('Tracking camera missing')
        self.client.start()
        try:
            proxy = self.preview_sources and (device['owners'] or not device['present'])
            self.source_modes['external'] = 'legacy_preview_proxy' if proxy else 'native_camera'
            source = 'http://127.0.0.1:8000/' if proxy else self.config['cameras']['external']
            self.captures['external'] = Capture(source, 'external', self.settings)
            for function in (self.monitor, self.inference):
                thread = threading.Thread(target=self.worker, args=(function,), daemon=True)
                self.threads.append(thread)
                thread.start()
        except BaseException:
            self.close()
            raise
