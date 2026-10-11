import json
from pathlib import Path
from queue import Queue
import tempfile
import unittest
from unittest.mock import Mock

from followme.phase2.cameras import Capture
from followme.phase2.evaluation import replay
from followme.phase2.tracking import Tracker, Zones
from test_phase2 import result, settings, zones


class ReplayAndCaptureTests(unittest.TestCase):
    def test_replay_matches_logged_states_and_transitions(self):
        cfg=settings()
        tracker=Tracker(cfg,zones())
        events=[{'type':'session','logged_monotonic_ns':0,'config':cfg,'zones':zones().config}]
        locations=[.2]*30+[.3,.4,.5,.6,.7]+[.7]*30
        for seq,x in enumerate(locations):
            stamp=1+seq*.1
            reply=result(seq,stamp,x)
            now=int(stamp*1e9)+1
            tracker.update(reply,now)
            events.append({'type':'observation','logged_monotonic_ns':now,'processed_monotonic_ns':now,'reply':reply})
            events.append({'type':'state','logged_monotonic_ns':now,'state':tracker.snapshot(now)})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observations.jsonl'
            path.write_text(''.join(json.dumps(e)+'\n' for e in events))
            report=replay(path)
        self.assertTrue(report['replay_matches'])
        self.assertEqual([t['zone'] for t in report['accepted_zones']],['station_a','station_b'])
        self.assertIn('pending',report['live_acceptance'])

    def test_capture_stall_terminates_only_its_process_without_blocking_monitor(self):
        capture=Capture.__new__(Capture)
        capture.settings={'capture_stall_s':1}
        capture.frames=Queue();capture.events=Queue()
        capture.latest={'sequence':1,'capture_monotonic_ns':1_000_000_000}
        capture.started=0;capture.error=None;capture.metadata={}
        capture.stopped=Mock();capture.process=Mock()
        capture.process.is_alive.return_value=True
        self.assertIsNone(capture.poll(2_100_000_000))
        capture.process.terminate.assert_called_once()
        capture.process.join.assert_not_called()

    def test_latest_capture_replaces_a_consumed_sequence(self):
        capture=Capture.__new__(Capture)
        capture.settings={'capture_stall_s':1}
        capture.frames=Queue();capture.events=Queue()
        capture.latest=None;capture.started=0;capture.error=None;capture.metadata={}
        capture.process=Mock();capture.process.is_alive.return_value=True
        for seq in (1,2):capture.frames.put({'sequence':seq,'capture_monotonic_ns':1_000_000_000})
        self.assertEqual(capture.poll(1_100_000_000)['sequence'],2)


if __name__=='__main__':unittest.main()
