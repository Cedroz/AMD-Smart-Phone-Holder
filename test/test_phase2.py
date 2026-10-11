import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from followme.phase2.tracking import Tracker, Zones, validate_settings

ROOT=Path(__file__).resolve().parents[1]


def settings():
    import yaml
    return yaml.safe_load((ROOT/'config/phase2/perception.yaml').read_text())


def zones():
    return Zones({'schema_version':1,'status':'calibrated','margin':.03,
                  'regions':{'station_a':[.05,.1,.4,.9],'station_b':[.6,.1,.95,.9]}})


def result(seq, stamp, x=.2, count=1):
    return {'camera_role':'external','sequence':seq,'capture_monotonic_ns':int(stamp*1e9),
            'face_count':count,'landmarks':[[{'x':x-.02,'y':.4},{'x':x+.02,'y':.6}]] if count else [],'error':None}


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.tracker=Tracker(settings(),zones())

    def feed(self,seq,stamp,x=.2,count=1):
        return self.tracker.update(result(seq,stamp,x,count),int(stamp*1e9)+1)

    def acquired(self):
        for i in range(20): self.feed(i,1+i*.1)
        self.assertTrue(self.tracker.state.tracking_valid)
        self.assertEqual(self.tracker.state.stable_zone,'station_a')

    def test_acquisition_and_dwell(self):
        for i in range(7): self.feed(i,1+i*.1)
        self.assertFalse(self.tracker.state.tracking_valid)
        self.acquired()

    def test_ema(self):
        self.feed(0,1,.2)
        state=self.feed(1,1.1,.3)
        self.assertAlmostEqual(state.face_center_filtered[0],.225)

    def test_duplicate_and_wrong_role_cannot_overwrite(self):
        self.acquired()
        previous=self.tracker.state
        self.feed(1,1.1,.7)
        self.assertEqual(previous,self.tracker.state)
        reply=result(100,3,.8);reply['camera_role']='wrist'
        self.tracker.update(reply,3_000_000_000)
        self.assertEqual(previous,self.tracker.state)

    def test_future_and_stale_results(self):
        self.acquired()
        state=self.tracker.update(result(30,4),3_000_000_000)
        self.assertFalse(state.tracking_valid)
        self.assertEqual(state.invalid_reason,'future_timestamp')
        state=self.tracker.update(result(31,3),4_000_000_000)
        self.assertFalse(state.tracking_valid)

    def test_stale_tick_clears_selection_without_an_inference_reply(self):
        self.acquired()
        state=self.tracker.tick(3_500_000_000)
        self.assertTrue(state.selection_required)
        self.assertIsNone(state.stable_zone)

    def test_ambiguity_requires_explicit_reacquire(self):
        self.acquired()
        state=self.feed(20,3,count=2)
        self.assertTrue(state.ambiguous)
        state=self.feed(21,3.1)
        self.assertFalse(state.tracking_valid)
        self.assertTrue(state.selection_required)
        self.tracker.reacquire()
        for i in range(22,40): self.feed(i,3.2+(i-22)*.1)
        self.assertEqual(self.tracker.state.track_generation,2)
        self.assertTrue(self.tracker.state.tracking_valid)

    def test_missing_face_resets_zone_dwell(self):
        self.acquired()
        state=self.feed(20,3,count=0)
        self.assertIsNone(state.stable_zone)
        self.assertIsNone(state.face_center_filtered)
        self.assertFalse(state.tracking_valid)

    def test_continuity_jump_does_not_select_another_person(self):
        self.acquired()
        state=self.feed(20,3,.8)
        self.assertEqual(state.invalid_reason,'continuity_break')
        self.assertTrue(state.selection_required)

    def test_malformed_landmarks(self):
        for value in (float('nan'),float('inf'),-1,2,True):
            reply=result(1,1);reply['landmarks'][0][0]['x']=value
            state=Tracker(settings(),zones()).update(reply,1_000_000_001)
            self.assertEqual(state.invalid_reason,'invalid_landmarks')

    def test_neutral_then_second_zone_requires_dwell(self):
        self.acquired()
        for i,x in enumerate([.3,.4,.5,.6,.7],20):self.feed(i,3+(i-20)*.1,x)
        self.assertNotEqual(self.tracker.state.stable_zone,'station_b')
        for i in range(25,48):self.feed(i,3.5+(i-25)*.1,.7)
        self.assertEqual(self.tracker.state.stable_zone,'station_b')

    def test_boundary_jitter_retains_current_zone(self):
        self.acquired()
        self.feed(20,3,.3)
        for i in range(21,40):self.feed(i,3+(i-20)*.1,.4 if i%2 else .41)
        self.assertEqual(self.tracker.state.stable_zone,'station_a')

    def test_unconfigured_zones_never_choose_a_destination(self):
        self.tracker.zones=Zones({'schema_version':1,'status':'pending','regions':{},'margin':.03})
        for i in range(30):self.feed(i,1+i*.1)
        self.assertTrue(self.tracker.state.tracking_valid)
        self.assertIsNone(self.tracker.state.stable_zone)

    def test_shoulders_do_not_silently_fall_back_to_face(self):
        cfg=settings();cfg.update(position_source='shoulders',pose_model_path='model',pose_model_manifest='manifest')
        self.tracker=Tracker(cfg,zones())
        for i in range(20):self.feed(i,1+i*.1)
        self.assertFalse(self.tracker.state.tracking_valid)
        self.assertEqual(self.tracker.state.invalid_reason,'missing_shoulders')


class ConfigurationTests(unittest.TestCase):
    def test_invalid_regions(self):
        for regions in ({'station_a':[0,0,.6,1],'station_b':[.5,0,1,1]},
                        {'station_a':[0,0,0,1],'station_b':[.6,0,1,1]}):
            with self.assertRaises(ValueError):Zones({'schema_version':1,'status':'calibrated','regions':regions})

    def test_invalid_settings(self):
        for key,value in [('alpha',0),('face_threshold',float('nan')),('external_hz',-1),('position_source','unknown')]:
            cfg=settings();cfg[key]=value
            with self.assertRaises(ValueError):validate_settings(cfg)

    def test_orientation_metadata(self):
        cfg=zones().config
        with self.assertRaises(ValueError):Zones(cfg,'wrong-path',settings())

    def test_package_and_replay_import_do_not_open_hardware(self):
        with patch('serial.Serial',side_effect=AssertionError('Serial accessed')),patch('cv2.VideoCapture',side_effect=AssertionError('Camera accessed')):
            from followme.phase2.runtime import Runtime
            from followme.phase2.evaluation import replay
            from followme.phase2.web import app_for
            # Constructor preflight uses filesystem checks, not hardware opens.
            with tempfile.TemporaryDirectory() as directory:
                runtime=Runtime(ROOT,directory)
                runtime.reacquire()
                runtime.status()
                app=app_for(runtime,'external')
                self.assertNotIn('/api/mode',{route.path for route in app.routes})
                runtime.close()


if __name__=='__main__':unittest.main()
