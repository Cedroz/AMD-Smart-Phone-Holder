"""Independent Phase 5 acceptance using controlled time and fake hardware."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from followme.phase5.config import load_config, inspect, verify_approvals
from followme.phase5.planner import Integration, observation
from followme.phase5.replay import Clock, run
from followme.phase5.runtime import synthetic, Runtime
from followme.phase5.web import app_for
from followme.robotics.adapters import FakeAdapter
from followme.robotics.controller import Controller
from followme.robotics.trajectories import fake_library

ROOT = Path(__file__).resolve().parents[1]


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.config = load_config(ROOT)
        self.adapter = FakeAdapter(self.clock)
        self.controller = Controller(self.adapter, fake_library(), clock=self.clock)
        self.integration = Integration(self.controller, self.config, self.clock)
        self.controller.tick()
        self.seq = 0
        self.since = self.clock()
        self.step('station_a', advance=False)
        self.integration.heartbeat('test')
        self.integration.set_mode('follow', 'test')
        self.controller.tick()

    def tearDown(self):
        self.controller.close()

    def step(self, zone='station_b', advance=True, change=None):
        self.seq += 1
        s = synthetic(self.clock(), self.seq, zone, self.since)
        if change: change(s)
        self.integration.heartbeat('test')
        self.integration.tick(s)
        self.controller.tick()
        if advance: self.clock.advance()
        return s

    def move(self):
        for _ in range(20): self.step()
        self.assertEqual(self.controller.status().state, 'REPOSITIONING')

    def test_dwell_threshold_and_single_request(self):
        for _ in range(15): self.step()
        self.assertEqual(len(self.integration.decisions), 0)
        self.step()
        self.assertEqual(len(self.integration.decisions), 1)
        for _ in range(30): self.step()
        self.assertEqual(len(self.integration.decisions), 1)

    def test_same_pose_never_repositions(self):
        for _ in range(2400): self.step('station_a')
        self.assertFalse(self.integration.decisions)

    def test_neutral_and_jitter_reset_dwell(self):
        for _ in range(100):
            self.step('station_b')
            self.step(None)
        self.assertFalse(self.integration.decisions)

    def test_duplicate_does_not_advance_dwell(self):
        snapshot = self.step(advance=False)
        for _ in range(8):
            self.clock.advance()
            self.integration.tick(snapshot)
            self.controller.tick()
        self.assertFalse(self.integration.decisions)

    def test_tracking_failure_holds_no_auto_resume(self):
        self.move()
        self.step(change=lambda s:s['state'].update(tracking_valid=False))
        self.assertEqual(self.integration.mode, 'hold')
        for _ in range(20): self.step()
        self.assertEqual(self.integration.mode, 'hold')

    def test_ambiguity_interrupts(self):
        self.move()
        self.step(change=lambda s:s['state'].update(ambiguous=True))
        self.assertFalse(self.controller.status().motion_enabled)

    def test_stale_observation_interrupts(self):
        self.move()
        self.step(change=lambda s:s['state'].update(capture_monotonic_ns=int((self.clock()-1)*1e9)))
        self.assertEqual(self.integration.mode, 'hold')

    def test_future_frame_rejected(self):
        self.step(change=lambda s:s['state'].update(capture_monotonic_ns=int((self.clock()+1)*1e9)))
        self.assertEqual(self.integration.last_reason, 'stale_or_future_observation')

    def test_generation_change_requires_follow(self):
        self.step(change=lambda s:s['state'].update(track_generation=1))
        self.assertEqual(self.integration.mode, 'hold')

    def test_regressing_sequence_rejected(self):
        self.step()
        self.step(change=lambda s:s['state'].update(sequence=0))
        self.assertEqual(self.integration.last_reason, 'regressing_sequence')

    def test_worker_failure_interrupts(self):
        self.move()
        self.step(change=lambda s:s.update(worker_error='crashed'))
        self.assertEqual(self.integration.mode, 'hold')

    def test_heartbeat_expiry_releases_owner(self):
        self.clock.now += 3.01
        self.integration.tick(synthetic(self.clock(),99,'station_b',self.since))
        self.controller.tick()
        self.assertEqual(self.integration.mode,'hold')
        self.integration.heartbeat('new-client')
        self.assertEqual(self.integration.owner,'new-client')

    def test_competing_operator_rejected(self):
        with self.assertRaisesRegex(ValueError,'owned'):
            self.integration.heartbeat('other')

    def test_unknown_anchor_returns_hold(self):
        self.adapter.positions['shoulder_pan']=5
        self.controller.tick()
        self.step()
        self.assertEqual(self.integration.mode,'hold')

    def test_controller_inhibit_never_auto_resumes(self):
        self.controller.request_hold('independent_watchdog')
        self.controller.tick()
        for _ in range(5):self.step()
        self.assertEqual(self.integration.mode,'hold')

    def test_stop_priority_and_latch(self):
        self.move()
        self.integration.stop()
        self.controller.tick()
        self.assertEqual(self.controller.status().state,'STOPPED')
        with self.assertRaisesRegex(ValueError,'healthy_anchor'):
            self.integration.set_mode('follow','test')

    def test_hold_stop_early_middle_late(self):
        for kind in ['hold','stop']:
            for fraction in [.1,.5,.8]:
                with self.subTest(kind=kind,fraction=fraction):
                    self.move()
                    start=self.clock()
                    while self.clock()-start < 3*fraction:self.step()
                    if kind=='hold':self.integration.hold('trial')
                    else:self.integration.stop()
                    self.controller.tick()
                    self.assertEqual(self.controller.status().state,'HOLDING' if kind=='hold' else 'STOPPED')
                    count=self.adapter.simulated_commands
                    for _ in range(3):self.step()
                    self.assertEqual(self.adapter.simulated_commands,count)
                    self.assertEqual(self.integration.mode,'hold')
                self.controller.close()
                self.setUp()

    def test_reset_returns_hold(self):
        self.integration.stop()
        self.controller.tick()
        self.integration.reset()
        self.controller.tick()
        self.assertEqual(self.integration.mode,'hold')
        self.assertFalse(self.controller.status().motion_enabled)

    def test_reset_midpath_requires_supervised_restore(self):
        for _ in range(46): self.step()
        self.integration.stop()
        self.controller.tick()
        with self.assertRaisesRegex(ValueError,'restoration'):
            self.integration.reset()

    def test_cooldown_starts_at_arrival(self):
        self.move()
        while self.controller.status().state == 'REPOSITIONING': self.step()
        self.step()
        arrived = self.clock()
        self.assertGreater(self.integration.cooldown_until, arrived+1.8)
        self.since = self.clock()
        for _ in range(30): self.step('station_a')
        self.assertEqual(len(self.integration.decisions),1)

    def test_no_queued_opposite_request(self):
        self.move()
        for _ in range(30): self.step('station_a')
        self.assertEqual(len(self.integration.decisions),1)

    def test_missing_route_fails_closed(self):
        self.controller.library.data['transitions'].clear()
        for _ in range(20): self.step()
        self.assertEqual(self.integration.mode,'hold')
        self.assertFalse(self.integration.decisions)

    def test_malformed_zone_fails_closed(self):
        self.step(change=lambda s:s['state'].update(candidate_zone=[]))
        self.assertEqual(self.integration.last_reason,'invalid_zone_type')

    def test_live_proxy_rejected_by_bridge(self):
        s=synthetic(self.clock(),100,'station_a')
        s['source_modes']['external']='legacy_preview_proxy'
        with self.assertRaisesRegex(ValueError,'native_capture'):
            observation(s,self.clock(),.5,True)

    def test_rejected_request_consumed(self):
        from followme.robotics.contracts import RequestResult
        with patch.object(self.controller,'request_safe_pose',return_value=RequestResult(False,'busy_worker')) as method:
            for _ in range(30): self.step()
            self.assertEqual(method.call_count,1)
            self.assertEqual(self.integration.mode,'hold')

    def test_api_auth_controls_and_no_joint_endpoint(self):
        from fastapi.testclient import TestClient
        outer=self
        class Stub:
            integration=outer.integration
            def status(self): return self.integration.status()
        client=TestClient(app_for(Stub(), self.config,token='test-token'))
        headers={'origin':'http://127.0.0.1:8020','authorization':'Bearer test-token','x-operator-id':'test'}
        self.assertEqual(client.post('/api/stop').status_code,403)
        self.assertEqual(client.post('/api/stop',headers={'origin':headers['origin']}).status_code,401)
        self.assertEqual(client.post('/api/stop',headers=headers).status_code,200)
        self.controller.tick()
        self.assertEqual(client.get('/api/status').json()['state'],'STOPPED')
        self.assertEqual(client.post('/api/joints',headers=headers,json={'shoulder_pan':1}).status_code,404)
        self.assertEqual(client.post('/api/mode',headers=headers,json={'mode':'follow','joints':{}}).status_code,409)
        self.assertEqual(client.post('/api/reset',headers=headers).status_code,200)
        self.controller.tick()
        self.assertEqual(self.integration.mode,'hold')

    def test_websocket_telemetry_does_not_renew_lease(self):
        from fastapi.testclient import TestClient
        outer=self
        class Stub:
            integration=outer.integration
            def status(self):return self.integration.status()
        client=TestClient(app_for(Stub(),self.config,token='test-token'))
        lease=self.integration.lease
        with client.websocket_connect('/ws/state',headers={'origin':'http://127.0.0.1:8020'},subprotocols=['followme','test-token']) as ws:
            self.assertEqual(ws.receive_json()['backend'],'fake')
        self.assertEqual(lease,self.integration.lease)

    def test_camera_panels_and_viewing_only_sources(self):
        from fastapi.testclient import TestClient
        outer=self
        class Stub:
            provider=None
            integration=outer.integration
            def status(self):return self.integration.status()
        client=TestClient(app_for(Stub(),self.config,token='test-token'))
        self.assertIn('Robot-mounted tracking camera',client.get('/').text)
        self.assertNotIn('id="wrist-camera"',client.get('/').text)
        info=client.get('/api/cameras').json()
        self.assertIn('simulated',info['note'])
        self.assertEqual(info['roles']['external'][0]['path'],'/camera-feeds/external')
        self.assertEqual(set(info['roles']),{'external'})
        self.assertEqual(client.get('/streams/external').status_code,503)
        self.assertEqual(client.get('/streams/unknown').status_code,404)

    def test_camera_proxy_falls_back_without_changing_operator_lease(self):
        from fastapi.testclient import TestClient
        import io
        import threading
        outer=self
        class Stub:
            provider=None
            stopped=threading.Event()
            integration=outer.integration
            def status(self):return self.integration.status()
        upstream=io.BytesIO(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8image\xff\xd9\r\n')
        upstream.headers={'Content-Type':'multipart/x-mixed-replace; boundary=frame'}
        client=TestClient(app_for(Stub(),self.config,token='test-token'))
        lease=self.integration.lease
        with patch('followme.phase5.web.urllib.request.urlopen',side_effect=[OSError('stale'),upstream]) as connect:
            response=client.get('/camera-feeds/external')
        self.assertEqual(response.status_code,200)
        self.assertIn(b'\xff\xd8image',response.content)
        self.assertIn('127.0.0.1:8000/',connect.call_args.args[0])
        self.assertTrue(upstream.closed)
        self.assertEqual(lease,self.integration.lease)
        self.assertEqual(client.get('/camera-feeds/unknown').status_code,404)
        snapshot=io.BytesIO(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8image\xff\xd9\r\n')
        snapshot.headers=upstream.headers
        with patch('followme.phase5.web.urllib.request.urlopen',return_value=snapshot):
            response=client.get('/camera-frames/external')
        self.assertEqual(response.headers['content-type'],'image/jpeg')
        self.assertEqual(response.content,b'\xff\xd8image\xff\xd9')
        self.assertTrue(snapshot.closed)

    def test_owned_camera_stream_uses_existing_preview_without_heartbeat(self):
        from fastapi.testclient import TestClient
        import threading
        outer=self
        class Provider:
            def __init__(self):self.calls=0
            def preview(self,role):
                self.calls+=1
                return (7,b'\xff\xd8preview\xff\xd9') if self.calls<=2 else None
        class Stub:
            provider=Provider()
            stopped=threading.Event()
            integration=outer.integration
            def status(self):return self.integration.status()
        client=TestClient(app_for(Stub(),self.config,token='test-token'))
        lease=self.integration.lease
        response=client.get('/streams/external')
        self.assertEqual(response.status_code,200)
        self.assertIn('multipart/x-mixed-replace',response.headers['content-type'])
        self.assertIn(b'\xff\xd8preview',response.content)
        self.assertEqual(lease,self.integration.lease)


class EnvironmentTests(unittest.TestCase):
    def test_replay_input_requires_explicit_follow(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.jsonl'
            p.write_text('\n'.join(json.dumps({'timestamp_s':100+i*.1,
                'snapshot':synthetic(100+i*.1,i,'station_b',100)}) for i in range(20)))
            result=run(load_config(ROOT),Path(d)/'out',p)
            self.assertFalse(result['decisions'])

    def test_approved_flags_cannot_replace_physical_evidence(self):
        from unittest.mock import patch
        config=load_config(ROOT)
        config.update(approved_session='draft.yaml',perception_acceptance='acceptance.json')
        with patch('followme.phase4.library.load',return_value={'validation':{'status':'draft','approved':True}}):
            with self.assertRaisesRegex(ValueError,'not_approved'):
                verify_approvals(ROOT,config)

    def test_approved_session_revalidates_evidence(self):
        config=load_config(ROOT)
        config.update(approved_session='approved.yaml',perception_acceptance='acceptance.json')
        with patch('followme.phase4.library.load',return_value={'validation':{'status':'approved'}}),patch(
                'followme.phase4.library.approve',side_effect=ValueError('changed_evidence_log')) as approval:
            with self.assertRaisesRegex(ValueError,'changed_evidence'):
                verify_approvals(ROOT,config)
            self.assertEqual(approval.call_count,1)
    def test_inspect_never_constructs_live_adapter(self):
        with patch('followme.robotics.adapters.LeRobotAdapter.connect',side_effect=AssertionError('serial')):
            report=inspect(ROOT,load_config(ROOT))
            self.assertFalse(report['hardware_opened'])
            self.assertFalse(report['live_ready'])

    def test_live_without_approval_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'approval_paths'):
                Runtime(ROOT,load_config(ROOT),live=True,perception='native',output=d)

    def test_dry_run_synthetic_does_not_import_perception(self):
        with tempfile.TemporaryDirectory() as d, patch('followme.robotics.adapters.LeRobotAdapter.connect',side_effect=AssertionError('serial')):
            r=Runtime(ROOT,load_config(ROOT),output=d)
            r.start()
            r.close()
            self.assertEqual(r.controller.adapter.physical_commands,0)
            self.assertIsNone(r.provider)

    def test_ten_alternating_trials(self):
        with tempfile.TemporaryDirectory() as d:
            result=run(load_config(ROOT),d)
            self.assertEqual(result['correct_destinations'],10)
            self.assertEqual(len(result['decisions']),10)
            self.assertEqual(result['physical_commands'],0)
            self.assertTrue(result['single_io_owner'])

    def test_weakened_config_rejected(self):
        import yaml
        for field,value in [('dwell_s',0),('cooldown_s',.1),('freshness_s',1),('heartbeat_s',4),('mode','follow')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                data=load_config(ROOT);data[field]=value
                p=Path(d)/'config.yaml';p.write_text(yaml.safe_dump(data))
                with self.assertRaises(ValueError):load_config(ROOT,p)


if __name__=='__main__':unittest.main()
