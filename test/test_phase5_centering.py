import unittest

from followme.phase5.centering import CenteringController, CenteringIntegration
from followme.phase5.replay import Clock
from followme.phase5.runtime import synthetic
from followme.robotics.adapters import FakeAdapter
from followme.robotics.trajectories import fake_library
from followme.robotics.contracts import Limits
from types import SimpleNamespace


class CenteringTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.adapter = FakeAdapter(self.clock)
        self.controller = CenteringController(self.adapter, fake_library(), clock=self.clock)
        self.integration = CenteringIntegration(self.controller, {'heartbeat_s':3}, clock=self.clock)
        self.controller.tick()
        self.frame(.8)
        self.integration.heartbeat('operator')
        self.integration.set_mode('follow', 'operator')
        self.controller.tick()

    def frame(self, x, valid=True):
        snapshot = synthetic(self.clock(), int(self.clock()*1000))
        snapshot['state'].update(face_center_filtered=[x,.5], tracking_valid=valid)
        self.integration.tick(snapshot)

    def step(self, x):
        self.clock.advance()
        self.frame(x)
        self.controller.tick()

    def test_direction_and_bounds(self):
        self.step(.9)
        self.assertGreater(self.adapter.positions['shoulder_pan'],0)
        self.assertLessEqual(self.adapter.positions['shoulder_pan'],.05)
        self.assertEqual(self.adapter.physical_commands,0)
        self.assertEqual(self.adapter.positions['gripper'],0)

    def test_deadband(self):
        self.step(.53)
        self.assertEqual(self.adapter.simulated_commands,0)

    def test_hold_beats_pending_follow(self):
        self.controller.request_hold()
        self.step(.9)
        self.assertEqual(self.adapter.simulated_commands,0)
        self.assertEqual(self.controller.status().mode,'hold')

    def test_invalid_tracking_holds(self):
        self.clock.advance()
        self.frame(.9,False)
        self.controller.tick()
        self.assertEqual(self.integration.mode,'hold')
        self.assertEqual(self.adapter.simulated_commands,0)

    def test_stale_frame_and_heartbeat(self):
        self.clock.now += 3.1
        self.controller.tick()
        self.assertEqual(self.controller.status().mode,'hold')
        self.assertEqual(self.adapter.simulated_commands,0)

    def test_stop_latches(self):
        self.integration.stop()
        self.step(.9)
        with self.assertRaisesRegex(ValueError,'latched'):
            self.integration.set_mode('follow','operator')
        self.assertEqual(self.adapter.simulated_commands,0)


class LivePolicyTests(unittest.TestCase):
    """Exercise live gates with fake I/O; no serial device is constructed."""
    def build(self):
        clock = Clock()
        class TrialFake(FakeAdapter):
            backend = 'live'
            def arm(self):self.armed=True
            def hold(self):self.call('hold')
        adapter = TrialFake(clock)
        adapter.positions['shoulder_pan']=15.
        library=SimpleNamespace(backend='live',limits=Limits(velocity=1.,acceleration=2.))
        controller=CenteringController(adapter,library,library.limits,clock=clock,bounds=[14.68,16.71])
        integration=CenteringIntegration(controller,{'heartbeat_s':3},clock=clock)
        controller.tick()
        return clock,adapter,controller,integration

    def frame(self,clock,adapter,integration,slope):
        snapshot=synthetic(clock(),int(clock()*1000))
        snapshot['source_modes']['external']='native_camera'
        snapshot['state']['face_center_filtered']=[.5+slope*(adapter.positions['shoulder_pan']-15),.5]
        integration.tick(snapshot)
        integration.heartbeat('operator')

    def test_live_follow_requires_direction_calibration(self):
        clock,adapter,controller,integration=self.build()
        self.frame(clock,adapter,integration,-.03)
        with self.assertRaisesRegex(ValueError,'calibrate_camera'):
            integration.set_mode('follow','operator')
        self.assertEqual(adapter.simulated_commands,0)

    def test_live_calibration_measures_both_response_signs(self):
        for slope,sign in [(-.03,1),(.03,-1)]:
            clock,adapter,controller,integration=self.build()
            self.frame(clock,adapter,integration,slope)
            integration.calibrate('operator')
            for _ in range(200):
                clock.advance()
                self.frame(clock,adapter,integration,slope)
                controller.tick()
            self.assertEqual(controller.response_sign,sign)
            self.assertEqual(controller.calibration_result['status'],'passed')
            self.assertEqual(controller.status().mode,'hold')
            self.assertLessEqual(adapter.positions['shoulder_pan'],16.71)

    def test_live_proxy_observations_cannot_start_calibration(self):
        clock,adapter,controller,integration=self.build()
        snapshot=synthetic(clock(),1)
        snapshot['state']['face_center_filtered']=[.5,.5]
        integration.tick(snapshot)
        integration.heartbeat('operator')
        with self.assertRaisesRegex(ValueError,'native_capture'):
            integration.calibrate('operator')

    def test_live_adapter_retains_robot_and_only_commands_pan(self):
        from unittest.mock import patch
        from followme.phase5.live_pan import LivePanAdapter
        class Bus:
            is_connected=False
            is_calibrated=True
            calibration={'shoulder_pan':SimpleNamespace(range_min=-6,range_max=8),
                         'wrist_flex':SimpleNamespace(range_min=-4,range_max=5)}
            motors={'shoulder_pan':SimpleNamespace(id=1),'wrist_flex':SimpleNamespace(id=2)}
            def __init__(self):self.writes=[];self.goal=0
            def connect(self):self.is_connected=True
            def sync_read(self,name):
                return dict.fromkeys(['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'],0.)
            def read(self,name,motor,**kwargs):return self.goal if name=='Goal_Position' else 0
            def write(self,name,motor,value,**kwargs):
                self.writes.append((name,motor,value))
                if name=='Goal_Position':self.goal=value
            def enable_torque(self,motor):self.writes.append(('Torque_Enable',motor,1))
            def _unnormalize(self,values):return {1:round(values[1]*10)}
            def _normalize(self,values):return {key:value/10 for key,value in values.items()}
            def disconnect(self,disable_torque):self.is_connected=False;self.retained=not disable_torque
        class Robot:
            def __init__(self,*args):self.bus=Bus()
            def __del__(self):self.bus.is_connected=False
        adapter=LivePanAdapter([-1,1],relative_window=2)
        adapter.bind_owner()
        with patch('lerobot.robots.so_follower.SO101Follower',Robot),patch('followme.phase5.live_pan.subprocess.run',return_value=SimpleNamespace(returncode=1,stdout='')),patch('followme.phase5.live_pan.fcntl.flock'):
            adapter.connect()
        self.assertIsNotNone(adapter.robot)
        self.assertTrue(adapter.bus.is_connected)
        adapter.arm()
        adapter.send({'shoulder_pan':.5})
        self.assertTrue(all(motor=='shoulder_pan' for _,motor,_ in adapter.bus.writes))
        with self.assertRaisesRegex(RuntimeError,'inhibited'):
            adapter.send({'shoulder_pan':2.1})
        adapter.disconnect()
        self.assertTrue(adapter.bus.retained)
        expanded=LivePanAdapter([-1,1],relative_window=90,tilt_window=25,calibrated_bounds=True)
        expanded.bind_owner()
        with patch('lerobot.robots.so_follower.SO101Follower',Robot),patch('followme.phase5.live_pan.subprocess.run',return_value=SimpleNamespace(returncode=1,stdout='')),patch('followme.phase5.live_pan.fcntl.flock'):
            expanded.connect()
        self.assertEqual(expanded.bounds,[-.6,.8])
        self.assertEqual(expanded.joint_bounds['wrist_flex'],[-.4,.5])
        self.assertEqual(expanded.physical_commands,0)
        expanded.disconnect()


class XYTests(unittest.TestCase):
    def run_target(self,point):
        from followme.phase5.track_xy import XYController
        clock=Clock()
        adapter=FakeAdapter(clock)
        controller=XYController(adapter,fake_library(),clock=clock)
        integration=CenteringIntegration(controller,{'heartbeat_s':3},clock=clock)
        controller.tick()
        def observe():
            snapshot=synthetic(clock(),int(clock()*1000))
            snapshot['state'].update(face_center_filtered=None,tracking_center=point,target_kind='person')
            integration.tick(snapshot)
            integration.heartbeat('operator')
        observe()
        integration.set_mode('follow','operator')
        controller.tick()
        for _ in range(20):
            clock.advance();observe();controller.tick()
        return adapter,controller,integration

    def test_body_target_moves_both_axes(self):
        adapter,controller,integration=self.run_target([.8,.8])
        self.assertGreater(adapter.positions['shoulder_pan'],0)
        self.assertGreater(adapter.positions['wrist_flex'],0)
        self.assertEqual(adapter.positions['gripper'],0)
        self.assertEqual(adapter.positions['elbow_flex'],0)
        self.assertEqual(controller.status().mode,'follow')

    def test_vertical_deadband_leaves_wrist_fixed(self):
        adapter,controller,integration=self.run_target([.8,.52])
        self.assertGreater(adapter.positions['shoulder_pan'],0)
        self.assertEqual(adapter.positions['wrist_flex'],0)
        integration.reverse('horizontal')
        controller.tick()
        self.assertEqual(controller.status().mode,'hold')
        self.assertEqual(controller.response_sign,-1)

    def test_brief_target_loss_keeps_follow_and_reacquires(self):
        adapter,controller,integration=self.run_target([.8,.8])
        clock=integration.clock
        clock.advance()
        missing=synthetic(clock(),int(clock()*1000))
        missing['state'].update(visible=False,tracking_valid=False,invalid_reason='target_missing_or_stale')
        integration.tick(missing)
        commands=adapter.simulated_commands
        controller.tick()
        self.assertEqual(controller.status().state,'SEARCHING')
        self.assertEqual(integration.mode,'follow')
        self.assertEqual(adapter.simulated_commands,commands)
        clock.advance()
        found=synthetic(clock(),int(clock()*1000))
        found['state']['tracking_center']=[.2,.2]
        integration.tick(found)
        controller.tick()
        self.assertEqual(controller.status().state,'TRACKING')
        self.assertEqual(controller.status().mode,'follow')

    def test_camera_staleness_still_returns_to_hold(self):
        adapter,controller,integration=self.run_target([.8,.8])
        clock=integration.clock
        snapshot=synthetic(clock()-1,int(clock()*1000))
        snapshot['state'].update(visible=False,tracking_valid=False,invalid_reason='camera_stale')
        integration.tick(snapshot)
        controller.tick()
        self.assertEqual(integration.mode,'hold')
        self.assertEqual(controller.status().mode,'hold')

