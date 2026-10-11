import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from followme.phase4.library import blank, capture, record, check, content_digest, approve, read_samples
from followme.phase4.replay import replay
from followme.phase4.cli import main
from followme.phase4.library import save
from followme.robotics.contracts import ARM_JOINTS, JOINTS


def samples(pan, duration=2.1):
    return [{'timestamp_s':i*.1,'positions':{**dict.fromkeys(JOINTS,0.),'shoulder_pan':pan,'gripper':25.}}
            for i in range(round(duration*10)+1)]


def candidate():
    data = blank({'hardware_sha256':'offline'},'a'*64)
    data['joint_limits'] = {j:[-30.,30.] for j in ARM_JOINTS}
    capture(data,'station_a',samples(0),'a.jsonl')
    capture(data,'station_b',samples(10),'b.jsonl')
    for source,destination in [('station_a','station_b'),('station_b','station_a')]:
        start=data['poses'][source]['joints']['shoulder_pan']
        end=data['poses'][destination]['joints']['shoulder_pan']
        route=samples(start)
        for i,s in enumerate(route):
            s['positions']['shoulder_pan']=start+(end-start)*i/(len(route)-1)
        record(data,source,destination,route,[0,10,len(route)-1],'route.jsonl')
    return data


class Phase4Tests(unittest.TestCase):
    def test_draft_has_both_valid_paths_and_no_approval(self):
        data=candidate()
        check(data)
        self.assertEqual(data['validation']['status'],'draft')
        self.assertFalse(data['transitions']['station_a->station_b']['approved'])

    def test_stable_capture_required(self):
        data=blank({},'a'*64)
        unstable=samples(0)
        unstable[-1]['positions']['wrist_roll']=2
        with self.assertRaisesRegex(ValueError,'unstable'):
            capture(data,'station_a',unstable,'x')
        with self.assertRaisesRegex(ValueError,'two_seconds'):
            capture(data,'station_a',samples(0,.5),'x')

    def test_gripper_fixed(self):
        data=candidate()
        s=samples(0)
        for value in s:
            value['positions']['gripper']=30
        with self.assertRaisesRegex(ValueError,'gripper_changed'):
            capture(data,'station_a',s,'x')

    def test_missing_direction_units_bounds_and_dynamics(self):
        for mutation in [lambda d:d['transitions'].pop('station_b->station_a'),
                         lambda d:d.update(units='percent'),
                         lambda d:d['joint_limits'].update(shoulder_pan=[0,0]),
                         lambda d:d['transitions']['station_a->station_b'].update(durations_s=[.01,.01])]:
            data=candidate();mutation(data)
            with self.assertRaises(ValueError): check(data)

    def test_approval_invalidated_by_edit(self):
        data=candidate()
        data['validation']={'status':'approved','content_sha256':content_digest(data)}
        data['setup']['new_payload']=True
        with self.assertRaisesRegex(ValueError,'invalidated'): check(data)

    def test_replay_is_serial_free(self):
        with tempfile.TemporaryDirectory() as root, patch('followme.robotics.adapters.LeRobotAdapter.connect',side_effect=AssertionError('serial')):
            result=replay(candidate(),'station_a','station_b',root)
        self.assertTrue(result['passed'])
        self.assertEqual(result['physical_commands'],0)

    def test_priority_interruptions_do_not_resume(self):
        for kind in ['hold','stop']:
            for at in [.1,.5,.8]:
                with tempfile.TemporaryDirectory() as root:
                    result=replay(candidate(),'station_a','station_b',root,interrupt=kind,at=at)
                self.assertTrue(result['passed'],result)
                self.assertLessEqual(result['latency_s'],.25)
                self.assertEqual(result['state'],'HOLDING' if kind=='hold' else 'STOPPED')

    def test_simulation_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as root:
            # Physical gate must fail before accepting any supplied attestation.
            p=Path(root)/'artifacts/phase-0';p.mkdir(parents=True)
            import yaml,hashlib
            hardware={'assessment':{'overall_outcome':'blocked'}}
            raw=yaml.safe_dump(hardware).encode();(p/'hardware.yaml').write_bytes(raw)
            data=candidate();data['setup']['hardware_sha256']=hashlib.sha256(raw).hexdigest()
            with self.assertRaisesRegex(ValueError,'physical_gates_pending'):
                approve(data,{'backend':'fake'},root)

    def test_sample_timestamps_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'x.jsonl'
            data=samples(0);data[2]['timestamp_s']=data[1]['timestamp_s']
            p.write_text(''.join(json.dumps(s)+'\n' for s in data))
            with self.assertRaisesRegex(ValueError,'unordered'):read_samples(p)

    def test_boolean_bounds_and_ambiguous_anchors_rejected(self):
        data=candidate();data['joint_limits']['wrist_roll']=[False,30]
        with self.assertRaisesRegex(ValueError,'invalid_bounds'):check(data)
        data=candidate()
        data['poses']['station_b']['joints']=dict(data['poses']['station_a']['joints'])
        with self.assertRaisesRegex(ValueError,'ambiguous_pose'):check(data)

    def test_cycles_cli_writes_ten_simulated_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);session=root/'session';session.mkdir()
            save(session/'safe_poses.yaml',candidate())
            with contextlib.redirect_stdout(io.StringIO()):
                code=main(['--root',str(root),'--session',str(session),'cycles'])
            self.assertEqual(code,0)
            logs=list(session.glob('cycle-*.json'))
            self.assertEqual(len(logs),10)
            for path in logs:
                trial=json.loads(path.read_text())
                self.assertTrue(trial['passed'])
                self.assertEqual(trial['backend'],'fake')
                self.assertTrue(all(leg['physical_commands']==0 for leg in trial['legs']))


if __name__ == '__main__':
    unittest.main()
