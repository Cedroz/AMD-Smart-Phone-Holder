"""Deterministic observation replay and independently annotated trial evaluation."""
import json
from pathlib import Path

from followme.phase2.tracking import Tracker, Zones, validate_settings


def replay(path, annotations=None):
    events=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    session=next(event for event in events if event['type']=='session')
    tracker=Tracker(validate_settings(session['config']),Zones(session['zones']))
    transitions=[]
    mismatches=[]
    last_zone=None
    for event in events:
        kind=event['type']
        now=event.get('processed_monotonic_ns',event['logged_monotonic_ns'])
        if kind=='observation': tracker.update(event['reply'],now)
        elif kind=='reacquire': tracker.reacquire()
        elif kind=='zones':
            tracker.zones=Zones(event['config']);tracker.reacquire()
        elif kind=='worker_failure': tracker.invalidate(now,'worker_failure',clear_selection=True)
        elif kind in ('state','state_transition'):
            recorded=event['state']
            if recorded['invalid_reason']=='external_camera_failure':
                tracker.invalidate(now,'external_camera_failure',clear_selection=True)
            tracker.tick(now)
            actual=tracker.state
            keys=('stable_zone','candidate_zone','tracking_valid','selection_required','track_generation')
            if any(getattr(actual,key)!=recorded[key] for key in keys):
                mismatches.append({'timestamp_ns':now,'expected':{key:recorded[key] for key in keys},
                                   'actual':{key:getattr(actual,key) for key in keys}})
        else: continue
        zone=tracker.state.stable_zone
        if zone!=last_zone:
            if zone: transitions.append({'timestamp_ns':now,'zone':zone})
            last_zone=zone
    report={'replay_matches':not mismatches,'mismatches':mismatches,'accepted_zones':transitions,
            'live_acceptance':'pending independent participant trial annotations','motion_commands':0}
    if annotations:
        data=json.loads(Path(annotations).read_text())
        trials=[]
        matched=set()
        for trial in data['transitions']:
            start,end=trial['entry_monotonic_ns'],trial['exit_monotonic_ns']
            matches=[(i,t) for i,t in enumerate(transitions) if start<=t['timestamp_ns']<=end and t['zone']==trial['zone']]
            if matches:
                i,t=matches[0];matched.add(i)
                latency=(t['timestamp_ns']-start)/1e9
            else: latency=None
            trials.append({'zone':trial['zone'],'latency_s':latency,'passed':latency is not None and latency<=2})
        stationary=[]
        for trial in data.get('stationary',[]):
            wrong=[t for t in transitions if trial['start_monotonic_ns']<=t['timestamp_ns']<=trial['end_monotonic_ns'] and t['zone']!=trial['zone']]
            duration=(trial['end_monotonic_ns']-trial['start_monotonic_ns'])/1e9
            stationary.append({'zone':trial['zone'],'duration_s':duration,'false_transitions':len(wrong),'passed':not wrong and duration>=120})
        extra=[t for i,t in enumerate(transitions) if i not in matched and
               any(a['entry_monotonic_ns']<=t['timestamp_ns']<=a['exit_monotonic_ns'] for a in data['transitions'])]
        alternating=all(a['zone']!=b['zone'] and a['entry_monotonic_ns']<b['entry_monotonic_ns'] for a,b in zip(data['transitions'],data['transitions'][1:]))
        passed=(len(trials)>=10 and alternating and all(t['passed'] for t in trials) and not extra
                and {t['zone'] for t in stationary}=={'station_a','station_b'} and all(t['passed'] for t in stationary)
                and not mismatches)
        report.update(trials=trials,stationary=stationary,extra_transitions=extra,
                      live_acceptance='blocked — proxy source does not verify native freshness' if session.get('preview_sources') else ('passed' if passed else 'failed'))
    return report
