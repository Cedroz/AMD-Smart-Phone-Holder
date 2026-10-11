"""Continuous pan/tilt goal accumulation using tracker/track.py's image-error rule."""
from .centering import CenteringController
from followme.robotics.contracts import ARM_JOINTS


class XYController(CenteringController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.response_sign = 1  # Same configurable starting sign as track.py.
        self.tilt_sign = 1
        self.axis_velocity = {'shoulder_pan':0.,'wrist_flex':0.}

    def track_step(self,now,measured,center,dt):
        bounds = getattr(self.adapter,'joint_bounds',{'shoulder_pan':self.bounds,'wrist_flex':[-2.,2.]})
        target = dict((j,measured.positions[j]) for j in ARM_JOINTS)
        errors = {'shoulder_pan':(center[0][0]-.5)*2,'wrist_flex':(center[0][1]-.5)*2}
        changed = False
        for joint,sign,fov in [('shoulder_pan',self.response_sign,60.),('wrist_flex',self.tilt_sign,45.)]:
            error=errors[joint]
            # track.py: sign * gain * normalized error * field of view / 2.
            velocity = 0. if abs(error)<=.10 else sign*.15*error*fov/2*10
            velocity=max(-self.limits.velocity,min(self.limits.velocity,velocity))
            previous=self.axis_velocity[joint]
            self.axis_velocity[joint]=previous+max(-self.limits.acceleration*dt,min(self.limits.acceleration*dt,velocity-previous))
            base=self.requested[joint] if self.requested else measured.positions[joint]
            target[joint]=max(bounds[joint][0],min(bounds[joint][1],base+self.axis_velocity[joint]*dt))
            if abs(target[joint]-measured.positions[joint])>2.:
                raise RuntimeError(f'{joint}_feedback_not_following')
            changed=changed or abs(target[joint]-base)>1e-8
        if changed:
            self.adapter.send(target)
            self.requested=target
            self.log('xy_tracking_command',image_error=errors,requested=target,
                     measured=dict(measured.positions),backend=self.adapter.backend)
        self.publish(state='TRACKING',motion_enabled=True,hold_confirmed=not changed)
