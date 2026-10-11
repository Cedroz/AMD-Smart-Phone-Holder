"""Combines held gestures (freeze/resume) with beckon steps (closer/farther) and the hand twist
(landscape/portrait)."""
from followme.gestures.interpreter import GestureInterpreter
from followme.gestures.motion import BeckonDetector, TwistDetector


class GesturePipeline:
    def __init__(self, interpreter=None, beckon=None, twist=None):
        self.interpreter = interpreter or GestureInterpreter()
        self.beckon = beckon or BeckonDetector(zone=self.interpreter.zone)
        self.twist = twist or TwistDetector(zone=self.interpreter.zone)

    def update(self, hands, now, face_center=None):
        """Feed one frame's hands; returns 'freeze', 'resume', 'closer', 'farther', 'landscape',
        'portrait' or None."""
        step = self.beckon.update(hands, now, face_center)
        turn = self.twist.update(hands, now, face_center)
        # A beckon passes through an open hand, and an edge-on twist may be labelled Open_Palm;
        # do not let either count towards a freeze hold.
        busy = self.beckon.active(now) or self.twist.active(now)
        held = self.interpreter.update([] if busy else hands, now, face_center)
        return step or turn or held
