"""Combines held gestures (freeze/resume) with beckon steps (closer/farther) and the hand twist
(landscape/portrait)."""
from followme.gestures.interpreter import FREEZE, GestureInterpreter
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
        # Motion can pass through Open_Palm; suppress freeze holds while keeping resume available.
        busy = self.beckon.active(now) or self.twist.active(now)
        static_hands = [hand for hand in hands if self.interpreter.bindings.get(hand.label) != FREEZE] if busy else hands
        held = self.interpreter.update(static_hands, now, face_center)
        return held or step or turn
