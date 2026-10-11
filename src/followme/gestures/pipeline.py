"""Combines held gestures (freeze/resume), the L sign (landscape/portrait), pointing up/down (raise/lower
the phone), two palms (party) and beckon steps (closer/farther)."""
from followme.gestures.interpreter import FREEZE, GestureInterpreter, LSignDetector, TwoPalmsDetector
from followme.gestures.motion import BeckonDetector, PointDetector


class GesturePipeline:
    def __init__(self, interpreter=None, beckon=None, lsign=None, point=None, two_palms=None):
        self.interpreter = interpreter or GestureInterpreter()
        self.beckon = beckon or BeckonDetector(zone=self.interpreter.zone)
        self.lsign = lsign or LSignDetector(zone=self.interpreter.zone)
        self.point = point or PointDetector(zone=self.interpreter.zone)
        self.two_palms = two_palms or TwoPalmsDetector()

    def update(self, hands, now, face_center=None):
        """Feed one frame's hands; returns 'freeze', 'resume', 'closer', 'farther', 'landscape',
        'portrait' or None."""
        step = self.beckon.update(hands, now, face_center)
        sign = self.lsign.update(hands, now, face_center)
        point = self.point.update(hands, now, face_center)
        party = self.two_palms.update(hands, now, face_center)
        # Motion can pass through Open_Palm, and two raised palms mean party, not freeze; suppress
        # freeze holds while keeping resume available.
        busy = self.beckon.active(now) or self.point.active(now) or self.two_palms.active(now)
        static_hands = [hand for hand in hands if self.interpreter.bindings.get(hand.label) != FREEZE] if busy else hands
        held = self.interpreter.update(static_hands, now, face_center)
        return party or held or step or sign or point
