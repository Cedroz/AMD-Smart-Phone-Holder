"""Combines held gestures (freeze/resume) with beckon steps (closer/farther)."""
from followme.gestures.interpreter import FREEZE, GestureInterpreter
from followme.gestures.motion import BeckonDetector


class GesturePipeline:
    def __init__(self, interpreter=None, beckon=None):
        self.interpreter = interpreter or GestureInterpreter()
        self.beckon = beckon or BeckonDetector(zone=self.interpreter.zone)

    def update(self, hands, now, face_center=None):
        """Feed one frame's hands; returns 'freeze', 'resume', 'closer', 'farther' or None."""
        step = self.beckon.update(hands, now, face_center)
        # A beckon passes through an open hand; do not let that count towards a freeze hold.
        static_hands = [hand for hand in hands if self.interpreter.bindings.get(hand.label) != FREEZE] if self.beckon.active(now) else hands
        held = self.interpreter.update(static_hands, now, face_center)
        return held or step  # a completed resume must not be hidden by a curl step
