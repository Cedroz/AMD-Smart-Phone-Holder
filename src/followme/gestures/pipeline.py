"""Combines held gestures (freeze/resume) with beckon steps (closer/farther)."""
from followme.gestures.interpreter import GestureInterpreter
from followme.gestures.motion import BeckonDetector


class GesturePipeline:
    def __init__(self, interpreter=None, beckon=None):
        self.interpreter = interpreter or GestureInterpreter()
        self.beckon = beckon or BeckonDetector(zone=self.interpreter.zone)

    def update(self, hands, now, face_center=None):
        """Feed one frame's hands; returns 'freeze', 'resume', 'closer', 'farther' or None."""
        step = self.beckon.update(hands, now, face_center)
        # A beckon passes through an open hand; do not let that count towards a freeze hold.
        held = self.interpreter.update([] if self.beckon.active(now) else hands, now, face_center)
        return step or held
