"""Turns per-frame hand observations into freeze/resume commands; no camera or robot imports."""
from dataclasses import dataclass

FREEZE = 'freeze'
RESUME = 'resume'
DEFAULT_BINDINGS = {'Open_Palm': FREEZE, 'Thumb_Up': RESUME}


@dataclass(frozen=True)
class HandObservation:
    label: str  # MediaPipe gesture category, e.g. 'Open_Palm'
    score: float
    x: float  # palm centre, normalised to the frame (0 left, 1 right)
    y: float  # palm centre, normalised to the frame (0 top, 1 bottom)
    openness: float | None = None  # mean fingertip distance in palm lengths; see motion.hand_shape
    fingers: str | None = None  # 'up', 'down' or None when the hand is sideways


@dataclass(frozen=True)
class GestureZone:
    """Where a hand must be for its gesture to count, in normalised frame units."""
    below_face: float = .25  # how far under the face centre a hand may be
    beside_face: float = .35  # how far left/right of the face centre a hand may be
    max_y_without_face: float = .6  # used when no face position is known

    def contains(self, hand, face_center=None):
        if face_center is None:
            return hand.y <= self.max_y_without_face
        return hand.y <= face_center[1] + self.below_face and abs(hand.x - face_center[0]) <= self.beside_face


class GestureInterpreter:
    """Emits a command once a bound gesture has been held in the zone for `hold_s`.

    After firing, further gestures are ignored for `cooldown_s`, and the gesture that fired
    must be released before it can fire again. Gaps shorter than `grace_s` do not reset a hold.
    """

    def __init__(self, bindings=None, hold_s=.7, cooldown_s=1., grace_s=.2, min_score=.6, zone=None):
        self.bindings = dict(DEFAULT_BINDINGS if bindings is None else bindings)
        self.hold_s = hold_s
        self.cooldown_s = cooldown_s
        self.grace_s = grace_s
        self.min_score = min_score
        self.zone = zone or GestureZone()
        self._candidate = self._since = self._last_seen = None
        self._latched = self._latched_seen = None
        self._cooldown_until = float('-inf')

    def _reset(self):
        self._candidate = self._since = self._last_seen = None

    def _select(self, hands, face_center):
        valid = [hand for hand in hands if hand.label in self.bindings
                 and hand.score >= self.min_score and self.zone.contains(hand, face_center)]
        return max(valid, key=lambda hand: hand.score).label if valid else None

    def update(self, hands, now, face_center=None):
        """Feed one frame's hands; returns 'freeze', 'resume' (or a custom binding) or None."""
        label = self._select(hands, face_center)
        if self._latched is not None:
            if label == self._latched:
                self._latched_seen = now
                self._reset()
                return None
            if now - self._latched_seen > self.grace_s:
                self._latched = None
        if now < self._cooldown_until:
            self._reset()
            return None
        if label is None:
            if self._candidate is not None and now - self._last_seen > self.grace_s:
                self._reset()
            return None
        if label != self._candidate:
            self._candidate, self._since = label, now
        self._last_seen = now
        if now - self._since < self.hold_s:
            return None
        self._latched, self._latched_seen = label, now
        self._cooldown_until = now + self.cooldown_s
        self._reset()
        return self.bindings[label]

    def pending(self, now):
        """(command, fraction of the hold completed) for on-screen feedback, or None."""
        if self._candidate is None:
            return None
        return self.bindings[self._candidate], min(1., (now - self._since) / self.hold_s)
