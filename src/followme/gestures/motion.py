"""Beckon/shoo detection from repeated finger curls; no camera or robot imports."""
import math

from followme.gestures.interpreter import GestureZone

CLOSER = 'closer'
FARTHER = 'farther'
WRIST, MIDDLE_KNUCKLE = 0, 9
KNUCKLES = (5, 9, 13, 17)
FINGERTIPS = (8, 12, 16, 20)


def hand_shape(points, aspect=1.):
    """Summarises 21 normalised (x, y) hand landmarks as (openness, fingers).

    openness is the mean fingertip-to-wrist distance in palm lengths: about 2 for a flat hand,
    about 1 for a fist. fingers is 'up' or 'down' when the hand is within 45 degrees of
    vertical, else None. aspect is frame width / height.
    """
    points = [(x * aspect, y) for x, y in points]
    wrist = points[WRIST]
    palm = math.dist(points[MIDDLE_KNUCKLE], wrist)
    if palm <= 0:
        return None, None
    openness = sum(math.dist(points[tip], wrist) for tip in FINGERTIPS) / (len(FINGERTIPS) * palm)
    dx = sum(points[k][0] for k in KNUCKLES) / len(KNUCKLES) - wrist[0]
    dy = sum(points[k][1] for k in KNUCKLES) / len(KNUCKLES) - wrist[1]
    fingers = None if abs(dy) < abs(dx) else 'up' if dy < 0 else 'down'
    return openness, fingers


class BeckonDetector:
    """Emits one step per finger-curl cycle once `cycles_to_start` cycles have been seen.

    Curling the fingers of a raised hand (fingers up) repeatedly means 'closer'; the same
    flicking motion with the hand hanging (fingers down) means 'farther'.
    """

    def __init__(self, open_above=1.6, closed_below=1.3, min_curl_s=.05, max_curl_s=.6,
                 max_gap_s=1.2, cycles_to_start=2, grace_s=.3, zone=None):
        self.open_above = open_above
        self.closed_below = closed_below
        self.min_curl_s = min_curl_s
        self.max_curl_s = max_curl_s
        self.max_gap_s = max_gap_s
        self.cycles_to_start = cycles_to_start
        self.grace_s = grace_s
        self.zone = zone or GestureZone()
        self._last_seen = None
        self._reset()

    def _reset(self):
        self._direction = self._phase = self._closed_at = self._last_cycle = None
        self._count = 0

    def _select(self, hands, face_center):
        valid = [hand for hand in hands if hand.openness is not None and self.zone.contains(hand, face_center)]
        return max(valid, key=lambda hand: hand.score) if valid else None

    def active(self, now):
        """True while a beckon is in progress, so static gestures can be suppressed."""
        return self._count > 0 and now - self._last_cycle <= self.max_gap_s

    def update(self, hands, now, face_center=None):
        """Feed one frame's hands; returns 'closer', 'farther' or None."""
        hand = self._select(hands, face_center)
        if hand is None:
            if self._last_seen is not None and now - self._last_seen > self.grace_s:
                self._reset()
            return None
        self._last_seen = now
        if hand.fingers is None:
            return None
        direction = CLOSER if hand.fingers == 'up' else FARTHER
        if direction != self._direction:
            self._reset()
            self._direction = direction
        phase = 'open' if hand.openness >= self.open_above else 'closed' if hand.openness <= self.closed_below else None
        if phase is None or phase == self._phase:
            return None
        previous, self._phase = self._phase, phase
        if phase == 'closed':
            self._closed_at = now if previous == 'open' else None
            return None
        if self._closed_at is None or not self.min_curl_s <= now - self._closed_at <= self.max_curl_s:
            self._count = 0
            return None
        if self._last_cycle is not None and now - self._last_cycle > self.max_gap_s:
            self._count = 0
        self._count += 1
        self._last_cycle = now
        return direction if self._count >= self.cycles_to_start else None
