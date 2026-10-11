"""Beckon/shoo detection from repeated finger curls, pointing up/down (raise/lower the phone), and the
L hand shape; no camera or robot imports."""
import math

from followme.gestures.interpreter import GestureZone

CLOSER = 'closer'
FARTHER = 'farther'
UP = 'up'
DOWN = 'down'
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


THUMB_BASE, THUMB_TIP, INDEX_KNUCKLE, INDEX_TIP = 2, 4, 5, 8
CURLED_FINGERS = ((12, 9), (16, 13), (20, 17))  # (tip, knuckle) for middle, ring, little finger


def _index_only(points):
    """(index finger straight and the other three curled, thumb out) for scaled landmarks, or None."""
    wrist = points[WRIST]
    palm = math.dist(points[MIDDLE_KNUCKLE], wrist)
    if palm <= 0:
        return None

    def reach(tip, knuckle):
        base = math.dist(points[knuckle], wrist)
        return math.dist(points[tip], wrist) / base if base > 0 else 0.

    if reach(INDEX_TIP, INDEX_KNUCKLE) < 1.5 or any(reach(t, k) > 1.3 for t, k in CURLED_FINGERS):
        return None
    return math.dist(points[THUMB_TIP], points[INDEX_KNUCKLE]) / palm >= .6


def pointing(points, aspect=1.):
    """'up' or 'down' when only the index finger is out (thumb tucked) and it points within 35
    degrees of straight up or down; otherwise None. A thumb sticking out makes it an L instead."""
    points = [(x * aspect, y) for x, y in points]
    if _index_only(points) is not False:  # None: not an index-only hand; True: thumb out (an L)
        return None
    dx = points[INDEX_TIP][0] - points[INDEX_KNUCKLE][0]
    dy = points[INDEX_TIP][1] - points[INDEX_KNUCKLE][1]
    if abs(dx) > abs(dy) * math.tan(math.radians(35)):
        return None
    return 'up' if dy < 0 else 'down'


def l_shape(points, aspect=1.):
    """True when 21 normalised (x, y) hand landmarks make an L.

    The index finger is straight (its tip about twice as far from the wrist as its knuckle), the
    thumb sticks out away from the index knuckle at roughly a right angle to the index finger, and
    the middle, ring and little fingers are curled back towards the palm.
    """
    points = [(x * aspect, y) for x, y in points]
    if not _index_only(points):
        return False
    thumb = (points[THUMB_TIP][0] - points[THUMB_BASE][0], points[THUMB_TIP][1] - points[THUMB_BASE][1])
    index = (points[INDEX_TIP][0] - points[INDEX_KNUCKLE][0], points[INDEX_TIP][1] - points[INDEX_KNUCKLE][1])
    norms = math.hypot(*thumb) * math.hypot(*index)
    if norms <= 0:
        return False
    angle = math.degrees(math.acos(max(-1., min(1., (thumb[0] * index[0] + thumb[1] * index[1]) / norms))))
    return 50. <= angle <= 130.


class PointDetector:
    """Pointing the index finger up raises the phone ('up'), pointing down lowers it ('down').

    The first step comes after `hold_s`, then one every `repeat_s` while the point is held, so the
    phone keeps moving until the hand drops. Gaps shorter than `grace_s` don't stop it.
    """

    def __init__(self, hold_s=.5, repeat_s=.4, grace_s=.2, zone=None):
        self.hold_s, self.repeat_s, self.grace_s = hold_s, repeat_s, grace_s
        self.zone = zone or GestureZone()
        self._reset()

    def _reset(self):
        self._direction = self._since = self._last_seen = self._last_step = None

    def active(self, now):
        """True while a point is being held, so static gestures can be suppressed."""
        return self._last_seen is not None and now - self._last_seen <= self.grace_s

    def update(self, hands, now, face_center=None):
        """Feed one frame's hands; returns 'up', 'down' or None."""
        valid = [hand for hand in hands if hand.pointing in ('up', 'down') and self.zone.contains(hand, face_center)]
        if not valid:
            if self._last_seen is not None and now - self._last_seen > self.grace_s:
                self._reset()
            return None
        direction = UP if max(valid, key=lambda hand: hand.score).pointing == 'up' else DOWN
        if direction != self._direction:
            self._reset()
            self._direction, self._since = direction, now
        self._last_seen = now
        if now - self._since < self.hold_s:
            return None
        if self._last_step is not None and now - self._last_step < self.repeat_s:
            return None
        self._last_step = now
        return direction


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
