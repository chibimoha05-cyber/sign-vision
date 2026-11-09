"""Hierarchical Gesture Engine & Temporal State Machine for Sign Language Translation.

Ported and enhanced from Caelestia Shell's industrial-grade gesture architecture:
- FeatureExtraction: Evaluates 5-bit binary finger mask, pinch distances, curl ratios, and spatial tilt.
- Pure Gesture Recognizers: Evaluates gesture candidates with zero hardware side-effects.
- Priority-Based Arbitration (GestureArbiter): Selects single winning candidate based on Priority + Confidence.
- Temporal State Machine (GestureStateMachine):
  Transitions: IDLE -> CANDIDATE -> CONFIRMED -> ACTIVE -> RELEASING -> IDLE.
  Enforces multi-frame hysteresis, prevents state flickering, and allows instant preemption for high-priority gestures.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class GestureName(str, enum.Enum):
    """Canonical Sign Language & Control Hand Gestures."""

    NONE = "none"
    FIST = "fist"  # Binary mask 0 (all curled)
    OPEN_PALM = "open_palm"  # Binary mask 31 (all open)
    POINT = "point"  # Index extended, others curled (mask 8 or 24)
    PEACE_V = "peace_v"  # Index + Middle extended (mask 12 or 28)
    THREE_FINGERS = "three_fingers"  # Index + Middle + Ring extended (mask 14)
    FOUR_FINGERS = "four_fingers"  # Index + Middle + Ring + Pinky (mask 15)
    ROCK_HORNS = "rock_horns"  # Index + Pinky extended (mask 9 or 25)
    SHAKA = "shaka"  # Thumb + Pinky extended (mask 17)
    OK_PINCH = "ok_pinch"  # Thumb touching Index tip
    PINCH_MID = "pinch_mid"  # Thumb touching Middle tip
    THUMB_UP = "thumb_up"  # Thumb extended upward, other fingers curled
    THUMB_DOWN = "thumb_down"  # Thumb extended downward, other fingers curled


# Priority Table: Higher priority gestures preempt lower ones
GESTURE_PRIORITY: Dict[GestureName, int] = {
    GestureName.NONE: 0,
    GestureName.OPEN_PALM: 50,
    GestureName.FOUR_FINGERS: 55,
    GestureName.THREE_FINGERS: 60,
    GestureName.PEACE_V: 65,
    GestureName.POINT: 70,
    GestureName.ROCK_HORNS: 75,
    GestureName.SHAKA: 75,
    GestureName.THUMB_DOWN: 80,
    GestureName.THUMB_UP: 80,
    GestureName.FIST: 85,
    GestureName.PINCH_MID: 90,
    GestureName.OK_PINCH: 95,
}


@dataclasses.dataclass(slots=True)
class GestureCandidate:
    """A recognized gesture proposal with confidence and priority."""

    gesture: GestureName
    hand: str  # "Left" | "Right" | "Any"
    confidence: float  # 0.0 - 1.0
    valid: bool
    data: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def priority(self) -> int:
        return GESTURE_PRIORITY.get(self.gesture, 0)

    @property
    def score(self) -> float:
        """Score metric: priority scaled to dominate confidence."""
        return self.priority * 10.0 + self.confidence


class BaseGestureRecognizer:
    """Base class for decoupled gesture recognition."""

    def recognise(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        raise NotImplementedError


class FistRecognizer(BaseGestureRecognizer):
    """All 5 fingers curled into the palm."""

    def recognise(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        # Mask 0 = all 5 fingers curled (or 16 if thumb slightly out, but curls < 0.25)
        is_fist = (binary_mask == 0) or (binary_mask == 16 and finger_curls[0] < 0.35 and np.all(finger_curls[1:] < 0.25))
        if not is_fist and np.all(finger_curls < 0.28):
            is_fist = True

        mean_curl = float(np.mean(finger_curls))
        confidence = float(np.clip(1.0 - mean_curl * 2.5, 0.0, 1.0)) if is_fist else 0.0
        return GestureCandidate(
            gesture=GestureName.FIST,
            hand=hand_label,
            confidence=confidence,
            valid=is_fist and confidence > 0.5,
        )


class OpenPalmRecognizer(BaseGestureRecognizer):
    """All 5 fingers extended straight."""

    def recognise(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        is_open = (binary_mask == 31) or (np.all(finger_curls > 0.70))
        mean_curl = float(np.mean(finger_curls))
        confidence = float(np.clip((mean_curl - 0.50) * 2.0, 0.0, 1.0)) if is_open else 0.0
        return GestureCandidate(
            gesture=GestureName.OPEN_PALM,
            hand=hand_label,
            confidence=confidence,
            valid=is_open and confidence > 0.5,
        )


class PointRecognizer(BaseGestureRecognizer):
    """Index finger extended, other fingers curled."""

    def recognise(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        # Binary mask: Index open (bit 3 = 8), Middle/Ring/Pinky closed. Thumb may be relaxed (mask 8 or 24).
        is_point = (binary_mask in (8, 24)) or (
            finger_curls[1] > 0.70 and finger_curls[2] < 0.35 and finger_curls[3] < 0.35 and finger_curls[4] < 0.35
        )
        confidence = float(finger_curls[1]) if is_point else 0.0
        return GestureCandidate(
            gesture=GestureName.POINT,
            hand=hand_label,
            confidence=confidence,
            valid=is_point and confidence > 0.6,
        )


class PeaceVRecognizer(BaseGestureRecognizer):
    """Index + Middle fingers extended in V shape."""

    def recognise(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        # Binary mask: Index (8) + Middle (4) = 12 (or 28 with thumb)
        is_peace = (binary_mask in (12, 28)) or (
            finger_curls[1] > 0.65
            and finger_curls[2] > 0.65
            and finger_curls[3] < 0.35
            and finger_curls[4] < 0.35
        )
        confidence = float((finger_curls[1] + finger_curls[2]) / 2.0) if is_peace else 0.0
        return GestureCandidate(
            gesture=GestureName.PEACE_V,
            hand=hand_label,
            confidence=confidence,
            valid=is_peace and confidence > 0.6,
        )


class ThreeFingersRecognizer(BaseGestureRecognizer):
    """Index + Middle + Ring fingers extended."""

    def recognise(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        # Binary mask: 8 + 4 + 2 = 14 (or 30 with thumb)
        is_three = (binary_mask in (14, 30)) or (
            finger_curls[1] > 0.65
            and finger_curls[2] > 0.65
            and finger_curls[3] > 0.65
            and finger_curls[4] < 0.35
        )
        confidence = float((finger_curls[1] + finger_curls[2] + finger_curls[3]) / 3.0) if is_three else 0.0
        return GestureCandidate(
            gesture=GestureName.THREE_FINGERS,
            hand=hand_label,
            confidence=confidence,
            valid=is_three and confidence > 0.6,
        )


class FourFingersRecognizer(BaseGestureRecognizer):
    """Index + Middle + Ring + Pinky extended, thumb curled."""

    def recognise(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        # Mask: 8 + 4 + 2 + 1 = 15
        is_four = (binary_mask == 15) or (
            finger_curls[0] < 0.40
            and finger_curls[1] > 0.65
            and finger_curls[2] > 0.65
            and finger_curls[3] > 0.65
            and finger_curls[4] > 0.65
        )
        confidence = float(np.mean(finger_curls[1:])) if is_four else 0.0
        return GestureCandidate(
            gesture=GestureName.FOUR_FINGERS,
            hand=hand_label,
            confidence=confidence,
            valid=is_four and confidence > 0.6,
        )


class RockHornsRecognizer(BaseGestureRecognizer):
    """Index + Pinky extended (Rock 🤘 sign), Middle + Ring curled."""

    def recognise(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        # Mask: 8 + 1 = 9 (or 25 with thumb)
        is_rock = (binary_mask in (9, 25)) or (
            finger_curls[1] > 0.65
            and finger_curls[4] > 0.65
            and finger_curls[2] < 0.35
            and finger_curls[3] < 0.35
        )
        confidence = float((finger_curls[1] + finger_curls[4]) / 2.0) if is_rock else 0.0
        return GestureCandidate(
            gesture=GestureName.ROCK_HORNS,
            hand=hand_label,
            confidence=confidence,
            valid=is_rock and confidence > 0.6,
        )


class ShakaRecognizer(BaseGestureRecognizer):
    """Thumb + Pinky extended (🤙 Shaka / Phone pose)."""

    def recognise(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        # Mask: 16 + 1 = 17
        is_shaka = (binary_mask == 17) or (
            finger_curls[0] > 0.65
            and finger_curls[4] > 0.65
            and finger_curls[1] < 0.35
            and finger_curls[2] < 0.35
            and finger_curls[3] < 0.35
        )
        confidence = float((finger_curls[0] + finger_curls[4]) / 2.0) if is_shaka else 0.0
        return GestureCandidate(
            gesture=GestureName.SHAKA,
            hand=hand_label,
            confidence=confidence,
            valid=is_shaka and confidence > 0.6,
        )


class OkPinchRecognizer(BaseGestureRecognizer):
    """Thumb tip touching Index tip (OK pinch gesture 👌)."""

    def recognise(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        # Pinch distance between Thumb and Index is pinch_distances[0]
        # In scale-invariant normalized units, pinch < 0.22 indicates contact
        d_ti = float(pinch_distances[0]) if len(pinch_distances) > 0 else 1.0
        is_ok = d_ti < 0.22 and (finger_curls[2] > 0.40 or finger_curls[3] > 0.40 or finger_curls[4] > 0.40)
        confidence = float(np.clip(1.0 - (d_ti / 0.22), 0.0, 1.0)) if is_ok else 0.0
        return GestureCandidate(
            gesture=GestureName.OK_PINCH,
            hand=hand_label,
            confidence=confidence,
            valid=is_ok and confidence > 0.5,
            data={"d_ti": d_ti},
        )


class ThumbUpRecognizer(BaseGestureRecognizer):
    """Thumb extended straight up (+Y orientation in tilt space), fingers curled."""

    def recognise(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        is_thumb_only = (binary_mask == 16) or (
            finger_curls[0] > 0.65 and np.all(finger_curls[1:] < 0.35)
        )
        # Upright tilt: |tilt_angle_deg| <= 55°
        is_up = is_thumb_only and abs(tilt_angle_deg) <= 55.0
        confidence = float(finger_curls[0]) if is_up else 0.0
        return GestureCandidate(
            gesture=GestureName.THUMB_UP,
            hand=hand_label,
            confidence=confidence,
            valid=is_up and confidence > 0.6,
        )


class ThumbDownRecognizer(BaseGestureRecognizer):
    """Thumb extended straight down, fingers curled."""

    def recognise(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        is_thumb_only = (binary_mask == 16) or (
            finger_curls[0] > 0.65 and np.all(finger_curls[1:] < 0.35)
        )
        # Downward tilt: |tilt_angle_deg| >= 125°
        is_down = is_thumb_only and abs(tilt_angle_deg) >= 125.0
        confidence = float(finger_curls[0]) if is_down else 0.0
        return GestureCandidate(
            gesture=GestureName.THUMB_DOWN,
            hand=hand_label,
            confidence=confidence,
            valid=is_down and confidence > 0.6,
        )


class GestureArbiter:
    """Arbitrates among competing gesture recognizers based on Priority and Confidence."""

    def __init__(self, recognizers: Optional[List[BaseGestureRecognizer]] = None) -> None:
        if recognizers is None:
            self.recognizers: List[BaseGestureRecognizer] = [
                FistRecognizer(),
                OkPinchRecognizer(),
                OpenPalmRecognizer(),
                PointRecognizer(),
                PeaceVRecognizer(),
                ThreeFingersRecognizer(),
                FourFingersRecognizer(),
                RockHornsRecognizer(),
                ShakaRecognizer(),
                ThumbUpRecognizer(),
                ThumbDownRecognizer(),
            ]
        else:
            self.recognizers = recognizers

    def arbitrate(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
        hand_label: str = "Right",
    ) -> GestureCandidate:
        candidates: List[GestureCandidate] = []
        for rec in self.recognizers:
            cand = rec.recognise(
                binary_mask=binary_mask,
                finger_curls=finger_curls,
                pinch_distances=pinch_distances,
                tilt_angle_deg=tilt_angle_deg,
                hand_label=hand_label,
            )
            if cand.valid:
                candidates.append(cand)

        if not candidates:
            return GestureCandidate(GestureName.NONE, hand_label, 0.0, False)

        return max(candidates, key=lambda c: c.score)


class GestureState(str, enum.Enum):
    """States of the temporal gesture state machine."""

    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    ACTIVE = "ACTIVE"
    RELEASING = "RELEASING"


class HandGestureStateMachine:
    """State Machine enforcing multi-frame confirmation and debounced release per hand."""

    def __init__(
        self,
        hand_label: str = "Right",
        confirm_frames: int = 2,
        release_frames: int = 2,
    ) -> None:
        self.hand_label = hand_label
        self.confirm_frames = max(1, confirm_frames)
        self.release_frames = max(1, release_frames)

        self.state: GestureState = GestureState.IDLE
        self.active_gesture: GestureName = GestureName.NONE
        self.candidate: Optional[GestureCandidate] = None
        self.candidate_frames: int = 0
        self.release_counter: int = 0
        self.just_activated: bool = False
        self.arbiter = GestureArbiter()

    def reset(self) -> None:
        """Resets the state machine to IDLE."""
        self.state = GestureState.IDLE
        self.active_gesture = GestureName.NONE
        self.candidate = None
        self.candidate_frames = 0
        self.release_counter = 0
        self.just_activated = False

    def process(
        self,
        binary_mask: int,
        finger_curls: np.ndarray,
        pinch_distances: np.ndarray,
        tilt_angle_deg: float,
    ) -> Tuple[GestureName, str]:
        """Processes single-frame kinematic features and returns (confirmed_gesture, state_name)."""
        winner = self.arbiter.arbitrate(
            binary_mask=binary_mask,
            finger_curls=finger_curls,
            pinch_distances=pinch_distances,
            tilt_angle_deg=tilt_angle_deg,
            hand_label=self.hand_label,
        )

        self.just_activated = False

        if self.state == GestureState.IDLE:
            if winner.valid and winner.gesture != GestureName.NONE:
                self.candidate = winner
                self.candidate_frames = 1
                if self.confirm_frames <= 1:
                    self.active_gesture = winner.gesture
                    self.state = GestureState.ACTIVE
                    self.just_activated = True
                else:
                    self.state = GestureState.CANDIDATE

        elif self.state == GestureState.CANDIDATE:
            if winner.valid and self.candidate and winner.gesture == self.candidate.gesture:
                self.candidate_frames += 1
                if self.candidate_frames >= self.confirm_frames:
                    self.active_gesture = winner.gesture
                    self.state = GestureState.ACTIVE
                    self.just_activated = True
                    self.release_counter = 0
            elif winner.valid and winner.gesture != GestureName.NONE:
                # Switch candidate
                self.candidate = winner
                self.candidate_frames = 1
            else:
                self.state = GestureState.IDLE
                self.candidate = None
                self.candidate_frames = 0

        elif self.state == GestureState.ACTIVE:
            # Check for instant preemption: higher priority gesture immediately interrupts active gesture
            if winner.valid and winner.gesture != self.active_gesture:
                winner_prio = winner.priority
                current_prio = GESTURE_PRIORITY.get(self.active_gesture, 0)
                if winner_prio > current_prio:
                    # Instant preemption
                    self.active_gesture = winner.gesture
                    self.state = GestureState.ACTIVE
                    self.just_activated = True
                    self.release_counter = 0
                    self.candidate = winner
                    self.candidate_frames = self.confirm_frames
                else:
                    self.state = GestureState.RELEASING
                    self.release_counter = 1
            elif not winner.valid or winner.gesture == GestureName.NONE:
                self.state = GestureState.RELEASING
                self.release_counter = 1
            else:
                # Still holding active gesture
                self.release_counter = 0

        elif self.state == GestureState.RELEASING:
            if winner.valid and winner.gesture == self.active_gesture:
                # Re-engaged
                self.state = GestureState.ACTIVE
                self.release_counter = 0
            elif winner.valid and winner.gesture != GestureName.NONE:
                self.candidate = winner
                self.candidate_frames = 1
                self.active_gesture = GestureName.NONE
                self.state = GestureState.CANDIDATE
            else:
                self.release_counter += 1
                if self.release_counter >= self.release_frames:
                    self.active_gesture = GestureName.NONE
                    self.state = GestureState.IDLE
                    self.candidate = None
                    self.candidate_frames = 0

        return self.active_gesture, self.state.value
