"""Biomechanical Hand Kinematics Engine.

Provides deterministic geometric and anatomical computations for sign language tracking:
1. Bone Length Rigidity Check: Enforces anatomical bone segment ratios and suppresses unphysical stretch.
2. Finger Curl Ratio Calculation: Computes [0.0, 1.0] extension/flexion ratio per finger.
3. Inter-Finger Splay Angles: Computes angular separation (radians) between adjacent proximal phalanges.
4. Pinch Distances: Computes normalized Euclidean distances from Thumb Tip to each finger tip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# Anatomical landmark indices (MediaPipe 21-landmark topology)
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Anatomical finger chains
THUMB_CHAIN = [WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP]
INDEX_CHAIN = [WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP]
MIDDLE_CHAIN = [WRIST, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP]
RING_CHAIN = [WRIST, RING_MCP, RING_PIP, RING_DIP, RING_TIP]
PINKY_CHAIN = [WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP]

FINGER_CHAINS = [THUMB_CHAIN, INDEX_CHAIN, MIDDLE_CHAIN, RING_CHAIN, PINKY_CHAIN]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Safe numerical floor
EPSILON: float = 1e-7


@dataclass(slots=True)
class KinematicFeatures:
    """Computed kinematic features for an individual hand."""

    finger_curls: np.ndarray  # shape (5,), float32 in [0.0, 1.0] [Thumb, Index, Mid, Ring, Pinky]
    finger_splays: np.ndarray  # shape (4,), float32 in radians [T-I, I-M, M-R, R-P]
    pinch_distances: np.ndarray  # shape (4,), float32 normalized distance [T-I, T-M, T-R, T-P]
    binary_mask: int = 0  # 5-bit finger mask (0..31: Thumb=16, Index=8, Mid=4, Ring=2, Pinky=1)
    tilt_angle_deg: float = 0.0  # Palm tilt angle in degrees (-180..180)


class HandKinematicsEngine:
    """Biomechanical analysis and anatomical constraint enforcer for hand skeletons."""

    def __init__(
        self,
        max_bone_stretch: float = 0.25,
        max_bone_compression: float = 0.35,
        projection_blend: float = 0.85,
    ) -> None:
        """Initializes the kinematics engine.

        Args:
            max_bone_stretch: Maximum allowed stretch beyond reference bone length (e.g., 0.25 = +25%).
            max_bone_compression: Maximum allowed compression below reference bone length (e.g., 0.35 = -35%).
            projection_blend: Soft projection blend factor (0.0 = no projection, 1.0 = hard snap).
        """
        self.max_bone_stretch = float(max_bone_stretch)
        self.max_bone_compression = float(max_bone_compression)
        self.projection_blend = float(projection_blend)

        # Stateful reference bone lengths per hand: handedness -> {(parent, child): length}
        self._reference_bones: Dict[str, Dict[Tuple[int, int], float]] = {
            "Left": {},
            "Right": {},
        }

    def reset(self, handedness: Optional[str] = None) -> None:
        """Resets stateful reference bone lengths."""
        if handedness is not None:
            self._reference_bones[handedness] = {}
        else:
            self._reference_bones = {"Left": {}, "Right": {}}

    @staticmethod
    def _compute_bone_length(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p2 - p1))

    def apply_bone_rigidity_constraints(
        self,
        landmarks: np.ndarray,
        handedness: Optional[str] = None,
        custom_ref_lengths: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> np.ndarray:
        """Enforces rigid bone constraints across anatomical finger segments.

        If a landmark deviates by more than the anatomical stretch limit relative to its
        reference segment length, it is softly projected back onto the kinematic bone ray.

        Args:
            landmarks: (21, 3) coordinates.
            handedness: "Left" or "Right" to use stateful smoothed reference lengths.
            custom_ref_lengths: Explicit dictionary of {(parent, child): ref_len} for unit tests.

        Returns:
            Projected coordinates of shape (21, 3), float32.
        """
        coords = landmarks.astype(np.float64, copy=True)
        ref_store = self._reference_bones.get(handedness or "", {}) if handedness else {}

        # Segments to constrain along each finger: (MCP->PIP, PIP->DIP, DIP->TIP)
        # and for Thumb: (CMC->MCP, MCP->IP, IP->TIP)
        segments_to_check: List[Tuple[int, int]] = [
            # Thumb
            (THUMB_CMC, THUMB_MCP),
            (THUMB_MCP, THUMB_IP),
            (THUMB_IP, THUMB_TIP),
            # Index
            (INDEX_MCP, INDEX_PIP),
            (INDEX_PIP, INDEX_DIP),
            (INDEX_DIP, INDEX_TIP),
            # Middle
            (MIDDLE_MCP, MIDDLE_PIP),
            (MIDDLE_PIP, MIDDLE_DIP),
            (MIDDLE_DIP, MIDDLE_TIP),
            # Ring
            (RING_MCP, RING_PIP),
            (RING_PIP, RING_DIP),
            (RING_DIP, RING_TIP),
            # Pinky
            (PINKY_MCP, PINKY_PIP),
            (PINKY_PIP, PINKY_DIP),
            (PINKY_DIP, PINKY_TIP),
        ]

        for parent, child in segments_to_check:
            p_parent = coords[parent]
            p_child = coords[child]
            cur_vec = p_child - p_parent
            cur_len = float(np.linalg.norm(cur_vec))

            if cur_len < EPSILON:
                continue

            unit_vec = cur_vec / cur_len

            # Determine reference length
            ref_len: Optional[float] = None
            if custom_ref_lengths and (parent, child) in custom_ref_lengths:
                ref_len = custom_ref_lengths[(parent, child)]
            elif (parent, child) in ref_store:
                ref_len = ref_store[(parent, child)]
            else:
                # Initialize reference length
                if handedness:
                    ref_store[(parent, child)] = cur_len
                ref_len = cur_len

            if ref_len is None or ref_len < EPSILON:
                continue

            max_allowed = ref_len * (1.0 + self.max_bone_stretch)
            min_allowed = ref_len * (1.0 - self.max_bone_compression)

            if cur_len > max_allowed:
                # Soft projection back onto max allowed length
                clamped_len = (1.0 - self.projection_blend) * cur_len + self.projection_blend * max_allowed
                coords[child] = p_parent + unit_vec * clamped_len
            elif cur_len < min_allowed:
                clamped_len = (1.0 - self.projection_blend) * cur_len + self.projection_blend * min_allowed
                coords[child] = p_parent + unit_vec * clamped_len
            else:
                # Smoothly update reference length when motion is valid
                if handedness and (parent, child) in ref_store:
                    ref_store[(parent, child)] = 0.95 * ref_store[(parent, child)] + 0.05 * cur_len

        return coords.astype(np.float32)

    @staticmethod
    def compute_finger_curls(
        landmarks: np.ndarray,
        world_landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes deterministic [0.0, 1.0] curl ratios for all 5 fingers.

        0.0 = fully curled into palm.
        1.0 = fully extended straight.

        When metric world_landmarks are provided, computes true 3D anatomical extension
        which is 100% immune to 2D perspective foreshortening. Otherwise falls back to
        planar landmark ratios.

        Args:
            landmarks: (21, 3) screen coordinates.
            world_landmarks: Optional (21, 3) metric 3D coordinates in meters.

        Returns:
            np.ndarray of shape (5,), dtype float32: [Thumb, Index, Middle, Ring, Pinky].
        """
        curls = np.zeros(5, dtype=np.float32)

        # Prefer world_landmarks if valid and non-zero
        use_world = world_landmarks is not None and np.any(world_landmarks != 0)
        calc_landmarks = world_landmarks if use_world else landmarks
        wrist = calc_landmarks[WRIST]

        # 1. Four main fingers: Index (1), Middle (2), Ring (3), Pinky (4)
        finger_indices = [
            (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, 1),
            (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, 2),
            (RING_MCP, RING_PIP, RING_DIP, RING_TIP, 3),
            (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, 4),
        ]

        for mcp_idx, pip_idx, dip_idx, tip_idx, out_idx in finger_indices:
            p_mcp = calc_landmarks[mcp_idx]
            p_pip = calc_landmarks[pip_idx]
            p_dip = calc_landmarks[dip_idx]
            p_tip = calc_landmarks[tip_idx]

            d_tip = float(np.linalg.norm(p_tip - wrist))
            d_mcp = float(np.linalg.norm(p_mcp - wrist))

            l_finger = (
                float(np.linalg.norm(p_pip - p_mcp))
                + float(np.linalg.norm(p_dip - p_pip))
                + float(np.linalg.norm(p_tip - p_dip))
            )

            if l_finger < EPSILON:
                curls[out_idx] = 0.0
            else:
                ratio = (d_tip - d_mcp) / l_finger
                curls[out_idx] = float(np.clip(ratio, 0.0, 1.0))

        # 2. Thumb curl calculation
        p_cmc = calc_landmarks[THUMB_CMC]
        p_mcp = calc_landmarks[THUMB_MCP]
        p_ip = calc_landmarks[THUMB_IP]
        p_tip = calc_landmarks[THUMB_TIP]

        d_tip_thumb = float(np.linalg.norm(p_tip - wrist))
        d_mcp_thumb = float(np.linalg.norm(p_mcp - wrist))
        l_thumb = float(np.linalg.norm(p_ip - p_mcp)) + float(np.linalg.norm(p_tip - p_ip))

        if l_thumb < EPSILON:
            curls[0] = 0.0
        else:
            dist_ratio = (d_tip_thumb - d_mcp_thumb) / l_thumb
            v_mcp_ip = p_ip - p_mcp
            v_ip_tip = p_tip - p_ip
            norm1 = float(np.linalg.norm(v_mcp_ip))
            norm2 = float(np.linalg.norm(v_ip_tip))
            if norm1 > EPSILON and norm2 > EPSILON:
                cos_angle = float(np.dot(v_mcp_ip, v_ip_tip) / (norm1 * norm2))
                angle_ratio = (cos_angle + 1.0) * 0.5
            else:
                angle_ratio = 1.0

            thumb_curl = 0.75 * dist_ratio + 0.25 * angle_ratio
            curls[0] = float(np.clip(thumb_curl, 0.0, 1.0))

        return curls

    @staticmethod
    def compute_finger_splays(landmarks: np.ndarray) -> np.ndarray:
        """Computes angular divergence (splay angles in radians) between adjacent proximal finger vectors.

        Returns:
            np.ndarray of shape (4,), dtype float32:
            [Thumb-Index, Index-Middle, Middle-Ring, Ring-Pinky]
        """
        splays = np.zeros(4, dtype=np.float32)

        # Proximal direction vectors
        v_thumb = landmarks[THUMB_IP] - landmarks[THUMB_MCP]
        v_index = landmarks[INDEX_PIP] - landmarks[INDEX_MCP]
        v_middle = landmarks[MIDDLE_PIP] - landmarks[MIDDLE_MCP]
        v_ring = landmarks[RING_PIP] - landmarks[RING_MCP]
        v_pinky = landmarks[PINKY_PIP] - landmarks[PINKY_MCP]

        pairs = [
            (v_thumb, v_index),
            (v_index, v_middle),
            (v_middle, v_ring),
            (v_ring, v_pinky),
        ]

        for i, (v1, v2) in enumerate(pairs):
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 > EPSILON and n2 > EPSILON:
                cos_val = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                splays[i] = float(np.arccos(cos_val))
            else:
                splays[i] = 0.0

        return splays

    @staticmethod
    def compute_pinch_distances(landmarks: np.ndarray) -> np.ndarray:
        """Computes scale-invariant Euclidean distances from Thumb Tip to each of the 4 finger tips.

        Returns:
            np.ndarray of shape (4,), dtype float32:
            [Thumb-Index, Thumb-Middle, Thumb-Ring, Thumb-Pinky]
        """
        pinches = np.zeros(4, dtype=np.float32)
        wrist = landmarks[WRIST]
        thumb_tip = landmarks[THUMB_TIP]

        diffs = landmarks[1:] - wrist
        distances = np.linalg.norm(diffs, axis=1)
        scale = float(np.max(distances)) if len(distances) > 0 else 1.0
        scale = max(scale, EPSILON)

        finger_tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        for i, tip_idx in enumerate(finger_tips):
            tip = landmarks[tip_idx]
            dist = float(np.linalg.norm(tip - thumb_tip))
            pinches[i] = float(dist / scale)

        return pinches

    @staticmethod
    def compute_binary_mask(
        landmarks: np.ndarray,
        finger_curls: np.ndarray,
    ) -> Tuple[int, Dict[str, bool]]:
        """Computes a 5-bit binary bitmask (0..31) indicating extended fingers.

        Bit mapping:
        - Bit 4 (16): Thumb
        - Bit 3 (8):  Index
        - Bit 2 (4):  Middle
        - Bit 1 (2):  Ring
        - Bit 0 (1):  Pinky

        Combines continuous curl ratio (> 0.52) with anatomical bone segment extension.
        """
        wrist = landmarks[WRIST]
        mid_mcp = landmarks[MIDDLE_MCP]
        palm_scale = float(np.linalg.norm(mid_mcp - wrist))
        if palm_scale < 1e-4:
            palm_scale = 0.2

        mask = 0
        states: Dict[str, bool] = {}

        # 1. Thumb (Bit 4 = 16)
        thumb_tip = landmarks[THUMB_TIP]
        thumb_open = finger_curls[0] > 0.52 or (
            float(np.linalg.norm(thumb_tip - mid_mcp)) / palm_scale > 0.68
            and float(np.linalg.norm(thumb_tip - landmarks[INDEX_MCP])) / palm_scale > 0.55
        )
        if thumb_open:
            mask |= 16
        states["Thumb"] = thumb_open

        # 2. Index (Bit 3 = 8)
        index_tip = landmarks[INDEX_TIP]
        d_tip_w = float(np.linalg.norm(index_tip - wrist))
        d_pip_w = float(np.linalg.norm(landmarks[INDEX_PIP] - wrist))
        index_open = (finger_curls[1] > 0.52) or (d_tip_w > d_pip_w and finger_curls[1] > 0.38)
        if index_open:
            mask |= 8
        states["Index"] = index_open

        # 3. Middle (Bit 2 = 4)
        middle_tip = landmarks[MIDDLE_TIP]
        d_mid_w = float(np.linalg.norm(middle_tip - wrist))
        d_mid_pip_w = float(np.linalg.norm(landmarks[MIDDLE_PIP] - wrist))
        middle_open = (finger_curls[2] > 0.52) or (d_mid_w > d_mid_pip_w and finger_curls[2] > 0.38)
        if middle_open:
            mask |= 4
        states["Middle"] = middle_open

        # 4. Ring (Bit 1 = 2)
        ring_tip = landmarks[RING_TIP]
        d_ring_w = float(np.linalg.norm(ring_tip - wrist))
        d_ring_pip_w = float(np.linalg.norm(landmarks[RING_PIP] - wrist))
        ring_open = (finger_curls[3] > 0.52) or (d_ring_w > d_ring_pip_w and finger_curls[3] > 0.38)
        if ring_open:
            mask |= 2
        states["Ring"] = ring_open

        # 5. Pinky (Bit 0 = 1)
        pinky_tip = landmarks[PINKY_TIP]
        d_pinky_w = float(np.linalg.norm(pinky_tip - wrist))
        d_pinky_pip_w = float(np.linalg.norm(landmarks[PINKY_PIP] - wrist))
        pinky_open = (finger_curls[4] > 0.52) or (d_pinky_w > d_pinky_pip_w and finger_curls[4] > 0.38)
        if pinky_open:
            mask |= 1
        states["Pinky"] = pinky_open

        return mask, states

    @staticmethod
    def compute_hand_tilt_angle(landmarks: np.ndarray) -> float:
        """Computes the physical palm tilt angle in degrees (-180° to +180°).

        Orientation:
        - 0° = Upright (pointing straight up)
        - +90° = Pointing Right
        - -90° = Pointing Left
        - ±180° = Pointing Down (inverted)
        """
        wrist = landmarks[WRIST]
        mid_tip_x = float((landmarks[INDEX_TIP, 0] + landmarks[MIDDLE_TIP, 0] + landmarks[RING_TIP, 0]) / 3.0)
        mid_tip_y = float((landmarks[INDEX_TIP, 1] + landmarks[MIDDLE_TIP, 1] + landmarks[RING_TIP, 1]) / 3.0)

        dx = mid_tip_x - float(wrist[0])
        dy = mid_tip_y - float(wrist[1])

        # math.atan2(dx, -dy): In standard camera coordinates (Y down), dy < 0 when pointing up -> 0°
        return float(np.degrees(np.arctan2(dx, -dy)))

    def extract_kinematics(
        self,
        landmarks: np.ndarray,
        handedness: Optional[str] = None,
        enforce_rigidity: bool = False,
        world_landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, KinematicFeatures]:
        """Extracts complete kinematic profile as purely read-only analytical features.

        Guarantees that raw/smoothed screen landmarks are NEVER modified or warped by synthetic
        bone projections, preserving true perspective foreshortening.

        Args:
            landmarks: (21, 3) raw or smoothed screen coordinates.
            handedness: Handedness label ("Left" or "Right").
            enforce_rigidity: Deprecated flag. Always False to protect detector fidelity.
            world_landmarks: Optional (21, 3) metric 3D coordinates in meters.

        Returns:
            Tuple of (unmodified_landmarks (21, 3), KinematicFeatures).
        """
        # Strictly preserve detector coordinates
        clean_landmarks = landmarks.astype(np.float32, copy=True)

        curls = self.compute_finger_curls(clean_landmarks, world_landmarks=world_landmarks)
        splays = self.compute_finger_splays(clean_landmarks)
        pinches = self.compute_pinch_distances(clean_landmarks)
        binary_mask, _ = self.compute_binary_mask(clean_landmarks, curls)
        tilt_angle_deg = self.compute_hand_tilt_angle(clean_landmarks)

        features = KinematicFeatures(
            finger_curls=curls,
            finger_splays=splays,
            pinch_distances=pinches,
            binary_mask=binary_mask,
            tilt_angle_deg=tilt_angle_deg,
        )
        return clean_landmarks, features
