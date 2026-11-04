"""Revisit guidance: bearing and distance to a stored plant.

Given a stored identity position and the current phone pose (site-local
metres, quaternion xyzw, camera looks down -Z with +Y up), return the
horizontal distance and the bearing relative to where the phone points:
0 degrees is dead ahead, +90 is to the right, -90 to the left, 180 behind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def rotate_vector(quaternion_xyzw, vector) -> tuple[float, float, float]:
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    vx, vy, vz = (float(v) for v in vector)
    # v' = v + 2 * q_vec x (q_vec x v + w * v)
    cx = y * vz - z * vy + w * vx
    cy = z * vx - x * vz + w * vy
    cz = x * vy - y * vx + w * vz
    return (
        vx + 2.0 * (y * cz - z * cy),
        vy + 2.0 * (z * cx - x * cz),
        vz + 2.0 * (x * cy - y * cx),
    )


def heading_xz(quaternion_xyzw) -> tuple[float, float]:
    """Horizontal unit vector the camera faces (XZ components)."""
    fx, _fy, fz = rotate_vector(quaternion_xyzw, (0.0, 0.0, -1.0))
    length = math.hypot(fx, fz)
    if length < 1e-9:
        return (0.0, -1.0)  # looking straight up/down; fall back to -Z
    return (fx / length, fz / length)


@dataclass(frozen=True)
class Guidance:
    distance_m: float
    bearing_deg: float
    direction: str

    def spoken(self, label: str) -> str:
        return f"{label} is {self.distance_m:.1f} metres {self.direction}."


def _direction_words(bearing_deg: float) -> str:
    a = abs(bearing_deg)
    if a <= 22.5:
        return "ahead of you"
    if a >= 157.5:
        return "behind you"
    side = "right" if bearing_deg > 0 else "left"
    if a <= 67.5:
        return f"ahead and to your {side}"
    if a <= 112.5:
        return f"to your {side}"
    return f"behind and to your {side}"


def guidance(current_position_m, current_quaternion_xyzw, target_position_m) -> Guidance:
    dx = float(target_position_m[0]) - float(current_position_m[0])
    dz = float(target_position_m[2]) - float(current_position_m[2])
    distance = math.hypot(dx, dz)
    if distance < 1e-9:
        return Guidance(distance_m=0.0, bearing_deg=0.0, direction="right here")
    tx, tz = dx / distance, dz / distance
    fx, fz = heading_xz(current_quaternion_xyzw)
    dot = fx * tx + fz * tz
    cross = fx * tz - fz * tx  # positive when the target is to the right
    bearing = math.degrees(math.atan2(cross, dot))
    return Guidance(
        distance_m=round(distance, 3),
        bearing_deg=round(bearing, 1),
        direction=_direction_words(bearing),
    )


def endpoint_from_guidance(current_position_m, current_quaternion_xyzw, guide: Guidance):
    """Where following the guidance lands, in XZ. Used by scripted tests."""
    fx, fz = heading_xz(current_quaternion_xyzw)
    theta = math.radians(guide.bearing_deg)
    # Rotate the heading by the bearing in the XZ plane: facing -Z with a
    # +90 degree bearing must land on +X (to the walker's right).
    rx = fx * math.cos(theta) - fz * math.sin(theta)
    rz = fx * math.sin(theta) + fz * math.cos(theta)
    return (
        float(current_position_m[0]) + rx * guide.distance_m,
        float(current_position_m[2]) + rz * guide.distance_m,
    )
