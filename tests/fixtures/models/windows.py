"""Build ZoneWindowFeatures without decoding video (for adapter/policy/overlay unit tests)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from crowdguard_contracts.features import (
    ZONE_WINDOW_FEATURES_V1,
    DataQuality,
    DataQualityState,
    Window,
    ZoneWindowFeatures,
)


def make_windows(zones: Iterable[str] = ("A", "B", "C"), n: int = 4, interval: float = 1.0,
                 counts: Optional[Dict[str, int]] = None, invalid: Iterable[tuple] = (),
                 override: Optional[Dict[tuple, dict]] = None) -> List[ZoneWindowFeatures]:
    counts = counts or {}
    invalid = set(invalid)
    override = override or {}
    out = []
    for k in range(n):
        for z in zones:
            c = counts.get(z, 1)
            values = {
                "people_count_avg": float(c),
                "people_count_end": c,
                "image_density_proxy_per_10000_px": 1.5,
                "mean_speed_px_per_s": 3.0,
                "dominant_direction": "E",
                "virtual_line_crossings_total": 0,
                "flow_crossings_per_s_interval": 0.0,
                "configured_capacity": None,
                "occupancy_ratio": None,
            }
            values.update(override.get((z, k), {}))
            state = DataQualityState.INVALID if (z, k) in invalid else DataQualityState.OK
            out.append(
                ZoneWindowFeatures(
                    camera_id="cam-1",
                    zone_id=z,
                    window=Window(index=k, start_s=k * interval, end_s=(k + 1) * interval),
                    feature_schema=ZONE_WINDOW_FEATURES_V1,
                    values=values,
                    data_quality=DataQuality(state=state, frames_expected=10, frames_decoded=10),
                )
            )
    return out
