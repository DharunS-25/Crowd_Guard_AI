"""Feature windows and data quality."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

FeatureValue = Union[float, int, str, None]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FeatureSchema(_Frozen):
    id: str
    version: str
    columns: List[str] = Field(min_length=1)
    units: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_columns(self) -> "FeatureSchema":
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("feature schema columns must be unique")
        return self

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"


# Columns produced by crowdguard/main.py per zone and interval (minus the window keys).
ZONE_WINDOW_FEATURES_V1 = FeatureSchema(
    id="zone_window_features",
    version="1.0.0",
    columns=[
        "people_count_avg",
        "people_count_end",
        "image_density_proxy_per_10000_px",
        "mean_speed_px_per_s",
        "dominant_direction",
        "virtual_line_crossings_total",
        "flow_crossings_per_s_interval",
        "configured_capacity",
        "occupancy_ratio",
    ],
    units={
        "people_count_avg": "persons",
        "people_count_end": "persons",
        "image_density_proxy_per_10000_px": "persons per 10k image px (proxy, not physical density)",
        "mean_speed_px_per_s": "px/s (proxy, not m/s)",
        "dominant_direction": "8-way image direction",
        "virtual_line_crossings_total": "crossings (cumulative, illustrative line)",
        "flow_crossings_per_s_interval": "crossings/s (cumulative baseline)",
        "configured_capacity": "persons (venue-sourced only)",
        "occupancy_ratio": "ratio (null without sourced capacity)",
    },
)


class Window(_Frozen):
    index: int = Field(ge=0)
    start_s: float = Field(ge=0)
    end_s: float

    @model_validator(mode="after")
    def _ordered(self) -> "Window":
        if not self.start_s < self.end_s:
            raise ValueError("window start_s must be < end_s")
        return self

    def contains(self, t: float) -> bool:
        return self.start_s <= t < self.end_s


class DataQualityState(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


class DataQuality(_Frozen):
    state: DataQualityState
    warnings: List[str] = Field(default_factory=list)
    frames_expected: Optional[int] = Field(default=None, ge=0)
    frames_decoded: Optional[int] = Field(default=None, ge=0)


class ZoneWindowFeatures(_Frozen):
    camera_id: str
    zone_id: str
    window: Window
    feature_schema: FeatureSchema
    values: Dict[str, FeatureValue]
    data_quality: DataQuality

    @model_validator(mode="after")
    def _values_match_schema(self) -> "ZoneWindowFeatures":
        if list(self.values) != self.feature_schema.columns:
            raise ValueError("feature values must follow the schema's column names and order")
        return self
