"""Analysis job status and zone configuration contracts (system-architecture.md §5, §7.1, §8)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ApiError


class AnalysisStatus(str, Enum):
    AWAITING_UPLOAD = "awaiting_upload"
    VALIDATING = "validating"
    READY = "ready"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_ANALYSIS_STATUSES = frozenset({AnalysisStatus.COMPLETED, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED})


class AnalysisStage(str, Enum):
    DOWNLOAD = "download"
    PROBE = "probe"
    DECODE_DETECT_TRACK = "decode_detect_track"
    FEATURES = "features"
    MODEL_ADAPTERS = "model_adapters"
    FUSION = "fusion"
    RENDER = "render"
    ENCODE = "encode"
    UPLOAD_OUTPUTS = "upload_outputs"
    REGISTER = "register"


class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str
    size_bytes: int = Field(ge=0)
    duration_s: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None
    frame_count: Optional[int] = None


class ZoneConfigRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    version: int
    sha256: str


class AnalysisStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "analysis_status.v1"
    analysis_id: str
    status: AnalysisStatus
    stage: Optional[AnalysisStage] = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime
    source: Optional[SourceMetadata] = None
    zone_config: Optional[ZoneConfigRef] = None
    error: Optional[ApiError] = None


class StartAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    zone_config_id: str = Field(min_length=1, max_length=64)


# ---------------------------------------------------------------- zone configs

Point = Tuple[float, float]


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    d1, d2 = _orientation(q1, q2, p1), _orientation(q1, q2, p2)
    d3, d4 = _orientation(p1, p2, q1), _orientation(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def polygon_area(points: List[Point]) -> float:
    n = len(points)
    return abs(sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1] for i in range(n))) / 2


def is_simple_polygon(points: List[Point]) -> bool:
    n = len(points)
    edges = [(points[i], points[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # adjacent edges share a vertex
            if _segments_cross(*edges[i], *edges[j]):
                return False
    return True


class ZoneDefinition(BaseModel):
    """A zone polygon in normalized image coordinates ([0,1] x [0,1], origin top-left)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    zone_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    points: List[Point] = Field(min_length=3, max_length=64)
    capacity: Optional[float] = Field(default=None, gt=0)
    capacity_source: Optional[str] = Field(default=None, min_length=3, max_length=500)

    @field_validator("points")
    @classmethod
    def _in_range(cls, points: List[Point]) -> List[Point]:
        for x, y in points:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("zone points must be normalized to [0, 1]")
        return points

    @model_validator(mode="after")
    def _geometry_and_capacity(self) -> "ZoneDefinition":
        if polygon_area(self.points) <= 1e-6:
            raise ValueError(f"zone {self.zone_id} has zero area")
        if not is_simple_polygon(self.points):
            raise ValueError(f"zone {self.zone_id} polygon self-intersects")
        if self.capacity is not None and not self.capacity_source:
            raise ValueError("capacity requires capacity_source (venue-approved source, owner, revision)")
        return self


class ZoneConfigCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    camera_id: str = Field(default="cam-1", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    zones: List[ZoneDefinition] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _unique_zone_ids(self) -> "ZoneConfigCreate":
        ids = [z.zone_id for z in self.zones]
        if len(set(ids)) != len(ids):
            raise ValueError("zone_id values must be unique")
        return self


class ZoneConfigResponse(ZoneConfigCreate):
    id: str
    version: int
    sha256: str
    created_at: datetime
