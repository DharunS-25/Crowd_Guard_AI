"""Upload-session API contracts (s3-upload.md §4). No S3 key, bucket, or S3 UploadId ever appears here."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .analysis import AnalysisStatus

SHA256_B64 = r"^[A-Za-z0-9+/]{43}=$"


class UploadStrategy(str, Enum):
    SINGLE = "single"
    MULTIPART = "multipart"


class UploadSessionState(str, Enum):
    CREATED = "created"
    UPLOADING = "uploading"
    COMPLETING = "completing"
    UPLOADED = "uploaded"
    ABORTED = "aborted"
    EXPIRED = "expired"
    FAILED = "failed"


class CreateUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    content_type: str = Field(min_length=1, max_length=100)
    authorization_attested: Literal[True]


class CreateUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    upload_id: str
    analysis_id: str
    strategy: UploadStrategy
    part_size: int
    part_count: int
    max_concurrency: int
    checksum_algorithm: Literal["SHA256"] = "SHA256"
    expires_at: datetime


class PartUrlRequestItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    part_number: int = Field(ge=1, le=10_000)
    size_bytes: int = Field(gt=0)
    checksum_sha256: str = Field(pattern=SHA256_B64)


class PartUrlsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parts: List[PartUrlRequestItem] = Field(min_length=1, max_length=50)


class PartUrl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    part_number: int
    url: str
    expires_at: datetime
    required_headers: Dict[str, str] = Field(default_factory=dict)


class PartUrlsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parts: List[PartUrl]


class UploadedPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    part_number: int
    size_bytes: int
    etag: str
    checksum_sha256: Optional[str] = None


class UploadSessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    upload_id: str
    analysis_id: str
    state: UploadSessionState
    strategy: UploadStrategy
    part_size: int
    part_count: int
    expires_at: datetime
    parts: List[UploadedPart] = Field(default_factory=list)


class CompletedPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    part_number: int = Field(ge=1, le=10_000)
    etag: str = Field(min_length=1, max_length=200)
    checksum_sha256: str = Field(pattern=SHA256_B64)


class CompleteUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parts: Optional[List[CompletedPart]] = Field(default=None, max_length=10_000)
    checksum_sha256: Optional[str] = Field(default=None, pattern=SHA256_B64)


class CompleteUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    upload_id: str
    analysis_id: str
    state: UploadSessionState
    analysis_status: AnalysisStatus
