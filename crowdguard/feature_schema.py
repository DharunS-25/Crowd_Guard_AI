"""Versioned feature schema for per-zone windows and the mapping from the legacy CSV row.

The column order is defined once in `crowdguard_contracts.features.ZONE_WINDOW_FEATURES_V1`. Never reorder or rename
columns here; add a new schema version instead.
"""

from __future__ import annotations

from typing import Dict

from crowdguard_contracts.features import ZONE_WINDOW_FEATURES_V1, FeatureSchema, FeatureValue


def schema() -> FeatureSchema:
    return ZONE_WINDOW_FEATURES_V1


def legacy_row_to_values(row: dict) -> Dict[str, FeatureValue]:
    """Convert one legacy CSV row into schema-ordered values. Blank legacy cells become None (never 0)."""

    values: Dict[str, FeatureValue] = {}
    for column in ZONE_WINDOW_FEATURES_V1.columns:
        value = row[column]
        values[column] = None if value == "" else value
    return values
