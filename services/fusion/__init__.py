"""
Multimodal Data Fusion Module
=============================

Exports the unified multimodal fusion pipeline:
- MultimodalDataFusionEngine
- RoadDefectEvent
- VisualEvidence, EnvironmentalEvidence, SensorEvidence, InfrastructureContext, TOPSISAssessment
- SensorDisagreementAnalyzer, EvidenceSummary, DisagreementCategory
"""

from .schemas import (
    VisualEvidence,
    EnvironmentalEvidence,
    SensorEvidence,
    InfrastructureContext,
    TOPSISAssessment,
    RoadDefectEvent
)
from .fusion_engine import MultimodalDataFusionEngine
from .disagreement_analyzer import (
    SensorDisagreementAnalyzer,
    EvidenceSummary,
    DisagreementCategory
)

__all__ = [
    "MultimodalDataFusionEngine",
    "RoadDefectEvent",
    "VisualEvidence",
    "EnvironmentalEvidence",
    "SensorEvidence",
    "InfrastructureContext",
    "TOPSISAssessment",
    "SensorDisagreementAnalyzer",
    "EvidenceSummary",
    "DisagreementCategory"
]
