"""
Sensor Disagreement Analysis & Detection Confidence Fusion
==========================================================

Implements formal multi-modal evidence fusion and disagreement classification.
Addresses research scenarios where individual sensors offer conflicting observations:
- Visual detection with weak/zero physical depth (superficial stain / shadow)
- Visual detection with verified cavity but no dynamic wheel engagement (straddled pass)
- Incontrovertible multimodal concordance (vision + deep cavity + acute shock)
- Submerged / rain conditions where optical sensors fail but acoustic sonar salvages

Strict Rule:
Does not arbitrarily declare one sensor as universally 'true'. Computes an
auditable multi-sensor evidence summary with explicit fusion rules.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field


class DisagreementCategory(str, Enum):
    """Formal taxonomy of multi-sensor agreement and divergence states."""
    CONCORDANT_CONFIRMED = "concordant_confirmed"
    VISUALLY_DETECTED_PHYSICALLY_WEAK = "visually_detected_physically_weak"
    SUPERFICIAL_VISUAL_ONLY = "superficial_visual_only"
    ACOUSTIC_SALVAGE_IN_RAIN = "acoustic_salvage_in_rain"
    DYNAMIC_IMPACT_UNCONFIRMED_VISUALLY = "dynamic_impact_unconfirmed_visually"
    CONCORDANT_NOMINAL = "concordant_nominal"


class EvidenceSummary(BaseModel):
    """
    Structured Multi-Sensor Evidence Breakdown
    ------------------------------------------
    Exposes quantitative confidence for each sensing dimension, the fused
    multimodal score, and the categorical disagreement diagnosis.
    """
    visual_confidence: float = Field(..., ge=0.0, le=1.0, description="Visual detection confidence score")
    depth_evidence: Dict[str, Any] = Field(..., description="Depth void confirmation metrics")
    impact_evidence: Dict[str, Any] = Field(..., description="Vertical acceleration dynamic shock metrics")
    environment_evidence: Dict[str, Any] = Field(..., description="Environmental validity factors")
    multimodal_confidence_score: float = Field(..., ge=0.0, le=1.0, description="Weighted composite evidence score")
    disagreement_category: DisagreementCategory = Field(..., description="Categorical diagnosis of sensor concordance/divergence")
    fusion_rationale: str = Field(..., description="Human-interpretable scientific rationale for the fusion decision")


class SensorDisagreementAnalyzer:
    """
    Analyzes multi-sensor telemetry for concordance or disagreement,
    calculating calibrated confidence weights and diagnostic classifications.
    """

    def __init__(self, impact_threshold_g: float = 1.50):
        self.impact_threshold_g = impact_threshold_g

    def evaluate_visual_evidence(
        self,
        damage_class: str,
        confidence: Optional[float],
        surface_area_sqm: float
    ) -> Tuple[float, str]:
        """Calculates visual evidence score ∈ [0.0, 1.0]."""
        base_conf = float(confidence) if confidence is not None else 0.90
        # Larger area increases visual saliency
        area_bonus = min(0.08, max(0.0, (surface_area_sqm - 0.5) * 0.04))
        score = min(1.0, max(0.05, round(base_conf + area_bonus, 3)))
        assessment = f"Visual confidence: {score:.2f} ({damage_class}, area: {surface_area_sqm:.2f} m²)"
        return score, assessment

    def evaluate_depth_evidence(
        self,
        damage_class: str,
        effective_depth_cm: float,
        active_modality: str
    ) -> Tuple[float, bool, str]:
        """
        Evaluates whether measured depth represents an acute cavity.
        Returns: (depth_score ∈ [0, 1], void_confirmed: bool, assessment)
        """
        d = max(0.0, float(effective_depth_cm))

        # Benchmark thresholds by damage class
        if damage_class == "D40":
            # Potholes: >= 4.0 cm confirmed void; >= 10.0 cm severe void
            void_confirmed = bool(d >= 4.0)
            score = min(1.0, max(0.0, d / 14.0))
        elif damage_class == "D20":
            # Alligator fatigue: >= 1.5 cm void
            void_confirmed = bool(d >= 1.5)
            score = min(1.0, max(0.0, d / 5.0))
        else:
            # Linear cracks (D00, D10): >= 0.5 cm void
            void_confirmed = bool(d >= 0.5)
            score = min(1.0, max(0.0, d / 2.5))

        score = round(score, 3)
        mod_label = "Acoustic Sonar" if "sonar" in active_modality.lower() else "Optical LiDAR"
        assessment = f"{mod_label} measured {d:.1f} cm (void_confirmed={void_confirmed}, score={score:.2f})"
        return score, void_confirmed, assessment

    def evaluate_impact_evidence(
        self,
        z_accel_g: float
    ) -> Tuple[float, bool, str, str]:
        """
        Evaluates dynamic wheel strike confirmation from vertical acceleration.
        Returns: (impact_score ∈ [0, 1], impact_confirmed: bool, regime, assessment)
        """
        z = max(0.0, float(z_accel_g))
        impact_confirmed = bool(z > self.impact_threshold_g)

        if z > self.impact_threshold_g:
            regime = "confirmed_impact"
            # Scale from 0.80 to 1.00 for shocks between 1.50g and 2.50g
            score = min(1.0, round(0.80 + 0.20 * min(1.0, (z - 1.50) / 1.0), 3))
            assessment = f"Acute dynamic shock: {z:.2f}g > {self.impact_threshold_g}g threshold"
        elif z > 1.35:
            regime = "borderline"
            score = round(0.40 + 0.35 * ((z - 1.35) / 0.15), 3)
            assessment = f"Borderline dynamic disturbance: {z:.2f}g (1.35g - 1.50g)"
        else:
            regime = "no_impact"
            score = max(0.05, round(0.10 * (z / 1.35), 3))
            assessment = f"Nominal vertical chassis response: {z:.2f}g (no significant impact)"

        return score, impact_confirmed, regime, assessment

    def evaluate_environment_evidence(
        self,
        rain_detected: bool,
        mean_luminance: float,
        active_modality: str
    ) -> Tuple[float, str]:
        """Evaluates environmental observation reliability."""
        is_wet = bool(rain_detected)
        is_dark = bool(mean_luminance < 40.0)

        if is_wet:
            score = 0.65
            desc = "Monsoon/rain condition: optical vision degraded; acoustic sonar required for valid depth."
        elif is_dark:
            score = 0.70
            desc = f"Low scene illuminance ({mean_luminance:.1f} lux): optical sensors penalized; acoustic sonar selected."
        else:
            score = 1.00
            desc = f"Clear dry daylight ({mean_luminance:.1f} lux): high optical fidelity for camera and LiDAR."

        return score, desc

    def analyze_evidence(
        self,
        damage_class: str,
        visual_confidence: Optional[float],
        surface_area_sqm: float,
        effective_depth_cm: float,
        active_modality: str,
        z_accel_g: float,
        rain_detected: bool = False,
        mean_luminance: float = 65.0
    ) -> EvidenceSummary:
        """
        Executes complete multi-sensor evidence synthesis and disagreement diagnosis.
        """
        # 1. Dimension evaluations
        c_vis, vis_desc = self.evaluate_visual_evidence(damage_class, visual_confidence, surface_area_sqm)
        c_dep, void_conf, dep_desc = self.evaluate_depth_evidence(damage_class, effective_depth_cm, active_modality)
        c_imp, imp_conf, regime, imp_desc = self.evaluate_impact_evidence(z_accel_g)
        c_env, env_desc = self.evaluate_environment_evidence(rain_detected, mean_luminance, active_modality)

        # 2. Dynamic weight allocation based on environmental state
        if rain_detected or mean_luminance < 40.0:
            # Environmental challenge: down-weight optical vision, prioritize acoustic sonar & accelerometer
            w_vis, w_dep, w_imp = 0.20, 0.50, 0.30
        else:
            # Optimal conditions: balanced multi-modal weighting
            w_vis, w_dep, w_imp = 0.35, 0.40, 0.25

        fused_score = round(w_vis * c_vis + w_dep * c_dep + w_imp * c_imp, 4)

        # 3. Categorical Disagreement Diagnosis
        if rain_detected and void_conf and imp_conf and "sonar" in active_modality.lower():
            category = DisagreementCategory.ACOUSTIC_SALVAGE_IN_RAIN
            rationale = (
                "Environmental Acoustic Salvage: Rain / standing water degraded optical reflectivity, "
                "but acoustic sonar successfully confirmed physical depth void, reinforced by acute vertical shock."
            )
        elif void_conf and imp_conf:
            category = DisagreementCategory.CONCORDANT_CONFIRMED
            rationale = (
                "Incontrovertible Multimodal Concordance: High visual distress corroborated by "
                f"physical cavity depth ({effective_depth_cm:.1f} cm) and confirmed dynamic wheel strike ({z_accel_g:.2f}g > 1.50g)."
            )
        elif c_vis >= 0.70 and void_conf and not imp_conf:
            category = DisagreementCategory.VISUALLY_DETECTED_PHYSICALLY_WEAK
            rationale = (
                "Visually & Geometrically Detected, but Dynamic Shock Weak: Camera and depth confirm "
                f"cavity ({effective_depth_cm:.1f} cm), but vertical acceleration is nominal ({z_accel_g:.2f}g <= 1.50g). "
                "Indicates vehicle straddled defect or traversed at conservative speed."
            )
        elif c_vis >= 0.70 and not void_conf and not imp_conf:
            category = DisagreementCategory.SUPERFICIAL_VISUAL_ONLY
            rationale = (
                "Superficial Visual Distress Only: Camera detects high visual distress, but physical depth "
                f"is negligible ({effective_depth_cm:.1f} cm) and chassis vibration is nominal ({z_accel_g:.2f}g). "
                "Classified as non-structural surface mark, oil stain, or optical shadow artifact."
            )
        elif imp_conf and c_vis < 0.50:
            category = DisagreementCategory.DYNAMIC_IMPACT_UNCONFIRMED_VISUALLY
            rationale = (
                "Dynamic Shock Without Clear Visual Confirmation: Accelerometer registered acute impact shock "
                f"({z_accel_g:.2f}g > 1.50g), but camera confidence is low or occluded."
            )
        else:
            category = DisagreementCategory.CONCORDANT_NOMINAL
            rationale = "Minor distress: visual, depth, and acceleration observations all indicate low severity."

        return EvidenceSummary(
            visual_confidence=c_vis,
            depth_evidence={
                "effective_depth_cm": effective_depth_cm,
                "void_confirmed": void_conf,
                "depth_score": c_dep,
                "active_modality": active_modality,
                "assessment": dep_desc
            },
            impact_evidence={
                "z_accel_g": z_accel_g,
                "impact_confirmed": imp_conf,
                "impact_score": c_imp,
                "impact_regime": regime,
                "assessment": imp_desc
            },
            environment_evidence={
                "rain_detected": rain_detected,
                "mean_luminance": mean_luminance,
                "environment_score": c_env,
                "assessment": env_desc
            },
            multimodal_confidence_score=fused_score,
            disagreement_category=category,
            fusion_rationale=rationale
        )
