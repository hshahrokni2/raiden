"""
ASHRAE Guideline 14 Calibration Metrics.

Implements standard calibration quality metrics for building energy models:
- NMBE (Normalized Mean Bias Error): Measures systematic bias
- CVRMSE (Coefficient of Variation of RMSE): Measures variability

Reference: ASHRAE Guideline 14-2014
- Monthly: NMBE < ±10%, CVRMSE < 30%
- Hourly: NMBE < ±5%, CVRMSE < 15%

Usage:
    from src.calibration.metrics import CalibrationMetrics

    # With monthly data
    metrics = CalibrationMetrics.from_monthly_data(measured, simulated)
    print(f"NMBE: {metrics.nmbe:.1f}%")
    print(f"CVRMSE: {metrics.cvrmse:.1f}%")
    print(f"Passes ASHRAE: {metrics.passes_ashrae_monthly}")

    # With annual data only
    metrics = CalibrationMetrics.from_annual_data(53.0, 51.8)
    print(f"Error: {metrics.annual_error_percent:.1f}%")
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """
    ASHRAE Guideline 14 calibration metrics.

    Attributes:
        nmbe: Normalized Mean Bias Error (%)
        cvrmse: Coefficient of Variation of RMSE (%)
        r_squared: Coefficient of determination
        annual_error_percent: |measured - simulated| / measured × 100
        data_resolution: 'monthly', 'hourly', or 'annual'
        n_points: Number of data points used
        passes_ashrae: Whether model passes ASHRAE criteria
        data_quality_warning: Warning about data quality limitations
        calibration_confidence: Confidence level (0-1) based on data quality
    """

    # Core metrics
    nmbe: float = 0.0
    cvrmse: float = 0.0
    r_squared: float = 0.0

    # Annual comparison
    measured_total: float = 0.0
    simulated_total: float = 0.0
    annual_error_percent: float = 0.0

    # Metadata
    data_resolution: str = "annual"
    n_points: int = 1

    # ASHRAE thresholds
    ashrae_nmbe_limit: float = 10.0  # Monthly default
    ashrae_cvrmse_limit: float = 30.0  # Monthly default

    # Data quality assessment
    data_quality_warning: str = ""
    calibration_confidence: float = 0.5  # 0-1 scale

    @property
    def passes_ashrae_nmbe(self) -> bool:
        """Check if NMBE passes ASHRAE threshold."""
        return abs(self.nmbe) <= self.ashrae_nmbe_limit

    @property
    def passes_ashrae_cvrmse(self) -> bool:
        """Check if CVRMSE passes ASHRAE threshold."""
        return self.cvrmse <= self.ashrae_cvrmse_limit

    @property
    def passes_ashrae(self) -> bool:
        """Check if model passes both ASHRAE criteria."""
        return self.passes_ashrae_nmbe and self.passes_ashrae_cvrmse

    @property
    def passes_ashrae_monthly(self) -> bool:
        """Check against monthly thresholds (NMBE < 10%, CVRMSE < 30%)."""
        return abs(self.nmbe) <= 10.0 and self.cvrmse <= 30.0

    @property
    def passes_ashrae_hourly(self) -> bool:
        """Check against hourly thresholds (NMBE < 5%, CVRMSE < 15%)."""
        return abs(self.nmbe) <= 5.0 and self.cvrmse <= 15.0

    @classmethod
    def from_monthly_data(
        cls,
        measured: List[float],
        simulated: List[float],
    ) -> "CalibrationMetrics":
        """
        Compute metrics from monthly energy data.

        Args:
            measured: 12 monthly measured values (kWh or kWh/m²)
            simulated: 12 monthly simulated values

        Returns:
            CalibrationMetrics with NMBE, CVRMSE, etc.
        """
        m = np.array(measured)
        s = np.array(simulated)

        if len(m) != len(s):
            raise ValueError(f"Measured ({len(m)}) and simulated ({len(s)}) must have same length")

        n = len(m)
        mean_m = np.mean(m)

        if mean_m == 0:
            logger.warning("Mean measured value is 0, cannot compute normalized metrics")
            return cls(
                data_resolution="monthly",
                n_points=n,
                measured_total=np.sum(m),
                simulated_total=np.sum(s),
            )

        # NMBE: Normalized Mean Bias Error
        # NMBE = Σ(Mi - Si) / (n × M_mean) × 100
        residuals = m - s
        nmbe = (np.sum(residuals) / (n * mean_m)) * 100

        # CVRMSE: Coefficient of Variation of RMSE
        # CVRMSE = sqrt(Σ(Mi - Si)² / n) / M_mean × 100
        rmse = np.sqrt(np.mean(residuals ** 2))
        cvrmse = (rmse / mean_m) * 100

        # R²: Coefficient of determination
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((m - mean_m) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Annual totals
        measured_total = np.sum(m)
        simulated_total = np.sum(s)
        annual_error = abs(measured_total - simulated_total) / measured_total * 100 if measured_total > 0 else 0

        # Confidence for monthly data is higher
        if abs(nmbe) <= 10 and cvrmse <= 30:
            confidence = 0.85  # Passes ASHRAE monthly
        elif abs(nmbe) <= 15 or cvrmse <= 40:
            confidence = 0.7  # Close to ASHRAE
        else:
            confidence = 0.5  # Fails ASHRAE

        return cls(
            nmbe=nmbe,
            cvrmse=cvrmse,
            r_squared=r_squared,
            measured_total=measured_total,
            simulated_total=simulated_total,
            annual_error_percent=annual_error,
            data_resolution="monthly",
            n_points=n,
            ashrae_nmbe_limit=10.0,  # Monthly threshold
            ashrae_cvrmse_limit=30.0,
            data_quality_warning="",  # No warning for monthly data
            calibration_confidence=confidence,
        )

    @classmethod
    def from_hourly_data(
        cls,
        measured: List[float],
        simulated: List[float],
    ) -> "CalibrationMetrics":
        """
        Compute metrics from hourly energy data.

        Args:
            measured: 8760 hourly measured values
            simulated: 8760 hourly simulated values

        Returns:
            CalibrationMetrics with NMBE, CVRMSE using hourly thresholds
        """
        m = np.array(measured)
        s = np.array(simulated)

        if len(m) != len(s):
            raise ValueError(f"Measured ({len(m)}) and simulated ({len(s)}) must have same length")

        n = len(m)
        mean_m = np.mean(m)

        if mean_m == 0:
            logger.warning("Mean measured value is 0, cannot compute normalized metrics")
            return cls(
                data_resolution="hourly",
                n_points=n,
                ashrae_nmbe_limit=5.0,
                ashrae_cvrmse_limit=15.0,
            )

        # Same formulas as monthly
        residuals = m - s
        nmbe = (np.sum(residuals) / (n * mean_m)) * 100
        rmse = np.sqrt(np.mean(residuals ** 2))
        cvrmse = (rmse / mean_m) * 100

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((m - mean_m) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        measured_total = np.sum(m)
        simulated_total = np.sum(s)
        annual_error = abs(measured_total - simulated_total) / measured_total * 100 if measured_total > 0 else 0

        # Confidence for hourly data is highest
        if abs(nmbe) <= 5 and cvrmse <= 15:
            confidence = 0.95  # Passes ASHRAE hourly
        elif abs(nmbe) <= 10 and cvrmse <= 30:
            confidence = 0.85  # Passes monthly threshold
        else:
            confidence = 0.6  # Fails both

        return cls(
            nmbe=nmbe,
            cvrmse=cvrmse,
            r_squared=r_squared,
            measured_total=measured_total,
            simulated_total=simulated_total,
            annual_error_percent=annual_error,
            data_resolution="hourly",
            n_points=n,
            ashrae_nmbe_limit=5.0,  # Hourly threshold (stricter)
            ashrae_cvrmse_limit=15.0,
            data_quality_warning="",  # No warning for hourly data
            calibration_confidence=confidence,
        )

    @classmethod
    def from_annual_data(
        cls,
        measured_kwh_m2: float,
        simulated_kwh_m2: float,
    ) -> "CalibrationMetrics":
        """
        Compute metrics from annual totals only.

        Note: With only annual data, NMBE = annual error and CVRMSE = |annual error|
        (no variability information available).

        IMPORTANT: Annual-only calibration has significant limitations:
        - Cannot distinguish "lucky calibration" from "accurate model"
        - R² is undefined with 1 data point
        - Model could have large monthly errors that cancel out
        - Recommended: Use monthly/hourly data when available

        Args:
            measured_kwh_m2: Measured annual heating (kWh/m²)
            simulated_kwh_m2: Simulated annual heating (kWh/m²)

        Returns:
            CalibrationMetrics with annual error and data quality warning
        """
        if measured_kwh_m2 == 0:
            logger.warning("Measured annual value is 0, cannot compute error")
            return cls(
                data_resolution="annual",
                n_points=1,
                data_quality_warning="⚠️ Measured value is 0 - cannot validate model",
                calibration_confidence=0.0,
            )

        # With only 1 point, NMBE = relative error
        error = measured_kwh_m2 - simulated_kwh_m2
        nmbe = (error / measured_kwh_m2) * 100  # Can be positive or negative

        # With 1 point, CVRMSE = |NMBE| (no variability)
        cvrmse = abs(nmbe)

        annual_error = abs(error) / measured_kwh_m2 * 100

        # Data quality warning for annual-only calibration
        warning = (
            "⚠️ ANNUAL DATA ONLY: Calibration based on single annual energy value. "
            "ASHRAE Guideline 14 recommends monthly or hourly data for reliable calibration. "
            "With only annual data: (1) R² cannot be computed, "
            "(2) Monthly errors may cancel out, (3) Model accuracy is uncertain. "
            "For production use, request monthly utility bills or sub-metering data."
        )

        # Confidence based on annual error magnitude
        # Low error = higher confidence, but never above 0.6 for annual-only
        if annual_error <= 5:
            confidence = 0.6  # Good annual match, but limited by data
        elif annual_error <= 10:
            confidence = 0.5  # Acceptable match
        elif annual_error <= 15:
            confidence = 0.4  # Marginal match
        else:
            confidence = 0.2  # Poor match

        return cls(
            nmbe=nmbe,
            cvrmse=cvrmse,
            r_squared=0.0,  # Not meaningful with 1 point
            measured_total=measured_kwh_m2,
            simulated_total=simulated_kwh_m2,
            annual_error_percent=annual_error,
            data_resolution="annual",
            n_points=1,
            ashrae_nmbe_limit=10.0,  # Use monthly threshold as reference
            ashrae_cvrmse_limit=30.0,
            data_quality_warning=warning,
            calibration_confidence=confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics to dictionary."""
        return {
            "nmbe_percent": round(self.nmbe, 2),
            "cvrmse_percent": round(self.cvrmse, 2),
            "r_squared": round(self.r_squared, 4),
            "annual_error_percent": round(self.annual_error_percent, 2),
            "measured_total": round(self.measured_total, 2),
            "simulated_total": round(self.simulated_total, 2),
            "data_resolution": self.data_resolution,
            "n_points": self.n_points,
            "passes_ashrae": self.passes_ashrae,
            "passes_ashrae_nmbe": self.passes_ashrae_nmbe,
            "passes_ashrae_cvrmse": self.passes_ashrae_cvrmse,
            "ashrae_nmbe_limit": self.ashrae_nmbe_limit,
            "ashrae_cvrmse_limit": self.ashrae_cvrmse_limit,
            "data_quality_warning": self.data_quality_warning,
            "calibration_confidence": round(self.calibration_confidence, 2),
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        status = "✓ PASSES" if self.passes_ashrae else "✗ FAILS"
        conf_label = self._confidence_label()

        lines = [
            f"ASHRAE Guideline 14 Metrics ({self.data_resolution}, n={self.n_points}):",
            f"  NMBE:   {self.nmbe:+.2f}% (limit: ±{self.ashrae_nmbe_limit}%) {'✓' if self.passes_ashrae_nmbe else '✗'}",
            f"  CVRMSE: {self.cvrmse:.2f}% (limit: {self.ashrae_cvrmse_limit}%) {'✓' if self.passes_ashrae_cvrmse else '✗'}",
            f"  R²:     {self.r_squared:.4f}",
            f"  Annual: {self.annual_error_percent:.1f}% error",
            f"  Confidence: {self.calibration_confidence:.0%} ({conf_label})",
            f"  Status: {status}",
        ]

        # Add warning if present
        if self.data_quality_warning:
            lines.append("")
            lines.append(self.data_quality_warning)

        return "\n".join(lines)

    def _confidence_label(self) -> str:
        """Get human-readable confidence label."""
        if self.calibration_confidence >= 0.8:
            return "High - Hourly/monthly data"
        elif self.calibration_confidence >= 0.6:
            return "Medium - Good annual match"
        elif self.calibration_confidence >= 0.4:
            return "Low - Annual only"
        else:
            return "Very Low - Poor match or limited data"


def compute_uncertainty_adjusted_metrics(
    measured_kwh_m2: float,
    simulated_kwh_m2: float,
    simulated_std: float,
) -> Tuple[CalibrationMetrics, float]:
    """
    Compute metrics with uncertainty consideration.

    When the simulated value has uncertainty (from Bayesian calibration),
    we can compute a probability that the model passes ASHRAE criteria.

    Args:
        measured_kwh_m2: Measured annual heating
        simulated_kwh_m2: Calibrated prediction mean
        simulated_std: Calibrated prediction standard deviation

    Returns:
        Tuple of (CalibrationMetrics, probability of passing ASHRAE)
    """
    from scipy import stats

    metrics = CalibrationMetrics.from_annual_data(measured_kwh_m2, simulated_kwh_m2)

    if simulated_std <= 0:
        # No uncertainty, binary pass/fail
        prob_pass = 1.0 if metrics.passes_ashrae else 0.0
        return metrics, prob_pass

    # Error threshold for ASHRAE (10% of measured)
    threshold = measured_kwh_m2 * (metrics.ashrae_nmbe_limit / 100)

    # Probability that |simulated - measured| < threshold
    # Using normal distribution assumption
    lower = measured_kwh_m2 - threshold
    upper = measured_kwh_m2 + threshold

    prob_pass = stats.norm.cdf(upper, loc=simulated_kwh_m2, scale=simulated_std) - \
                stats.norm.cdf(lower, loc=simulated_kwh_m2, scale=simulated_std)

    return metrics, prob_pass
