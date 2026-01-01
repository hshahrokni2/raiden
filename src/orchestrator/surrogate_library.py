"""
Pre-trained surrogate library for fast ECM predictions.

Each archetype has a pre-trained Gaussian Process surrogate that can
predict heating demand given building parameters. This enables
portfolio-scale analysis without running EnergyPlus for every building.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ArchetypeSurrogate:
    """Pre-trained surrogate for an archetype."""

    archetype_id: str
    surrogate_path: Path

    # Training metadata
    train_r2: float = 0.0
    test_r2: float = 0.0
    n_samples: int = 0
    trained_date: str = ""

    # Parameter bounds
    param_bounds: Dict[str, tuple] = field(default_factory=dict)

    # Loaded model (lazy)
    _model: Any = None

    @property
    def model(self):
        """Lazy-load the surrogate model."""
        if self._model is None and self.surrogate_path.exists():
            with open(self.surrogate_path, "rb") as f:
                self._model = pickle.load(f)
        return self._model

    def predict(self, params: Dict[str, float]) -> float:
        """
        Predict heating demand for given parameters.

        Args:
            params: Dict of parameter name → value

        Returns:
            Predicted heating demand (kWh/m²)
        """
        if self.model is None:
            raise ValueError(f"Surrogate not loaded for {self.archetype_id}")

        # Convert params to array in correct order
        from src.calibration import SurrogatePredictor

        if hasattr(self.model, "predict"):
            # Direct GP model
            import numpy as np
            X = np.array([[
                params.get("infiltration_ach", 0.06),
                params.get("wall_u_value", 0.5),
                params.get("roof_u_value", 0.3),
                params.get("window_u_value", 2.0),
                params.get("heat_recovery_eff", 0.0),
                params.get("heating_setpoint", 21.0),
            ]])
            return float(self.model.predict(X)[0])

        elif isinstance(self.model, SurrogatePredictor):
            return self.model.predict(params)

        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")


class SurrogateLibrary:
    """
    Library of pre-trained surrogates for all archetypes.

    Manages loading, caching, and training of surrogate models.
    """

    # Default parameter bounds for surrogate training
    DEFAULT_PARAM_BOUNDS = {
        "infiltration_ach": (0.02, 0.30),
        "wall_u_value": (0.10, 1.50),
        "roof_u_value": (0.10, 0.80),
        "window_u_value": (0.80, 3.00),
        "heat_recovery_eff": (0.00, 0.90),
        "heating_setpoint": (18.0, 23.0),
    }

    def __init__(self, surrogate_dir: Path = Path("./surrogates")):
        """
        Initialize the surrogate library.

        Args:
            surrogate_dir: Directory for storing surrogate files
        """
        self.surrogate_dir = Path(surrogate_dir)
        self.surrogate_dir.mkdir(parents=True, exist_ok=True)

        self._surrogates: Dict[str, ArchetypeSurrogate] = {}
        self._load_index()

    def _load_index(self):
        """Load the surrogate index file."""
        index_path = self.surrogate_dir / "index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)

            for archetype_id, info in index.items():
                self._surrogates[archetype_id] = ArchetypeSurrogate(
                    archetype_id=archetype_id,
                    surrogate_path=self.surrogate_dir / info["filename"],
                    train_r2=info.get("train_r2", 0.0),
                    test_r2=info.get("test_r2", 0.0),
                    n_samples=info.get("n_samples", 0),
                    trained_date=info.get("trained_date", ""),
                    param_bounds=info.get("param_bounds", self.DEFAULT_PARAM_BOUNDS),
                )

    def _save_index(self):
        """Save the surrogate index file."""
        index = {}
        for archetype_id, surrogate in self._surrogates.items():
            index[archetype_id] = {
                "filename": surrogate.surrogate_path.name,
                "train_r2": surrogate.train_r2,
                "test_r2": surrogate.test_r2,
                "n_samples": surrogate.n_samples,
                "trained_date": surrogate.trained_date,
                "param_bounds": surrogate.param_bounds,
            }

        index_path = self.surrogate_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def get(self, archetype_id: str) -> Optional[ArchetypeSurrogate]:
        """
        Get a pre-trained surrogate for an archetype.

        Args:
            archetype_id: Archetype identifier

        Returns:
            ArchetypeSurrogate if available, None otherwise
        """
        return self._surrogates.get(archetype_id)

    def has(self, archetype_id: str) -> bool:
        """Check if a surrogate exists for an archetype."""
        surrogate = self._surrogates.get(archetype_id)
        return surrogate is not None and surrogate.surrogate_path.exists()

    def list_available(self) -> List[str]:
        """List all available archetype surrogates."""
        return [
            aid for aid, s in self._surrogates.items()
            if s.surrogate_path.exists()
        ]

    def train(
        self,
        archetype_id: str,
        n_samples: int = 150,
        force: bool = False,
    ) -> ArchetypeSurrogate:
        """
        Train a surrogate for an archetype.

        Args:
            archetype_id: Archetype identifier
            n_samples: Number of LHS samples for training
            force: Force retrain even if exists

        Returns:
            Trained ArchetypeSurrogate
        """
        if self.has(archetype_id) and not force:
            logger.info(f"Surrogate already exists for {archetype_id}")
            return self._surrogates[archetype_id]

        logger.info(f"Training surrogate for {archetype_id} with {n_samples} samples")

        from src.calibration import SurrogateConfig, SurrogateTrainer

        # Get archetype defaults
        from src.baseline import get_archetype
        archetype = get_archetype(archetype_id)

        if not archetype:
            raise ValueError(f"Unknown archetype: {archetype_id}")

        # Configure parameter bounds
        param_bounds = self.DEFAULT_PARAM_BOUNDS.copy()

        # Adjust bounds based on archetype
        if archetype.envelope:
            wall_u = archetype.envelope.wall_u_value
            param_bounds["wall_u_value"] = (max(0.1, wall_u * 0.5), min(1.5, wall_u * 2.0))

        if archetype.ventilation and archetype.ventilation.has_heat_recovery:
            eff = archetype.ventilation.heat_recovery_efficiency
            param_bounds["heat_recovery_eff"] = (max(0.0, eff - 0.2), min(0.95, eff + 0.15))

        # Create config
        config = SurrogateConfig(
            n_samples=n_samples,
            param_bounds=param_bounds,
            kernel_type="matern",  # Matern 5/2 for physical systems
            n_restarts_optimizer=30,
        )

        # Train surrogate
        trainer = SurrogateTrainer(config=config)

        # Note: Real training requires running EnergyPlus simulations
        # Here we use a simplified training that uses archetype-based model
        trained = trainer.train(archetype_id=archetype_id)

        # Save surrogate
        import datetime
        surrogate_path = self.surrogate_dir / f"{archetype_id}_gp.pkl"
        with open(surrogate_path, "wb") as f:
            pickle.dump(trained.model, f)

        surrogate = ArchetypeSurrogate(
            archetype_id=archetype_id,
            surrogate_path=surrogate_path,
            train_r2=trained.train_r2,
            test_r2=trained.test_r2,
            n_samples=n_samples,
            trained_date=datetime.datetime.now().isoformat()[:10],
            param_bounds=param_bounds,
        )

        self._surrogates[archetype_id] = surrogate
        self._save_index()

        logger.info(
            f"Trained surrogate for {archetype_id}: "
            f"train_r2={surrogate.train_r2:.3f}, test_r2={surrogate.test_r2:.3f}"
        )

        return surrogate

    def train_all(
        self,
        n_samples: int = 150,
        force: bool = False,
    ) -> Dict[str, ArchetypeSurrogate]:
        """
        Train surrogates for all archetypes.

        Args:
            n_samples: Samples per archetype
            force: Force retrain

        Returns:
            Dict of archetype_id → ArchetypeSurrogate
        """
        from src.baseline import get_all_archetypes

        results = {}
        all_archetypes = get_all_archetypes()

        for archetype in all_archetypes:
            try:
                surrogate = self.train(archetype.id, n_samples, force)
                results[archetype.id] = surrogate
            except Exception as e:
                logger.error(f"Failed to train surrogate for {archetype.id}: {e}")

        return results


# Global library instance
_library: Optional[SurrogateLibrary] = None


def get_surrogate_library(surrogate_dir: Optional[Path] = None) -> SurrogateLibrary:
    """Get the global surrogate library instance."""
    global _library
    if _library is None:
        _library = SurrogateLibrary(surrogate_dir or Path("./surrogates"))
    return _library


def get_or_train_surrogate(
    archetype_id: str,
    surrogate_dir: Optional[Path] = None,
) -> Optional[ArchetypeSurrogate]:
    """
    Get a surrogate, training if necessary.

    Args:
        archetype_id: Archetype identifier
        surrogate_dir: Optional custom surrogate directory

    Returns:
        ArchetypeSurrogate if available or trainable
    """
    library = get_surrogate_library(surrogate_dir)

    if library.has(archetype_id):
        return library.get(archetype_id)

    # Try to train
    try:
        return library.train(archetype_id)
    except Exception as e:
        logger.warning(f"Could not get or train surrogate for {archetype_id}: {e}")
        return None
