"""
Simulation Runner - Execute EnergyPlus simulations.

Handles:
- Single simulation execution
- Parallel batch execution
- Error handling and retries
- Progress tracking
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    idf_path: Path
    output_dir: Path
    success: bool
    runtime_seconds: float
    error_message: Optional[str] = None

    # Parsed results (populated by ResultsParser)
    heating_kwh: Optional[float] = None
    cooling_kwh: Optional[float] = None
    lighting_kwh: Optional[float] = None
    equipment_kwh: Optional[float] = None
    fan_kwh: Optional[float] = None
    total_electricity_kwh: Optional[float] = None


class SimulationRunner:
    """
    Run EnergyPlus simulations.

    Usage:
        runner = SimulationRunner()

        # Single simulation
        result = runner.run(
            idf_path=Path('./model.idf'),
            weather_path=Path('./weather.epw'),
            output_dir=Path('./output')
        )

        # Batch simulation
        results = runner.run_batch(
            idf_paths=[...],
            weather_path=Path('./weather.epw'),
            output_base=Path('./scenarios'),
            parallel=4
        )
    """

    def __init__(self, energyplus_path: Optional[str] = None):
        """
        Initialize runner.

        Args:
            energyplus_path: Path to EnergyPlus executable
        """
        self.energyplus_path = energyplus_path or self._find_energyplus()

    def run(
        self,
        idf_path: Path,
        weather_path: Path,
        output_dir: Path,
        timeout_seconds: int = 600
    ) -> SimulationResult:
        """
        Run a single simulation.

        Args:
            idf_path: Path to IDF file
            weather_path: Path to weather file
            output_dir: Directory for outputs
            timeout_seconds: Maximum runtime

        Returns:
            SimulationResult with success status and paths
        """
        import time
        start_time = time.time()

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.energyplus_path,
            '-w', str(weather_path),
            '-d', str(output_dir),
            str(idf_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )

            runtime = time.time() - start_time

            # Check for success
            err_file = output_dir / 'eplusout.err'
            success = result.returncode == 0 and err_file.exists()

            error_msg = None
            if not success:
                error_msg = result.stderr or f"Exit code: {result.returncode}"

            return SimulationResult(
                idf_path=idf_path,
                output_dir=output_dir,
                success=success,
                runtime_seconds=runtime,
                error_message=error_msg
            )

        except subprocess.TimeoutExpired:
            return SimulationResult(
                idf_path=idf_path,
                output_dir=output_dir,
                success=False,
                runtime_seconds=timeout_seconds,
                error_message="Simulation timed out"
            )

        except Exception as e:
            return SimulationResult(
                idf_path=idf_path,
                output_dir=output_dir,
                success=False,
                runtime_seconds=time.time() - start_time,
                error_message=str(e)
            )

    def run_batch(
        self,
        idf_paths: List[Path],
        weather_path: Path,
        output_base: Path,
        parallel: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[SimulationResult]:
        """
        Run multiple simulations in parallel.

        Args:
            idf_paths: List of IDF files to simulate
            weather_path: Path to weather file
            output_base: Base directory for outputs
            parallel: Number of parallel processes
            progress_callback: Called with (completed, total)

        Returns:
            List of SimulationResult objects
        """
        results = []
        total = len(idf_paths)

        with ProcessPoolExecutor(max_workers=parallel) as executor:
            # Submit all jobs
            futures = {}
            for idf_path in idf_paths:
                output_dir = output_base / idf_path.stem
                future = executor.submit(
                    self.run, idf_path, weather_path, output_dir
                )
                futures[future] = idf_path

            # Collect results
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return results

    def _find_energyplus(self) -> str:
        """Auto-detect EnergyPlus installation."""
        candidates = [
            '/usr/local/EnergyPlus-25-1-0/energyplus',
            '/Applications/EnergyPlus-25-1-0/energyplus',
            shutil.which('energyplus'),
        ]
        for path in candidates:
            if path and Path(path).exists():
                return path
        raise RuntimeError(
            "EnergyPlus not found. Install EnergyPlus 25.1.0 or specify path."
        )
