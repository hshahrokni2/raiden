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
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from .results import ResultsParser, AnnualResults

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    idf_path: Path
    output_dir: Path
    success: bool
    runtime_seconds: float
    error_message: Optional[str] = None

    # Parsed results (populated by ResultsParser)
    parsed_results: Optional[AnnualResults] = None


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

        Raises:
            FileNotFoundError: If IDF or weather file doesn't exist
            ValueError: If file paths are invalid
        """
        import time
        start_time = time.time()

        # Validate input files exist
        self._validate_input_files(idf_path, weather_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running simulation: {idf_path.name}")

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
                logger.error(f"Simulation failed: {idf_path.name} - {error_msg}")
            else:
                logger.info(f"Simulation complete: {idf_path.name} ({runtime:.1f}s)")

            return SimulationResult(
                idf_path=idf_path,
                output_dir=output_dir,
                success=success,
                runtime_seconds=runtime,
                error_message=error_msg
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Simulation timed out: {idf_path.name} (>{timeout_seconds}s)")
            return SimulationResult(
                idf_path=idf_path,
                output_dir=output_dir,
                success=False,
                runtime_seconds=timeout_seconds,
                error_message="Simulation timed out"
            )

        except Exception as e:
            logger.error(f"Simulation error: {idf_path.name} - {e}", exc_info=True)
            return SimulationResult(
                idf_path=idf_path,
                output_dir=output_dir,
                success=False,
                runtime_seconds=time.time() - start_time,
                error_message=str(e)
            )

    def _validate_input_files(self, idf_path: Path, weather_path: Path) -> None:
        """
        Validate that input files exist and are valid.

        Args:
            idf_path: Path to IDF file
            weather_path: Path to weather file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file has wrong extension
        """
        idf_path = Path(idf_path)
        weather_path = Path(weather_path)

        # Check IDF file
        if not idf_path.exists():
            raise FileNotFoundError(f"IDF file not found: {idf_path}")
        if not idf_path.is_file():
            raise ValueError(f"IDF path is not a file: {idf_path}")
        if idf_path.suffix.lower() not in ('.idf', '.imf'):
            logger.warning(f"IDF file has unexpected extension: {idf_path.suffix}")

        # Check weather file
        if not weather_path.exists():
            raise FileNotFoundError(f"Weather file not found: {weather_path}")
        if not weather_path.is_file():
            raise ValueError(f"Weather path is not a file: {weather_path}")
        if weather_path.suffix.lower() != '.epw':
            logger.warning(f"Weather file has unexpected extension: {weather_path.suffix}")

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

        Raises:
            FileNotFoundError: If weather file doesn't exist
            ValueError: If idf_paths is empty
        """
        if not idf_paths:
            raise ValueError("No IDF files provided for batch simulation")

        # Validate weather file (IDF files validated individually in run())
        weather_path = Path(weather_path)
        if not weather_path.exists():
            raise FileNotFoundError(f"Weather file not found: {weather_path}")

        results = []
        total = len(idf_paths)
        logger.info(f"Starting batch simulation: {total} models, {parallel} parallel workers")

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

        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        logger.info(f"Batch complete: {successful}/{total} successful, {failed} failed")

        return results

    def run_and_parse(
        self,
        idf_path: Path,
        weather_path: Path,
        output_dir: Path,
        timeout_seconds: int = 600
    ) -> SimulationResult:
        """
        Run simulation and parse results.

        Args:
            idf_path: Path to IDF file
            weather_path: Path to weather file
            output_dir: Directory for outputs
            timeout_seconds: Maximum runtime

        Returns:
            SimulationResult with parsed results
        """
        result = self.run(idf_path, weather_path, output_dir, timeout_seconds)

        if result.success:
            parser = ResultsParser()
            result.parsed_results = parser.parse(output_dir)

        return result

    def _find_energyplus(self) -> str:
        """Auto-detect EnergyPlus installation."""
        candidates = [
            '/usr/local/EnergyPlus-25-1-0/energyplus',
            '/usr/local/EnergyPlus-25-1-0/energyplus-25.1.0',
            '/Applications/EnergyPlus-25-1-0/energyplus',
            '/Applications/EnergyPlus-25-1-0/energyplus-25.1.0',
            shutil.which('energyplus'),
            shutil.which('energyplus-25.1.0'),
        ]
        for path in candidates:
            if path and Path(path).exists():
                logger.debug(f"Found EnergyPlus at: {path}")
                return path
        raise RuntimeError(
            "EnergyPlus not found. Install EnergyPlus 25.1.0 or specify path."
        )


def run_simulation(
    idf_path: Path,
    weather_path: Path,
    output_dir: Path,
    parse_results: bool = True
) -> SimulationResult:
    """
    Convenience function to run a simulation.

    Args:
        idf_path: Path to IDF file
        weather_path: Path to weather file
        output_dir: Directory for outputs
        parse_results: Whether to parse results after

    Returns:
        SimulationResult
    """
    runner = SimulationRunner()
    if parse_results:
        return runner.run_and_parse(idf_path, weather_path, output_dir)
    return runner.run(idf_path, weather_path, output_dir)
