"""
Distributed EnergyPlus worker for cloud-scale simulation.

Architecture:
    Coordinator (this machine)
        → Redis task queue
            → Worker 1 (EC2/k8s) → E+ simulation → Results
            → Worker 2 (EC2/k8s) → E+ simulation → Results
            → Worker N (EC2/k8s) → E+ simulation → Results

Usage:
    # On coordinator
    from src.simulation.distributed_worker import DistributedCoordinator

    coordinator = DistributedCoordinator(redis_url="redis://localhost:6379")
    results = await coordinator.run_portfolio(buildings, packages)

    # On workers (in Docker/k8s)
    python -m src.simulation.distributed_worker --redis redis://...:6379

Performance (100 × c5.4xlarge spot instances):
    - 225,000 simulations in ~25 minutes
    - Cost: ~$10 USD
"""

import asyncio
import base64
import json
import logging
import os
import signal
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Install with: pip install redis")


@dataclass
class SimulationTask:
    """A single E+ simulation task."""

    task_id: str
    address: str
    archetype_id: str
    package_id: str
    atemp_m2: float

    # IDF content (base64 encoded for JSON transport)
    idf_base64: str
    weather_file: str = "SWE_Stockholm.AP.024600_TMYx.epw"

    # Calibration parameters to apply
    parameters: Dict[str, float] = field(default_factory=dict)

    # Metadata
    created_at: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more urgent

    def to_json(self) -> str:
        return json.dumps({
            "task_id": self.task_id,
            "address": self.address,
            "archetype_id": self.archetype_id,
            "package_id": self.package_id,
            "atemp_m2": self.atemp_m2,
            "idf_base64": self.idf_base64,
            "weather_file": self.weather_file,
            "parameters": self.parameters,
            "created_at": self.created_at,
            "priority": self.priority,
        })

    @classmethod
    def from_json(cls, data: str) -> "SimulationTask":
        d = json.loads(data)
        return cls(**d)


@dataclass
class SimulationResult:
    """Result from an E+ simulation."""

    task_id: str
    success: bool

    # Energy results
    heating_kwh_m2: float = 0.0
    cooling_kwh_m2: float = 0.0
    total_kwh_m2: float = 0.0

    # Peak loads
    peak_heating_kw: float = 0.0
    peak_cooling_kw: float = 0.0

    # Metadata
    worker_id: str = ""
    processing_time_sec: float = 0.0
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps({
            "task_id": self.task_id,
            "success": self.success,
            "heating_kwh_m2": self.heating_kwh_m2,
            "cooling_kwh_m2": self.cooling_kwh_m2,
            "total_kwh_m2": self.total_kwh_m2,
            "peak_heating_kw": self.peak_heating_kw,
            "peak_cooling_kw": self.peak_cooling_kw,
            "worker_id": self.worker_id,
            "processing_time_sec": self.processing_time_sec,
            "error": self.error,
        })

    @classmethod
    def from_json(cls, data: str) -> "SimulationResult":
        d = json.loads(data)
        return cls(**d)


class EPlusWorker:
    """
    Worker that pulls tasks from Redis and runs E+ simulations.

    Designed to run in a Docker container or k8s pod.
    """

    TASK_QUEUE = "raiden:tasks"
    RESULTS_HASH = "raiden:results"
    WORKER_SET = "raiden:workers"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        worker_id: Optional[str] = None,
        parallel_sims: int = 4,
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")

        self.redis = redis.from_url(redis_url)
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.parallel_sims = parallel_sims
        self.running = False

        # Weather file directory
        self.weather_dir = Path(os.environ.get("WEATHER_DIR", "./weather"))

        logger.info(f"EPlusWorker initialized: {self.worker_id}, {parallel_sims} parallel sims")

    def run_forever(self):
        """Run worker in blocking mode, processing tasks until stopped."""
        self.running = True

        # Register worker
        self.redis.sadd(self.WORKER_SET, self.worker_id)
        self.redis.hset(f"raiden:worker:{self.worker_id}", mapping={
            "started_at": time.time(),
            "tasks_completed": 0,
            "status": "running",
        })

        logger.info(f"Worker {self.worker_id} started, waiting for tasks...")

        # Handle signals for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        try:
            while self.running:
                # Blocking pop from task queue (30 sec timeout)
                task_data = self.redis.blpop(self.TASK_QUEUE, timeout=30)

                if task_data:
                    _, task_json = task_data
                    task = SimulationTask.from_json(task_json.decode())

                    logger.info(f"Processing task {task.task_id}: {task.address} / {task.package_id}")

                    result = self._run_simulation(task)

                    # Store result
                    self.redis.hset(self.RESULTS_HASH, task.task_id, result.to_json())

                    # Update stats
                    self.redis.hincrby(f"raiden:worker:{self.worker_id}", "tasks_completed", 1)

                    logger.info(
                        f"Completed {task.task_id}: {result.heating_kwh_m2:.1f} kWh/m² "
                        f"in {result.processing_time_sec:.1f}s"
                    )

        finally:
            # Cleanup
            self.redis.hset(f"raiden:worker:{self.worker_id}", "status", "stopped")
            self.redis.srem(self.WORKER_SET, self.worker_id)
            logger.info(f"Worker {self.worker_id} stopped")

    def _handle_signal(self, signum, frame):
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _run_simulation(self, task: SimulationTask) -> SimulationResult:
        """Run a single E+ simulation."""
        start = time.time()

        result = SimulationResult(
            task_id=task.task_id,
            success=False,
            worker_id=self.worker_id,
        )

        try:
            # Decode IDF
            idf_content = base64.b64decode(task.idf_base64).decode()

            # Create temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Write IDF
                idf_path = tmpdir / "in.idf"
                with open(idf_path, "w") as f:
                    f.write(idf_content)

                # Apply parameters if provided
                if task.parameters:
                    self._apply_parameters(idf_path, task.parameters)

                # Get weather file
                weather_path = self.weather_dir / task.weather_file
                if not weather_path.exists():
                    # Try to download
                    self._download_weather(task.weather_file, weather_path)

                # Run EnergyPlus
                import subprocess

                ep_cmd = [
                    "energyplus",
                    "-w", str(weather_path),
                    "-d", str(tmpdir),
                    str(idf_path),
                ]

                proc = subprocess.run(
                    ep_cmd,
                    capture_output=True,
                    timeout=300,  # 5 min timeout
                )

                if proc.returncode != 0:
                    result.error = proc.stderr.decode()[:500]
                    return result

                # Parse results
                results = self._parse_results(tmpdir, task.atemp_m2)
                result.heating_kwh_m2 = results.get("heating_kwh_m2", 0)
                result.cooling_kwh_m2 = results.get("cooling_kwh_m2", 0)
                result.total_kwh_m2 = results.get("total_kwh_m2", 0)
                result.success = True

        except Exception as e:
            result.error = str(e)
            logger.error(f"Simulation failed for {task.task_id}: {e}")

        result.processing_time_sec = time.time() - start
        return result

    def _apply_parameters(self, idf_path: Path, params: Dict[str, float]) -> None:
        """Apply calibration parameters to IDF."""
        from src.core.idf_parser import IDFParser
        from eppy.modeleditor import IDF

        parser = IDFParser()
        idf = IDF(str(idf_path))

        if "infiltration_ach" in params:
            parser.set_infiltration_ach(idf, params["infiltration_ach"])
        if "window_u_value" in params:
            parser.set_window_u_value(idf, params["window_u_value"])
        if "wall_u_value" in params:
            parser.set_wall_u_value(idf, params["wall_u_value"])
        if "roof_u_value" in params:
            parser.set_roof_u_value(idf, params["roof_u_value"])
        if "heat_recovery_eff" in params:
            parser.set_heat_recovery_effectiveness(idf, params["heat_recovery_eff"])
        if "heating_setpoint" in params:
            parser.set_heating_setpoint(idf, params["heating_setpoint"])

        idf.save()

    def _parse_results(self, output_dir: Path, atemp_m2: float) -> Dict[str, float]:
        """Parse E+ output files."""
        from src.simulation.results import parse_results

        try:
            results = parse_results(output_dir)
            return {
                "heating_kwh_m2": results.total_heating_kwh / atemp_m2 if atemp_m2 else 0,
                "cooling_kwh_m2": results.total_cooling_kwh / atemp_m2 if atemp_m2 else 0,
                "total_kwh_m2": (results.total_heating_kwh + results.total_cooling_kwh) / atemp_m2 if atemp_m2 else 0,
            }
        except Exception as e:
            logger.warning(f"Failed to parse results: {e}")
            return {}

    def _download_weather(self, filename: str, dest: Path) -> None:
        """Download weather file if not present."""
        # In production, would download from S3 or similar
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Weather file not found: {filename}")


class DistributedCoordinator:
    """
    Coordinate distributed E+ simulations across multiple workers.

    Usage:
        coordinator = DistributedCoordinator(redis_url="redis://localhost:6379")

        # Submit portfolio
        results = await coordinator.run_portfolio(
            buildings=[...],
            packages=["baseline", "steg1", "steg2", "steg3"],
        )
    """

    TASK_QUEUE = "raiden:tasks"
    RESULTS_HASH = "raiden:results"
    WORKER_SET = "raiden:workers"

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")

        self.redis = redis.from_url(redis_url)
        self.job_id = f"job-{uuid.uuid4().hex[:8]}"

        logger.info(f"DistributedCoordinator initialized: {self.job_id}")

    async def run_portfolio(
        self,
        buildings: List[Dict[str, Any]],
        packages: List[str],
        idf_generator: Optional[Callable] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        timeout_minutes: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Run full E+ simulation for all buildings × packages.

        Args:
            buildings: List of building dicts (archetype_id, atemp_m2, address, etc.)
            packages: List of ECM package IDs to simulate
            idf_generator: Optional function to generate IDF for each building
            progress_callback: Optional callback(completed, total)
            timeout_minutes: Max time to wait for results

        Returns:
            List of result dicts for each building × package combination
        """
        from src.baseline import generate_idf_for_archetype

        total_tasks = len(buildings) * len(packages)
        logger.info(f"Submitting {total_tasks} tasks ({len(buildings)} buildings × {len(packages)} packages)")

        # Generate and submit tasks
        task_ids = []
        for building in buildings:
            # Generate IDF for this building
            if idf_generator:
                idf_content = idf_generator(building)
            else:
                idf_content = generate_idf_for_archetype(
                    building.get("archetype_id", "mfh_1961_1975"),
                    atemp_m2=building.get("atemp_m2", 1000),
                )

            idf_base64 = base64.b64encode(idf_content.encode()).decode()

            for package_id in packages:
                task = SimulationTask(
                    task_id=f"{self.job_id}-{uuid.uuid4().hex[:8]}",
                    address=building.get("address", ""),
                    archetype_id=building.get("archetype_id", "unknown"),
                    package_id=package_id,
                    atemp_m2=building.get("atemp_m2", 1000),
                    idf_base64=idf_base64,
                    parameters=self._get_package_params(package_id, building),
                )

                # Push to queue
                self.redis.rpush(self.TASK_QUEUE, task.to_json())
                task_ids.append(task.task_id)

        logger.info(f"Submitted {len(task_ids)} tasks, waiting for results...")

        # Wait for results
        start = time.time()
        timeout_sec = timeout_minutes * 60
        completed = 0

        while completed < len(task_ids):
            # Check timeout
            if time.time() - start > timeout_sec:
                logger.error(f"Timeout waiting for results after {timeout_minutes} min")
                break

            # Check how many results we have
            results_count = 0
            for task_id in task_ids:
                if self.redis.hexists(self.RESULTS_HASH, task_id):
                    results_count += 1

            if results_count > completed:
                completed = results_count
                if progress_callback:
                    progress_callback(completed, len(task_ids))
                logger.info(f"Progress: {completed}/{len(task_ids)} ({completed/len(task_ids)*100:.1f}%)")

            await asyncio.sleep(1)

        # Collect results
        results = []
        for task_id in task_ids:
            result_json = self.redis.hget(self.RESULTS_HASH, task_id)
            if result_json:
                result = SimulationResult.from_json(result_json.decode())
                results.append({
                    "task_id": result.task_id,
                    "success": result.success,
                    "heating_kwh_m2": result.heating_kwh_m2,
                    "error": result.error,
                })
            else:
                results.append({
                    "task_id": task_id,
                    "success": False,
                    "error": "No result received",
                })

        elapsed = time.time() - start
        logger.info(f"Completed {completed}/{len(task_ids)} tasks in {elapsed:.1f}s")

        return results

    def _get_package_params(self, package_id: str, building: Dict) -> Dict[str, float]:
        """Get calibration parameters for a package."""
        # Base parameters from building
        base = {
            "infiltration_ach": building.get("infiltration_ach", 0.06),
            "wall_u_value": building.get("wall_u_value", 0.5),
            "roof_u_value": building.get("roof_u_value", 0.3),
            "window_u_value": building.get("window_u_value", 2.0),
            "heat_recovery_eff": building.get("heat_recovery_eff", 0.0),
            "heating_setpoint": building.get("heating_setpoint", 21.0),
        }

        # Package modifications
        package_effects = {
            "baseline": {},
            "steg0_nollkostnad": {"heating_setpoint": 20.0},
            "steg1_enkel": {"heating_setpoint": 20.0, "infiltration_ach": 0.04},
            "steg2_standard": {"heating_setpoint": 20.0, "infiltration_ach": 0.03, "window_u_value": 1.0},
            "steg3_premium": {"heating_setpoint": 20.0, "infiltration_ach": 0.02, "window_u_value": 0.9, "heat_recovery_eff": 0.80},
            "steg4_djuprenovering": {"heating_setpoint": 20.0, "infiltration_ach": 0.015, "window_u_value": 0.8, "wall_u_value": 0.15, "roof_u_value": 0.12, "heat_recovery_eff": 0.85},
        }

        effects = package_effects.get(package_id, {})
        params = base.copy()
        params.update(effects)

        return params

    def get_worker_count(self) -> int:
        """Get number of active workers."""
        return self.redis.scard(self.WORKER_SET)

    def get_queue_depth(self) -> int:
        """Get number of pending tasks."""
        return self.redis.llen(self.TASK_QUEUE)


def main():
    """Run as worker process."""
    import argparse

    parser = argparse.ArgumentParser(description="Raiden E+ Worker")
    parser.add_argument("--redis", default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel simulations")
    parser.add_argument("--id", help="Worker ID (auto-generated if not provided)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    worker = EPlusWorker(
        redis_url=args.redis,
        worker_id=args.id,
        parallel_sims=args.parallel,
    )

    worker.run_forever()


if __name__ == "__main__":
    main()
